//! End to end: source -> analysis (cached) -> moments -> rendered vertical clips.
use crate::ingest::{DlProgress, HeatmapPoint};
use crate::llm::Llm;
use crate::media::{Encoder, pick_encoder, probe};
use crate::settings::{DEFAULT_NICHE, Settings};
use crate::picker::{self, Moment, PickOpts};
use crate::render::{self, Layout, RenderOpts};
use crate::transcribe::Segment;
use crate::{fastdl, ingest, paths, signals, transcribe};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Options {
    pub out: PathBuf,
    pub max_clips: usize,
    pub min_len: f64,
    pub max_len: f64,
    pub target_len: f64,
    pub layout: Layout,
    pub asr: String, // whisper | none
    pub asr_model: String,
    pub language: Option<String>,
    pub llm: String, // none | gemini | lmstudio | ollama | openrouter
    pub llm_model: Option<String>,
    pub vision: bool,
    pub niche: String,
    pub subs: bool,
    pub hook: bool,
    pub normalize_audio: bool,
    pub embed_tag: bool,
    pub codec: String,
    pub quality: String,
    pub software: bool,
    pub workers: usize,
    pub fresh: bool,
    /// Light mode for machines without a dedicated GPU: 30 fps output, one render at a time.
    pub lite: bool,
    /// URLs only: set to turn on long-video handling (`long_mode`). In "chunks" mode it is also
    /// the window length in seconds.
    pub chunk_len: Option<f64>,
    /// "scan" (default): pick from the whole video by its audio, fetch only the picked moments.
    /// "chunks": fetch, analyse and render window by window.
    pub long_mode: Option<String>,
    /// Seconds shared by neighbouring chunks so a moment on a boundary is seen whole (default 30).
    pub chunk_overlap: Option<f64>,
    /// Process the most-replayed chunks first when YouTube has a replay heatmap (default true).
    pub heatmap_prioritize: Option<bool>,
    /// Pause after picking so the user can untick, retitle and trim moments before rendering.
    pub review: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            out: crate::platform::default_out_dir(),
            max_clips: 8,
            min_len: 15.0,
            max_len: 60.0,
            target_len: 30.0,
            layout: Layout::default(),
            asr: "whisper".into(),
            asr_model: "turbo".into(),
            language: None,
            llm: "none".into(),
            llm_model: None,
            vision: false,
            niche: DEFAULT_NICHE.into(),
            subs: true,
            hook: true,
            normalize_audio: true,
            embed_tag: true,
            codec: "h264".into(),
            quality: "high".into(),
            software: false,
            workers: 3,
            fresh: false,
            lite: false,
            chunk_len: std::env::var("CLIPFORGE_CHUNK_LEN").ok().and_then(|s| s.parse().ok()),
            chunk_overlap: None,
            long_mode: None,
            heatmap_prioritize: None,
            review: false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Event {
    Stage(&'static str),
    Log(String),
    Progress(DlProgress),
    /// Picked moments wait for the user; the approved (possibly edited) list comes back on `reply`.
    Review { moments: Vec<Moment>, preselect: usize, reply: std::sync::mpsc::Sender<Vec<Moment>> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClipMeta {
    pub file: Option<String>,
    #[serde(flatten)]
    pub moment: Moment,
    pub caption: String,
}

pub struct Outcome {
    pub out_dir: PathBuf,
    pub clips: Vec<ClipMeta>,
}

fn cache_key(p: &Path) -> Result<String> {
    let md = std::fs::metadata(p)?;
    let mtime = md.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_secs();
    let raw = format!("{}|{}|{}", p.canonicalize()?.display(), md.len(), mtime);
    let digest = Sha1::digest(raw.as_bytes());
    Ok(digest.iter().take(8).map(|b| format!("{b:02x}")).collect())
}

impl Options {
    fn render_opts(&self) -> RenderOpts {
        RenderOpts {
            subs: self.subs,
            hook: self.hook,
            normalize_audio: self.normalize_audio,
            embed_tag: self.embed_tag,
            workers: if self.lite { 1 } else { self.workers },
            max_fps: if self.lite { 30 } else { 60 },
        }
    }

    /// The ranking model, if one is chosen and configured. A missing key or unknown provider
    /// must not cost the user a long run: log it and rank on signals, like a failed API call.
    fn llm(&self, log: &dyn Fn(&str)) -> Option<Llm> {
        match self.llm.as_str() {
            "none" | "" => None,
            provider => match Llm::new(provider, self.llm_model.as_deref(), self.vision, &Settings::load()) {
                Ok(l) => Some(l),
                Err(e) => {
                    log(&format!("AI ranking off: {e:#}; ranking on signals only"));
                    None
                }
            },
        }
    }
}

/// The whole video: a local file as is; a URL over parallel range requests when its streams
/// allow it (yt-dlp's own download runs at ~4 MB/s on YouTube), else through yt-dlp.
fn fetch_video(source: &str, cache_root: &Path, emit: &(dyn Fn(Event) + Sync)) -> Result<PathBuf> {
    let progress = |p| emit(Event::Progress(p));
    if ingest::is_url(source) {
        match fetch_video_fast(source, cache_root, &progress) {
            Ok(Some(p)) => return Ok(p),
            Ok(None) => {}
            Err(e) => {
                crate::cancel::check()?;
                emit(Event::Log(format!("parallel download failed ({e:#}); using yt-dlp")));
            }
        }
    }
    ingest::fetch(source, cache_root, &progress)
}

fn fetch_video_fast(source: &str, cache_root: &Path, progress: &dyn Fn(DlProgress)) -> Result<Option<PathBuf>> {
    let Some((id, v, a)) = ingest::resolve_streams(source)? else { return Ok(None) };
    // ids are case-sensitive; keep them, just drop anything unsafe in a file name
    let id: String = id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').take(64).collect();
    if id.is_empty() {
        return Ok(None);
    }
    let dl = cache_root.join("downloads");
    std::fs::create_dir_all(&dl)?;
    // same name yt-dlp uses, so either path reuses the other's download
    let dest = dl.join(format!("{id}.mp4"));
    if dest.is_file() {
        return Ok(Some(dest));
    }
    let (vi, ai) = (fastdl::read_index(&v)?, fastdl::read_index(&a)?);
    let tmp = dl.join(format!("{id}.parts"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    let part = tmp.join("whole.mp4");
    let result = fastdl::fetch_range((&v, &vi), (&a, &ai), 0.0, f64::MAX / 4.0, &tmp, &part, progress)
        .and_then(|_| Ok(std::fs::rename(&part, &dest)?));
    let _ = std::fs::remove_dir_all(&tmp);
    result.map(|_| Some(dest))
}

/// Hand the picked moments to the user and wait for the approved list (cancel still works).
/// Edited times are clamped to the video; moments under a second are dropped.
fn review(emit: &(dyn Fn(Event) + Sync), moments: Vec<Moment>, preselect: usize, duration: f64) -> Result<Vec<Moment>> {
    let (tx, rx) = std::sync::mpsc::channel();
    emit(Event::Review { moments, preselect, reply: tx });
    loop {
        crate::cancel::check()?;
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(approved) => {
                let approved: Vec<Moment> = approved
                    .into_iter()
                    .map(|mut m| {
                        m.start = m.start.clamp(0.0, duration);
                        m.end = m.end.clamp(m.start, duration);
                        m.peak = m.peak.clamp(m.start, m.end);
                        m
                    })
                    .filter(|m| m.duration() >= 1.0)
                    .collect();
                if approved.is_empty() {
                    bail!("no moments were approved");
                }
                emit(Event::Log(format!("{} moments approved", approved.len())));
                return Ok(approved);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => bail!("review was abandoned"),
        }
    }
}

/// Announce a stage; also where a cancelled job stops.
fn stage(emit: &(dyn Fn(Event) + Sync), name: &'static str) -> Result<()> {
    crate::cancel::check()?;
    emit(Event::Stage(name));
    Ok(())
}

pub fn process(source: &str, opt: &Options, emit: &(dyn Fn(Event) + Sync)) -> Result<Outcome> {
    if let Some(len) = opt.chunk_len.filter(|l| *l > 0.0) {
        if ingest::is_url(source) {
            return match opt.long_mode.as_deref() {
                Some("chunks") => process_chunked(source, opt, emit, len),
                _ => process_scan(source, opt, emit, len),
            };
        }
        emit(Event::Log("long-video mode is for URLs; local files run in one pass".into()));
    }
    process_whole(source, opt, emit)
}

/// Download (or open) the whole video, analyse it once, render the best moments.
fn process_whole(source: &str, opt: &Options, emit: &(dyn Fn(Event) + Sync)) -> Result<Outcome> {
    let t0 = Instant::now();
    let log = |m: &str| emit(Event::Log(m.to_string()));
    let cache_root = paths::cache_dir();
    std::fs::create_dir_all(&cache_root)?;

    stage(emit, "fetch")?;
    let video = fetch_video(source, &cache_root, emit)?;
    let info = probe(&video)?;
    let work = cache_root.join(cache_key(&video)?);
    std::fs::create_dir_all(&work)?;
    if opt.fresh {
        for e in std::fs::read_dir(&work)?.flatten() {
            if e.path().extension().is_some_and(|x| x == "json") {
                std::fs::remove_file(e.path())?;
            }
        }
    }
    let enc = pick_encoder(&opt.codec, &opt.quality, opt.software);
    log(&format!(
        "{}  {}x{} @ {:.0}fps  {:.1} min  encoder: {}",
        video.file_name().unwrap_or_default().to_string_lossy(),
        info.width, info.height, info.fps, info.duration / 60.0, enc.name
    ));

    stage(emit, "audio")?;
    let wav = if info.has_audio { Some(ingest::extract_audio(&video, &work.join("audio.wav"))?) } else { None };

    stage(emit, "transcribe")?;
    let mut segs: Vec<Segment> = vec![];
    if let (Some(w), true) = (&wav, opt.asr != "none") {
        let cache = work.join(format!("transcript_{}.json", opt.asr_model));
        match transcribe::transcribe(w, &cache, &opt.asr_model, opt.language.as_deref(), &log) {
            Ok(s) => segs = s,
            Err(e) => log(&format!("transcription failed ({e}); continuing without speech")),
        }
    }

    stage(emit, "analyse")?;
    let sig = signals::compute(&video, wav.as_deref(), info.duration, &enc, &work.join("signals.json"))?;

    stage(emit, "pick")?;
    let llm = opt.llm(&log);
    let moments_cache = work.join(format!(
        "moments_{}_{}_{}_{}{}.json",
        opt.llm, opt.max_clips, opt.min_len as i64, opt.max_len as i64, if opt.vision { "_v" } else { "" }
    ));
    let pick_opts = PickOpts {
        max_clips: opt.max_clips, min_len: opt.min_len, max_len: opt.max_len, target: opt.target_len,
        llm: llm.as_ref(), vision: opt.vision, niche: &opt.niche, video: &video, enc: &enc, bias: &[],
    };
    let moments = picker::pick(&sig, &segs, info.duration, &pick_opts, Some(&moments_cache), &log)?;
    if moments.is_empty() {
        bail!("no moments found");
    }
    let moments = if opt.review { review(emit, moments, opt.max_clips, info.duration)? } else { moments };

    let stem = video.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    let out_dir = opt.out.join(render::slug(&stem, 60));
    stage(emit, "render")?;
    log(&format!("rendering {} clips -> {} ({})", moments.len(), out_dir.display(), opt.layout.mode));
    for (i, m) in moments.iter().enumerate() {
        log(&format!("{:02}. {:7.1}s-{:7.1}s ({:4.1}s) {}", i + 1, m.start, m.end, m.duration(), m.title));
    }
    let paths = render::render_all(&moments, &info, &segs, &out_dir, &enc, &opt.layout, &opt.render_opts(), &log)?;

    let clips: Vec<ClipMeta> = moments
        .into_iter()
        .enumerate()
        .map(|(i, m)| {
            let prefix = format!("{:02}_", i + 1);
            let file = paths
                .iter()
                .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .find(|n| n.starts_with(&prefix));
            let caption = format!("{} {}", m.title, m.hashtags.join(" ")).trim().to_string();
            ClipMeta { file, moment: m, caption }
        })
        .collect();
    std::fs::write(out_dir.join("clips.json"), serde_json::to_string_pretty(&clips)?)?;
    log(&format!("{} clips in {:.0}s -> {}", paths.len(), t0.elapsed().as_secs_f64(), out_dir.display()));
    Ok(Outcome { out_dir, clips })
}

/// One download window of a chunked run.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkPlan {
    /// 1-based position in time order.
    pub idx: usize,
    pub start: f64,
    pub len: f64,
    pub heatmap_score: f64,
    /// Entirely inside the intro or outro margin.
    pub is_bait: bool,
}

/// Intro and outro margin: 4% of the video, between 1 and 5 minutes. Replay spikes there are
/// mostly people skipping the intro or jumping to the end, not the best moments.
fn bait_margin(duration: f64) -> f64 {
    (duration * 0.04).clamp(60.0, 300.0)
}

/// Overlapping windows over `[0, duration)`. A tail under a minute joins the window before it.
/// With a heatmap and `prioritize`, non-bait windows come first, most replayed first; otherwise
/// windows stay in time order.
pub fn plan_chunks(duration: f64, chunk_len: f64, overlap: f64, heatmap: &[HeatmapPoint], prioritize: bool) -> Vec<ChunkPlan> {
    if duration <= 0.0 || chunk_len <= 0.0 {
        return vec![];
    }
    let step = (chunk_len - overlap).max(10.0);
    let margin = bait_margin(duration);
    let (intro_end, outro_start) = (margin, (duration - margin).max(margin));
    let mut plans: Vec<ChunkPlan> = vec![];
    let mut start = 0.0;
    loop {
        let mut end = (start + chunk_len).min(duration);
        if duration - end < 60.0 {
            end = duration;
        }
        let (mut weighted, mut peak) = (0.0, 0.0f64);
        for p in heatmap {
            let bait = p.start_time < intro_end || p.end_time > outro_start;
            let shared = p.end_time.min(end) - p.start_time.max(start);
            if !bait && shared > 0.0 {
                weighted += p.value * shared;
                peak = peak.max(p.value);
            }
        }
        plans.push(ChunkPlan {
            idx: plans.len() + 1,
            start,
            len: end - start,
            heatmap_score: peak * 0.6 + weighted / (end - start).max(1.0) * 0.4,
            is_bait: end <= intro_end || start >= outro_start,
        });
        if end >= duration {
            break;
        }
        start += step;
    }
    if prioritize && !heatmap.is_empty() {
        plans.sort_by(|a, b| {
            a.is_bait.cmp(&b.is_bait).then(b.heatmap_score.total_cmp(&a.heatmap_score)).then(a.idx.cmp(&b.idx))
        });
    }
    plans
}

fn hms(t: f64) -> String {
    let t = t.max(0.0) as u64;
    if t >= 3600 { format!("{}:{:02}:{:02}", t / 3600, t / 60 % 60, t % 60) } else { format!("{}:{:02}", t / 60, t % 60) }
}

/// Direct stream URLs plus their indexes, for parallel range downloads (`fastdl`).
struct FastSource {
    video: (fastdl::Stream, fastdl::Index),
    audio: (fastdl::Stream, fastdl::Index),
    at: Instant,
}

/// googlevideo URLs expire after ~6 h; refresh well before that.
const FAST_SOURCE_TTL: Duration = Duration::from_secs(3600);

fn fast_source(url: &str) -> Result<Option<FastSource>> {
    let Some((_, v, a)) = ingest::resolve_streams(url)? else { return Ok(None) };
    let (vi, ai) = (fastdl::read_index(&v)?, fastdl::read_index(&a)?);
    Ok(Some(FastSource { video: (v, vi), audio: (a, ai), at: Instant::now() }))
}

/// Stops a chunked run after this many failed chunks in a row (network down, video removed).
const MAX_CHUNK_FAILURES: usize = 3;

/// Long URLs: fetch one window at a time, analyse and render it, delete it. Disk and RAM stay at
/// one window's worth instead of the whole video. Falls back to one pass when the length is
/// unknown (live streams) or the video fits in a single window.
fn process_chunked(source: &str, opt: &Options, emit: &(dyn Fn(Event) + Sync), chunk_len: f64) -> Result<Outcome> {
    let t0 = Instant::now();
    let log = |m: &str| emit(Event::Log(m.to_string()));
    stage(emit, "fetch")?;
    let meta = ingest::probe_meta(source)?;
    let duration = match meta.duration {
        Some(d) if d > 0.0 && meta.is_live != Some(true) => d,
        _ => {
            log("chunked processing needs a finished video of known length; downloading in one pass");
            return process_whole(source, opt, emit);
        }
    };
    let chunk_len = chunk_len.clamp((opt.max_len * 2.0).max(120.0), 3600.0);
    if duration <= chunk_len + 60.0 {
        log(&format!("video is {} long, shorter than one chunk; downloading in one pass", hms(duration)));
        return process_whole(source, opt, emit);
    }
    let overlap = opt.chunk_overlap.unwrap_or(30.0).clamp(0.0, chunk_len / 2.0);
    let prioritize = opt.heatmap_prioritize.unwrap_or(true);
    let plans = plan_chunks(duration, chunk_len, overlap, &meta.heatmap, prioritize);
    let title = if meta.title.trim().is_empty() { "video".to_string() } else { meta.title.clone() };
    let enc = pick_encoder(&opt.codec, &opt.quality, opt.software);
    log(&format!(
        "\"{title}\" ({})  {}  {} chunks of {} ({}s overlap)  encoder: {}",
        meta.id, hms(duration), plans.len(), hms(chunk_len), overlap as u64, enc.name
    ));
    if prioritize && !meta.heatmap.is_empty() {
        log("replay heatmap found: most-replayed chunks first, intro and outro last");
    } else if prioritize {
        log("no replay heatmap for this video; chunks in time order");
    }

    if opt.review {
        log("review needs every moment up front; chunk-by-chunk mode renders as it goes");
    }
    let llm = opt.llm(&log);
    let ro = opt.render_opts();
    let out_dir = opt.out.join(render::slug(&title, 60));
    let digest = Sha1::digest(source.as_bytes());
    let key: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    let work_root = paths::cache_dir().join(format!("chunks_{key}"));
    let mut clips: Vec<ClipMeta> = vec![];
    let mut failures = 0;
    let mut fast: Option<FastSource> = None;
    let mut fast_ok = true;
    for (n, plan) in plans.iter().enumerate() {
        if clips.len() >= opt.max_clips || crate::cancel::requested() {
            break;
        }
        if fast_ok && fast.as_ref().is_none_or(|f| f.at.elapsed() > FAST_SOURCE_TTL) {
            match fast_source(source) {
                Ok(Some(f)) => {
                    if fast.is_none() {
                        log(&format!("parallel range downloads: {} connections", fastdl::CONNECTIONS));
                    }
                    fast = Some(f);
                }
                Ok(None) => {
                    log("this video has no separate HTTPS streams; chunks download through yt-dlp");
                    fast_ok = false;
                }
                Err(e) => {
                    log(&format!("parallel downloads unavailable ({e:#}); chunks download through yt-dlp"));
                    fast_ok = false;
                }
            }
        }
        let label = format!("chunk {}/{} ({}-{})", n + 1, plans.len(), hms(plan.start), hms(plan.start + plan.len));
        log(&format!("--- {label} ---"));
        let work = work_root.join(format!("{:.0}", plan.start));
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work)?;
        let ctx = ChunkCtx {
            source, opt, enc: &enc, llm: llm.as_ref(), ro: &ro, out_dir: &out_dir, work: &work, fast: fast.as_ref(),
        };
        let result = process_chunk(&ctx, plan, &clips, emit);
        let _ = std::fs::remove_dir_all(&work);
        match result {
            Ok(mut new) => {
                failures = 0;
                clips.append(&mut new);
                // keep clips.json current so a crash late in a long run still leaves usable output
                let _ = serde_json::to_string_pretty(&clips).map(|t| std::fs::write(out_dir.join("clips.json"), t));
            }
            Err(e) => {
                failures += 1;
                log(&format!("{label} skipped: {e:#}"));
                if failures >= MAX_CHUNK_FAILURES {
                    log(&format!("{MAX_CHUNK_FAILURES} chunks in a row failed; stopping"));
                    break;
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&work_root);
    crate::cancel::check()?;
    if clips.is_empty() {
        bail!("no moments found in any chunk");
    }
    clips.sort_by(|a, b| a.moment.start.total_cmp(&b.moment.start));
    std::fs::write(out_dir.join("clips.json"), serde_json::to_string_pretty(&clips)?)?;
    log(&format!("{} clips in {:.0}s -> {}", clips.len(), t0.elapsed().as_secs_f64(), out_dir.display()));
    Ok(Outcome { out_dir, clips })
}

struct ChunkCtx<'a> {
    source: &'a str,
    opt: &'a Options,
    enc: &'a Encoder,
    llm: Option<&'a Llm>,
    ro: &'a RenderOpts,
    out_dir: &'a Path,
    work: &'a Path,
    fast: Option<&'a FastSource>,
}

/// `[start, start + len)` of a URL as an mp4, plus where it starts in the source when known
/// exactly. Parallel range requests when possible, else yt-dlp.
fn fetch_piece(
    source: &str, fast: Option<&FastSource>, start: f64, len: f64, dir: &Path, emit: &(dyn Fn(Event) + Sync),
) -> Result<(PathBuf, Option<f64>)> {
    let progress = |p| emit(Event::Progress(p));
    if let Some(f) = fast {
        let dest = dir.join("piece.mp4");
        let video = (&f.video.0, &f.video.1);
        let audio = (&f.audio.0, &f.audio.1);
        match fastdl::fetch_range(video, audio, start, len, dir, &dest, &progress) {
            Ok(at) => return Ok((dest, Some(at))),
            Err(e) => {
                crate::cancel::check()?;
                emit(Event::Log(format!("parallel download failed ({e:#}); retrying through yt-dlp")));
            }
        }
    }
    Ok((ingest::fetch_chunk(source, start, len, dir, &progress)?, None))
}

/// Where a fetched piece starts in the source: exact from the index, else estimated from the
/// keyframe lead-in yt-dlp adds at the front.
fn piece_offset(requested: f64, _len: f64, _got: f64, exact: Option<f64>) -> f64 {
    exact.unwrap_or(requested)
}

/// Fetch, analyse, pick and render one window. Returns its clips on the full video's timeline.
fn process_chunk(c: &ChunkCtx, plan: &ChunkPlan, accepted: &[ClipMeta], emit: &(dyn Fn(Event) + Sync)) -> Result<Vec<ClipMeta>> {
    let log = |m: &str| emit(Event::Log(m.to_string()));
    let opt = c.opt;
    stage(emit, "fetch")?;
    let (video, exact_start) = fetch_piece(c.source, c.fast, plan.start, plan.len, c.work, emit)?;
    let info = probe(&video)?;
    if info.duration < opt.min_len {
        bail!("only {:.0}s of video", info.duration);
    }
    let offset = piece_offset(plan.start, plan.len, info.duration, exact_start);

    stage(emit, "audio")?;
    let wav = if info.has_audio { Some(ingest::extract_audio(&video, &c.work.join("audio.wav"))?) } else { None };

    stage(emit, "transcribe")?;
    let mut segs: Vec<Segment> = vec![];
    if let (Some(w), true) = (&wav, opt.asr != "none") {
        let cache = c.work.join("transcript.json");
        match transcribe::transcribe(w, &cache, &opt.asr_model, opt.language.as_deref(), &log) {
            Ok(s) => segs = s,
            Err(e) => log(&format!("transcription failed ({e}); continuing without speech")),
        }
    }

    stage(emit, "analyse")?;
    let sig = signals::compute(&video, wav.as_deref(), info.duration, c.enc, &c.work.join("signals.json"))?;

    stage(emit, "pick")?;
    // a few per chunk so one busy stretch cannot take every slot; one spare for cross-chunk dedupe
    let want = opt.max_clips.saturating_sub(accepted.len()).min(3);
    let pick_opts = PickOpts {
        max_clips: want + 1, min_len: opt.min_len, max_len: opt.max_len, target: opt.target_len,
        llm: c.llm, vision: opt.vision, niche: &opt.niche, video: &video, enc: c.enc, bias: &[],
    };
    let picked = picker::pick(&sig, &segs, info.duration, &pick_opts, None, &log)?;
    let shift = |m: &Moment| Moment {
        start: ((m.start + offset) * 100.0).round() / 100.0,
        end: ((m.end + offset) * 100.0).round() / 100.0,
        peak: ((m.peak + offset) * 100.0).round() / 100.0,
        ..m.clone()
    };
    // neighbouring chunks overlap, so the same moment can be found twice
    let mut moments: Vec<Moment> = vec![];
    for (i, mut m) in picked.into_iter().enumerate() {
        let g = shift(&m);
        if accepted.iter().all(|a| !picker::overlapping(&g, &a.moment)) && moments.iter().all(|k| !picker::overlapping(&g, &shift(k))) {
            // placeholder titles count per chunk; renumber them for the whole run
            if m.title == picker::placeholder_title(&picker::game_name(&opt.niche), i + 1) {
                m.title = picker::placeholder_title(&picker::game_name(&opt.niche), accepted.len() + moments.len() + 1);
            }
            moments.push(m);
        }
    }
    moments.truncate(want);
    if moments.is_empty() {
        log("no new moments in this chunk");
        return Ok(vec![]);
    }

    stage(emit, "render")?;
    let first = accepted.len() + 1;
    for (i, m) in moments.iter().enumerate() {
        let g = shift(m);
        log(&format!("{:02}. {}-{} ({:4.1}s) {}", first + i, hms(g.start), hms(g.end), m.duration(), m.title));
    }
    let paths = render::render_from(first, &moments, &info, &segs, c.out_dir, c.enc, &opt.layout, c.ro, &log)?;
    Ok(moments
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            let prefix = format!("{:02}_", first + i);
            // a clip whose render failed is left out, so numbering stays tied to real files
            let file = paths
                .iter()
                .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .find(|n| n.starts_with(&prefix))?;
            let caption = format!("{} {}", m.title, m.hashtags.join(" ")).trim().to_string();
            Some(ClipMeta { file: Some(file), moment: shift(m), caption })
        })
        .collect())
}

/// Finished and rejected candidates of a scan run, so a re-run of the same job resumes.
#[derive(Default, Serialize, Deserialize)]
struct ScanProgress {
    done: Vec<(usize, ClipMeta)>,
    skipped: Vec<usize>,
}

/// Short hash of everything that changes the output (all options except `fresh`).
fn options_key(opt: &Options) -> String {
    let mut o = opt.clone();
    o.fresh = false;
    let digest = Sha1::digest(serde_json::to_string(&o).unwrap_or_default().as_bytes());
    digest.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

/// Chat messages per second, once the background download is done. Waits at most 45 s more;
/// past that the run goes on without chat (the download still finishes for next time).
fn chat_signal(
    file: &Path, rx: Option<std::sync::mpsc::Receiver<Result<PathBuf>>>, duration: f64, log: &dyn Fn(&str),
) -> Option<Vec<f64>> {
    if !file.is_file() {
        match rx?.recv_timeout(Duration::from_secs(45)) {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                log(&format!("{e:#}"));
                return None;
            }
            Err(_) => {
                log("chat replay is still downloading; using it the next time this video is processed");
                return None;
            }
        }
    }
    let rate = ingest::chat_rate(file, duration.ceil() as usize).ok()?;
    (rate.iter().sum::<f64>() >= 50.0).then_some(rate)
}

/// Chat spikes as a per-second bias: messages smoothed over 10 s, z-scored, and moved 6 s
/// earlier because chat reacts after the moment it is about.
pub fn chat_bias(rate: &[f64]) -> Vec<f64> {
    const LAG: usize = 6;
    const HALF: usize = 5;
    let n = rate.len();
    let smooth: Vec<f64> = (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(HALF), (i + HALF + 1).min(n));
            rate[a..b].iter().sum::<f64>() / (b - a) as f64
        })
        .collect();
    let z = signals::zscore(&smooth);
    (0..n).map(|i| 0.6 * z.get(i + LAG).copied().unwrap_or(0.0).clamp(-1.0, 3.0)).collect()
}

/// Per-second bias for scan mode: replay heatmap (z-scored) and an intro/outro penalty.
fn scan_bias(duration: f64, heatmap: &[HeatmapPoint], prioritize: bool) -> Vec<f64> {
    let secs = (duration.ceil() as usize).max(1);
    let mut bias = vec![0.0; secs];
    if !prioritize {
        return bias;
    }
    if !heatmap.is_empty() {
        let mut heat = vec![0.0; secs];
        for p in heatmap {
            let (a, b) = ((p.start_time.max(0.0) as usize).min(secs), (p.end_time.ceil() as usize).min(secs));
            heat[a..b.max(a)].fill(p.value);
        }
        for (b, z) in bias.iter_mut().zip(signals::zscore(&heat)) {
            *b += 0.5 * z;
        }
    }
    let margin = bait_margin(duration);
    for (i, b) in bias.iter_mut().enumerate() {
        if (i as f64) < margin || i as f64 > duration - margin {
            *b -= 1.0;
        }
    }
    bias
}

/// Transcribe stretches of the scan WAV, each cached on disk. Loads whisper on first need and
/// detects the language once.
fn transcribe_regions(
    regions: &[(f64, f64)], wav: &Path, work: &Path, opt: &Options, whisper: &mut Option<transcribe::Whisper>,
    log: &dyn Fn(&str),
) -> Vec<Segment> {
    // detected once per video and cached, so re-renders skip it
    let lang_file = work.join("language.txt");
    let cached = std::fs::read_to_string(&lang_file).ok().map(|l| l.trim().to_string()).filter(|l| !l.is_empty());
    let known = opt.language.clone().or(cached);
    let mut model = match whisper.as_ref() {
        Some(w) => w.model.clone(),
        None => whisper_model_for(&opt.asr_model, known.as_deref()).to_string(),
    };
    let cache = |m: &str, a: f64, b: f64| work.join(format!("tr_{m}_{a:.0}_{b:.0}.json"));
    let todo: Vec<(f64, f64)> = regions.iter().copied().filter(|(a, b)| !cache(&model, *a, *b).is_file()).collect();
    if !todo.is_empty() && whisper.is_none() {
        let secs: f64 = todo.iter().map(|(a, b)| b - a).sum();
        log(&format!("transcribing {} stretches ({} of audio)", todo.len(), hms(secs)));
        match transcribe::Whisper::load(&model, known.as_deref(), log) {
            Ok(mut w) => {
                if known.is_none() {
                    let dur = transcribe::wav_seconds(wav).unwrap_or(0.0);
                    // spread over the video, away from the intro and outro
                    let at: Vec<f64> = [0.3, 0.5, 0.15, 0.7].iter().map(|f| dur * f).collect();
                    match w.detect_language(wav, &at) {
                        Some(l) => {
                            log(&format!("language: {l}"));
                            let _ = std::fs::write(&lang_file, &l);
                            let better = whisper_model_for(&model, Some(&l));
                            if better != model {
                                log(&format!(
                                    "whisper {model} can't transcribe {l} speech reliably; using {better} (slower, readable captions)"
                                ));
                                match transcribe::Whisper::load(better, Some(&l), log) {
                                    Ok(w2) => {
                                        w = w2;
                                        model = better.to_string();
                                    }
                                    Err(e) => log(&format!("could not load {better} ({e}); staying on {model}")),
                                }
                            }
                        }
                        None => log("language unclear; detecting per window"),
                    }
                } else if model != opt.asr_model {
                    log(&format!("{} speech: using whisper {model} instead of {}", known.as_deref().unwrap_or(""), opt.asr_model));
                }
                *whisper = Some(w);
            }
            Err(e) => log(&format!("transcription failed ({e}); continuing without speech")),
        }
    } else if !todo.is_empty() {
        let secs: f64 = todo.iter().map(|(a, b)| b - a).sum();
        log(&format!("transcribing {} stretches ({} of audio)", todo.len(), hms(secs)));
    }
    let mut segs = vec![];
    for (a, b) in regions {
        let path = cache(&model, *a, *b);
        if let Some(s) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Vec<Segment>>(&t).ok()) {
            segs.extend(s);
            continue;
        }
        let Some(w) = whisper.as_ref() else { break };
        match w.range(wav, *a, *b) {
            Ok(s) => {
                let _ = serde_json::to_string(&s).map(|t| std::fs::write(&path, t));
                segs.extend(s);
            }
            Err(e) => {
                if crate::cancel::requested() {
                    break;
                }
                log(&format!("transcription of {}-{} failed ({e})", hms(*a), hms(*b)));
            }
        }
    }
    segs
}

/// `base` turns non-English speech (Hindi/Hinglish commentary over game audio, measured) into
/// junk; `small` gets it right. Other models are kept as chosen.
fn whisper_model_for<'a>(chosen: &'a str, language: Option<&str>) -> &'a str {
    match language {
        Some(l) if chosen == "base" && l != "en" => "small",
        _ => chosen,
    }
}

/// Candidate windows widened by `pad`, clamped, sorted and merged where they touch.
fn merge_regions(ms: &[Moment], pad: f64, duration: f64) -> Vec<(f64, f64)> {
    let mut r: Vec<(f64, f64)> = ms.iter().map(|m| ((m.start - pad).max(0.0), (m.end + pad).min(duration))).collect();
    r.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(f64, f64)> = vec![];
    for (a, b) in r {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// Segments overlapping `[from, to)`, moved by `-by` (source time to piece time).
fn shift_segments(segs: &[Segment], from: f64, to: f64, by: f64) -> Vec<Segment> {
    segs.iter()
        .filter(|s| s.e > from && s.s < to)
        .map(|s| {
            let mut s = s.clone();
            (s.s, s.e) = (s.s - by, s.e - by);
            for w in &mut s.words {
                (w.s, w.e) = (w.s - by, w.e - by);
            }
            s
        })
        .collect()
}

/// Long URLs, best of the whole video: download only the audio (fast, ~50 MB/hour), find the
/// loud and most-replayed stretches over the full length, transcribe only those, pick, then
/// fetch just each picked moment's video and render it. Bandwidth and disk scale with the
/// number of clips, not the length of the stream.
fn process_scan(source: &str, opt: &Options, emit: &(dyn Fn(Event) + Sync), chunk_len: f64) -> Result<Outcome> {
    let t0 = Instant::now();
    let log = |m: &str| emit(Event::Log(m.to_string()));
    stage(emit, "fetch")?;
    let meta = ingest::probe_meta(source)?;
    let duration = match meta.duration {
        Some(d) if d > 0.0 && meta.is_live != Some(true) => d,
        _ => {
            log("scanning needs a finished video of known length; downloading in one pass");
            return process_whole(source, opt, emit);
        }
    };
    if duration <= chunk_len.clamp(120.0, 3600.0) + 60.0 {
        log(&format!("video is {} long; downloading in one pass", hms(duration)));
        return process_whole(source, opt, emit);
    }
    let title = if meta.title.trim().is_empty() { "video".to_string() } else { meta.title.clone() };
    let enc = pick_encoder(&opt.codec, &opt.quality, opt.software);
    log(&format!("\"{title}\" ({})  {}  scanning the whole video  encoder: {}", meta.id, hms(duration), enc.name));
    let digest = Sha1::digest(source.as_bytes());
    let key: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    let work = paths::cache_dir().join(format!("scan_{key}"));
    std::fs::create_dir_all(&work)?;
    // chat replay: YouTube serves it in small sequential batches, so it downloads in the
    // background while the audio is fetched and scanned. Kept for the next run either way.
    let chat_file = work.join("chat.live_chat.json");
    let chat_rx = (meta.has_chat && !chat_file.is_file()).then(|| {
        log("chat replay found: downloading it alongside the audio");
        let (tx, rx) = std::sync::mpsc::channel();
        let (url, dir) = (source.to_string(), work.clone());
        std::thread::spawn(move || {
            let _ = tx.send(ingest::fetch_chat(&url, &dir));
        });
        rx
    });
    if opt.fresh {
        for e in std::fs::read_dir(&work)?.flatten() {
            if e.path().extension().is_some_and(|x| x == "json") {
                std::fs::remove_file(e.path())?;
            }
        }
    }
    let fast = match fast_source(source) {
        Ok(f) => f,
        Err(e) => {
            log(&format!("parallel downloads unavailable ({e:#}); using yt-dlp"));
            None
        }
    };

    // 1. audio of the whole video, cached as 16 kHz mono WAV
    let wav = work.join("audio.wav");
    if !wav.is_file() {
        let progress = |p| emit(Event::Progress(p));
        let fetched = match &fast {
            Some(f) => {
                let ext = if f.audio.0.ext.is_empty() { "m4a" } else { &f.audio.0.ext };
                let p = work.join(format!("audio_src.{ext}"));
                fastdl::fetch_whole(&f.audio.0, &f.audio.1, &p, &progress).map(|_| p)
            }
            None => Err(anyhow::anyhow!("no direct stream")),
        };
        let src = match fetched {
            Ok(p) => p,
            Err(_) => ingest::fetch_audio(source, &work, &progress)?,
        };
        stage(emit, "audio")?;
        let tmp = work.join("audio.part.wav");
        ingest::extract_audio_parallel(&src, &tmp)?;
        std::fs::rename(&tmp, &wav)?;
        let _ = std::fs::remove_file(&src);
    }
    stage(emit, "audio")?;
    let duration = duration.min(transcribe::wav_seconds(&wav).unwrap_or(duration));

    // 2. signals over the full length, then only the promising stretches get transcribed
    let sig = signals::audio_only(Some(&wav), duration)?;
    let prioritize = opt.heatmap_prioritize.unwrap_or(true);
    let mut bias = scan_bias(duration, &meta.heatmap, prioritize);
    if let Some(rate) = chat_signal(&chat_file, chat_rx, duration, &log) {
        for (b, c) in bias.iter_mut().zip(chat_bias(&rate)) {
            *b += c;
        }
        log("chat replay: weighting the moments chat reacted to");
    }
    if prioritize && !meta.heatmap.is_empty() {
        log("replay heatmap found: weighting the most-replayed stretches");
    }
    let llm = opt.llm(&log);
    let asr = opt.asr != "none";
    let mut whisper: Option<transcribe::Whisper> = None;
    let mut segs: Vec<Segment> = vec![];

    // Download YouTube subtitles if available (fast, perfect word accuracy)
    let subs_dir = work.join("subtitles");
    let yt_subs: Option<Vec<Segment>> = if asr && ingest::is_url(source) {
        match ingest::fetch_subtitles(source, &subs_dir) {
            Ok(Some(path)) => {
                log(&format!("using YouTube subtitles from {}", path.file_name().unwrap_or_default().to_string_lossy()));
                ingest::parse_json3_subtitles(&path).ok()
            }
            _ => None,
        }
    } else {
        None
    };

    if asr && llm.is_some() {
        if let Some(subs) = &yt_subs {
            segs = subs.clone();
        } else {
            let scouted = picker::scout(&sig, &bias, duration, opt.max_clips + 2, opt.min_len, opt.max_len, opt.target_len);
            let gist: Vec<Moment> =
                scouted.iter().map(|m| Moment { start: m.peak - 15.0, end: m.peak + 15.0, ..m.clone() }).collect();
            let regions = merge_regions(&gist, 0.0, duration);
            stage(emit, "transcribe")?;
            segs = transcribe_regions(&regions, &wav, &work, opt, &mut whisper, &log);
        }
    }

    // 3. pick from the whole timeline, with spares for candidates that turn out to be menus
    stage(emit, "pick")?;
    if opt.vision {
        log("vision ranking needs video frames; scan mode ranks on audio and speech");
    }
    // Spares cost nothing until used (audio-only picks), and loud moments in games are often
    // pause menus: on an 8 h GTA VI stream 3 of the top 5 were. One spare per clip, at least 3.
    let spares = opt.max_clips.max(3);
    let moments_cache = work.join(format!(
        "moments_{}_{}_{}_{}_{}.json",
        opt.llm, opt.max_clips + spares, opt.min_len as i64, opt.max_len as i64, if prioritize { "h" } else { "n" }
    ));
    let pick_opts = PickOpts {
        max_clips: opt.max_clips + spares, min_len: opt.min_len, max_len: opt.max_len, target: opt.target_len,
        llm: llm.as_ref(), vision: false, niche: &opt.niche, video: &wav, enc: &enc, bias: &bias,
    };
    let candidates = picker::pick(&sig, &segs, duration, &pick_opts, Some(&moments_cache), &log)?;
    if candidates.is_empty() {
        bail!("no moments found");
    }
    // with review, the user's list replaces the spares: render exactly what was approved
    let (candidates, want) = if opt.review {
        let approved = review(emit, candidates, opt.max_clips, duration)?;
        let n = approved.len();
        (approved, n)
    } else {
        (candidates, opt.max_clips)
    };

    // 4. per candidate: fetch its video (the next one downloads while this one renders), drop
    // pause menus and loading screens, transcribe just this clip, render
    stage(emit, "render")?;
    let out_dir = opt.out.join(render::slug(&title, 60));
    log(&format!("rendering {} clips -> {} ({})", want.min(candidates.len()), out_dir.display(), opt.layout.mode));
    let ro = opt.render_opts();
    let fast = std::sync::Mutex::new(fast.map(std::sync::Arc::new));
    let window = |m: &Moment| ((m.start - 1.0).max(0.0), (m.end + 1.0).min(duration));
    let piece_dir = |i: usize| work.join(format!("piece_{i}"));
    let fetch = |i: usize| -> Result<(PathBuf, Option<f64>)> {
        let (a, b) = window(&candidates[i]);
        let dir = piece_dir(i);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        let f = {
            let mut g = fast.lock().unwrap();
            if g.as_ref().is_some_and(|f| f.at.elapsed() > FAST_SOURCE_TTL) {
                *g = fast_source(source).ok().flatten().map(std::sync::Arc::new);
            }
            g.clone()
        };
        fetch_piece(source, f.as_deref(), a, b - a, &dir, emit)
    };
    let fetch = &fetch;
    // resume: clips this exact job (same options) already finished, and candidates it rejected
    let progress_file = work.join(format!("progress_{}.json", options_key(opt)));
    // a reviewed list can differ between runs, so its progress can't be matched up
    if opt.fresh || opt.review {
        let _ = std::fs::remove_file(&progress_file);
    }
    let mut progress: ScanProgress =
        std::fs::read_to_string(&progress_file).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    progress.done.retain(|(_, c)| c.file.as_ref().is_some_and(|f| out_dir.join(f).is_file()));
    let mut clips: Vec<ClipMeta> = progress.done.iter().map(|(_, c)| c.clone()).collect();
    clips.truncate(want);
    if !clips.is_empty() {
        log(&format!("resuming: {} clips already rendered", clips.len()));
    }
    let todo: Vec<usize> = (0..candidates.len())
        .filter(|i| !progress.skipped.contains(i) && !progress.done.iter().any(|(d, _)| d == i))
        .collect();
    let mut skipped = 0;
    std::thread::scope(|sc| {
        if clips.len() >= want {
            return;
        }
        let mut next = todo.first().map(|&i| sc.spawn(move || fetch(i)));
        for (k, &i) in todo.iter().enumerate() {
            let cand = &candidates[i];
            let Some(handle) = next.take() else { break };
            if crate::cancel::requested() {
                let _ = handle.join();
                break;
            }
            let fetched = handle.join().unwrap_or_else(|_| Err(anyhow::anyhow!("download thread panicked")));
            // prefetch only what we may need: if this one is accepted and fills the last slot,
            // the next download would be wasted
            if let Some(&j) = todo.get(k + 1)
                && clips.len() + 1 < want
            {
                next = Some(sc.spawn(move || fetch(j)));
            }
            let result = (|| -> Result<Option<ClipMeta>> {
                crate::cancel::check()?;
                let (video, exact) = fetched?;
                let info = probe(&video)?;
                let (a, b) = window(cand);
                let off = piece_offset(a, b - a, info.duration, exact);
                let menu = signals::menu_share(&video, cand.start - off, cand.end - off).unwrap_or(0.0);
                crate::cancel::check()?;
                if menu >= 0.3 {
                    log(&format!(
                        "skipped {}-{}: {:.0}% pause menu or loading screen",
                        hms(cand.start), hms(cand.end), menu * 100.0
                    ));
                    return Ok(None);
                }
                let n = clips.len() + 1;
                let mut m = cand.clone();
                let placeholder = picker::is_placeholder(&m.title);
                let mut local_segs = vec![];
                if asr {
                    let said = if let Some(subs) = &yt_subs {
                        transcribe::segments_in(subs, cand.start - 1.0, cand.end + 1.0)
                    } else {
                        // one whisper pass per 30 s: pad only into the slack of the last window
                        let pad = ((30.0 * (m.duration() / 30.0).ceil().max(1.0) - m.duration()) / 2.0).clamp(0.0, 3.0);
                        let region = ((m.start - pad).max(0.0), (m.end + pad).min(duration));
                        transcribe_regions(&[region], &wav, &work, opt, &mut whisper, &log)
                    };
                    if placeholder {
                        // a title from what was said, where the picker only had a placeholder
                        let mut one = [Moment { title: String::new(), ..m.clone() }];
                        picker::fallback_titles(&mut one, &said, &picker::game_name(&opt.niche), &picker::niche_hashtags(&opt.niche));
                        m.title = one[0].title.clone();
                    }
                    local_segs = shift_segments(&said, a, b, off);
                }
                if picker::is_placeholder(&m.title) {
                    m.title = picker::placeholder_title(&picker::game_name(&opt.niche), n);
                }
                log(&format!("{n:02}. {}-{} ({:4.1}s) {}", hms(m.start), hms(m.end), m.duration(), m.title));
                let r2 = |x: f64| (x * 100.0).round() / 100.0;
                let local = Moment { start: r2(m.start - off), end: r2(m.end - off), peak: r2(m.peak - off), ..m.clone() };
                let paths = render::render_from(n, &[local], &info, &local_segs, &out_dir, &enc, &opt.layout, &ro, &log)?;
                let caption = format!("{} {}", m.title, m.hashtags.join(" ")).trim().to_string();
                Ok(paths
                    .first()
                    .and_then(|p| p.file_name())
                    .map(|f| ClipMeta { file: Some(f.to_string_lossy().into_owned()), moment: m, caption }))
            })();
            let _ = std::fs::remove_dir_all(piece_dir(i));
            match result {
                Ok(Some(c)) => {
                    progress.done.push((i, c.clone()));
                    clips.push(c);
                }
                Ok(None) => {
                    skipped += 1;
                    progress.skipped.push(i);
                }
                Err(e) => log(&format!("{}-{} skipped: {e:#}", hms(cand.start), hms(cand.end))),
            }
            let _ = serde_json::to_string(&progress).map(|t| std::fs::write(&progress_file, t));
            if clips.len() >= want {
                break;
            }
        }
        // a download still running when we stop: let it finish, then clean up below
        if let Some(h) = next.take() {
            let _ = h.join();
        }
    });
    for i in 0..candidates.len() {
        let _ = std::fs::remove_dir_all(piece_dir(i));
    }
    crate::cancel::check()?;
    if skipped > 0 {
        log(&format!("{skipped} candidates were menus or loading screens and were replaced"));
    }
    if clips.is_empty() {
        bail!("no clips could be rendered");
    }
    std::fs::write(out_dir.join("clips.json"), serde_json::to_string_pretty(&clips)?)?;
    log(&format!("{} clips in {:.0}s -> {}", clips.len(), t0.elapsed().as_secs_f64(), out_dir.display()));
    Ok(Outcome { out_dir, clips })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hp(start_time: f64, end_time: f64, value: f64) -> HeatmapPoint {
        HeatmapPoint { start_time, end_time, value }
    }

    #[test]
    fn legacy_options_still_parse() {
        let opt: Options = serde_json::from_str(r#"{"max_clips": 5, "workers": 2}"#).unwrap();
        assert_eq!(opt.max_clips, 5);
        assert!(!opt.lite);
        assert_eq!(opt.chunk_overlap, None);
        assert_eq!(opt.heatmap_prioritize, None);
    }

    #[test]
    fn chunk_options_accept_null() {
        let opt: Options =
            serde_json::from_str(r#"{"chunk_len": 600, "chunk_overlap": null, "heatmap_prioritize": null, "lite": true}"#).unwrap();
        assert_eq!(opt.chunk_len, Some(600.0));
        assert!(opt.lite);
        assert_eq!(opt.render_opts().max_fps, 30);
        assert_eq!(opt.render_opts().workers, 1);
    }

    #[test]
    fn chunks_cover_the_video_in_order_without_heatmap() {
        let plans = plan_chunks(1200.0, 300.0, 30.0, &[], true);
        assert_eq!(plans.first().unwrap().start, 0.0);
        let last = plans.last().unwrap();
        assert_eq!(last.start + last.len, 1200.0);
        for (i, w) in plans.windows(2).enumerate() {
            assert_eq!(w[0].idx, i + 1);
            assert_eq!(w[1].start - w[0].start, 270.0);
            assert!(w[0].start + w[0].len > w[1].start, "neighbours must overlap");
        }
    }

    #[test]
    fn short_tail_joins_previous_chunk() {
        // 0-300, 270-570, 540-840, 810-1110; 1110-1130 is only 20s so the last window stretches
        let plans = plan_chunks(1130.0, 300.0, 30.0, &[], false);
        let last = plans.last().unwrap();
        assert_eq!(last.start + last.len, 1130.0);
        assert!(last.len >= 300.0);
    }

    #[test]
    fn replay_peak_goes_first_and_bait_last() {
        let heatmap = vec![
            hp(0.0, 15.0, 1.0),         // intro spike: bait, ignored
            hp(300.0, 320.0, 0.2),      // mild
            hp(1100.0, 1150.0, 0.85),   // the real highlight
            hp(3550.0, 3580.0, 0.95),   // outro spike: bait, ignored
        ];
        let plans = plan_chunks(3600.0, 300.0, 30.0, &heatmap, true);
        let top = &plans[0];
        assert!(top.start <= 1100.0 && top.start + top.len >= 1150.0, "top chunk starts at {}", top.start);
        assert!(top.heatmap_score > 0.5);
        let first_bait = plans.iter().position(|p| p.is_bait).expect("outro window is bait");
        assert!(plans[first_bait..].iter().all(|p| p.is_bait), "bait windows sort last");
    }

    #[test]
    fn heatmap_ignored_when_not_prioritizing() {
        let plans = plan_chunks(3600.0, 300.0, 30.0, &[hp(1100.0, 1150.0, 0.9)], false);
        assert!(plans.windows(2).all(|w| w[0].idx < w[1].idx));
    }
}

#[cfg(test)]
mod scan_tests {
    use super::*;

    fn m(start: f64, end: f64) -> Moment {
        Moment { start, end, ..Default::default() }
    }

    #[test]
    fn regions_pad_clamp_and_merge() {
        let r = merge_regions(&[m(100.0, 130.0), m(10.0, 30.0), m(140.0, 170.0)], 20.0, 180.0);
        assert_eq!(r, vec![(0.0, 50.0), (80.0, 180.0)]);
    }

    #[test]
    fn bias_penalises_edges_and_follows_heatmap() {
        let heat = [HeatmapPoint { start_time: 1000.0, end_time: 1100.0, value: 1.0 }];
        let b = scan_bias(3600.0, &heat, true);
        assert!(b[10] < -0.9 && b[3590] < -0.9, "intro and outro are penalised");
        assert!(b[1050] > 1.0, "replayed stretch is boosted");
        assert!(scan_bias(3600.0, &heat, false).iter().all(|x| *x == 0.0));
    }

    #[test]
    fn segments_move_to_piece_time() {
        let seg = Segment {
            s: 105.0, e: 108.0, text: "no way".into(),
            words: vec![crate::transcribe::Word { w: "no".into(), s: 105.0, e: 105.5 }],
        };
        let out = shift_segments(&[seg], 100.0, 140.0, 99.0);
        assert_eq!((out[0].s, out[0].words[0].s), (6.0, 6.0));
        assert!(shift_segments(&out, 500.0, 600.0, 0.0).is_empty());
    }

    #[test]
    fn piece_offset_prefers_exact() {
        assert_eq!(piece_offset(100.0, 30.0, 33.0, Some(98.5)), 98.5);
        assert_eq!(piece_offset(100.0, 30.0, 33.0, None), 100.0);
    }
}

#[cfg(test)]
mod resume_tests {
    use super::*;

    #[test]
    fn options_key_ignores_fresh_only() {
        let a = Options::default();
        let b = Options { fresh: true, ..Options::default() };
        let c = Options { max_clips: 3, ..Options::default() };
        assert_eq!(options_key(&a), options_key(&b));
        assert_ne!(options_key(&a), options_key(&c));
    }
}

#[cfg(test)]
mod chat_tests {
    use super::*;

    #[test]
    fn chat_spike_boosts_the_seconds_before_it() {
        let mut rate = vec![1.0; 300];
        for r in &mut rate[150..160] {
            *r = 40.0;
        }
        let b = chat_bias(&rate);
        let top = (0..300).max_by(|&x, &y| b[x].total_cmp(&b[y])).unwrap();
        assert!((140..155).contains(&top), "peak bias at {top}");
        assert!(b[50] <= 0.0 && b[top] > 1.0);
    }
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn base_upgrades_for_non_english_only() {
        assert_eq!(whisper_model_for("base", Some("hi")), "small");
        assert_eq!(whisper_model_for("base", Some("en")), "base");
        assert_eq!(whisper_model_for("base", None), "base");
        assert_eq!(whisper_model_for("turbo", Some("hi")), "turbo");
    }
}
