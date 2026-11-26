use crate::media::{require, run};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DlProgress {
    pub percent: f64,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub speed: Option<f64>,
    pub eta: Option<u64>,
    pub kind: String, // video | audio
}

/// One point of YouTube's "most replayed" curve (value is 0..1).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeatmapPoint {
    pub start_time: f64,
    pub end_time: f64,
    pub value: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct SourceMeta {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub title: String,
    pub duration: Option<f64>,
    #[serde(default)]
    pub is_live: Option<bool>,
    #[serde(default, deserialize_with = "null_as_empty")]
    pub heatmap: Vec<HeatmapPoint>,
    /// YouTube kept the live chat replay.
    #[serde(skip)]
    pub has_chat: bool,
}

fn null_as_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<HeatmapPoint>, D::Error> {
    Ok(Option::<Vec<HeatmapPoint>>::deserialize(d)?.unwrap_or_default())
}

const MARK: &str = "CFP|";
const TEMPLATE: &str = "CFP|%(progress.downloaded_bytes)s|%(progress.total_bytes)s|%(progress.total_bytes_estimate)s|%(progress.speed)s|%(progress.eta)s|%(info.vcodec)s";

fn parse_progress(line: &str) -> Option<DlProgress> {
    let f: Vec<&str> = line.strip_prefix(MARK)?.split('|').collect();
    let num = |i: usize| f.get(i).and_then(|v| v.trim().parse::<f64>().ok());
    let downloaded = num(0)? as u64;
    let total = num(1).or_else(|| num(2)).filter(|t| *t > 0.0).map(|t| t as u64);
    let percent = total.map_or(0.0, |t| (downloaded as f64 / t as f64 * 100.0).min(100.0));
    let kind = if f.get(5).is_some_and(|v| v.trim() == "none") { "audio" } else { "video" };
    Some(DlProgress { percent, downloaded, total, speed: num(3), eta: num(4).map(|e| e as u64), kind: kind.into() })
}

const FORMAT: &str = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b";

/// `yt-dlp --help`, read once, to only pass flags this yt-dlp version knows.
fn ytdlp_help() -> &'static str {
    static H: OnceLock<String> = OnceLock::new();
    H.get_or_init(|| {
        require("yt-dlp")
            .and_then(|b| Ok(Command::new(b).arg("--help").output()?))
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default()
    })
}

/// yt-dlp with cookies, retries and download speed-ups shared by every call.
fn yt_dlp() -> Result<Command> {
    crate::cancel::check()?;
    let mut cmd = Command::new(require("yt-dlp")?);
    let cookies = std::env::var_os("CLIPFORGE_COOKIES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cookies.txt"));
    if cookies.is_file() {
        cmd.arg("--cookies").arg(&cookies);
    }
    cmd.args(["--socket-timeout", "30", "--retries", "5"]);
    // Parallel DASH/HLS fragments, and ranged requests that dodge YouTube's per-connection throttle.
    cmd.args(["--concurrent-fragments", "4", "--http-chunk-size", "10M"]);
    // YouTube's JS challenge needs a JS runtime. yt-dlp enables deno by itself; node and bun must
    // be named. Without a solver YouTube serves throttled or no formats.
    let help = ytdlp_help();
    let extra_runtime = || ["node", "bun"].iter().find_map(|n| require(n).ok().map(|p| (n, p)));
    if help.contains("--js-runtimes") && require("deno").is_err()
        && let Some((name, path)) = extra_runtime()
    {
        cmd.arg("--js-runtimes").arg(format!("{name}:{}", path.display()));
    }
    if help.contains("--remote-components") {
        cmd.args(["--remote-components", "ejs:github"]);
    }
    Ok(cmd)
}

fn expand_local(source: &str) -> Result<PathBuf> {
    let expanded = match source.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(source),
    };
    expanded.canonicalize().with_context(|| format!("file not found: {source}"))
}

pub fn fetch(source: &str, workdir: &Path, on_progress: &dyn Fn(DlProgress)) -> Result<PathBuf> {
    if !is_url(source) {
        return expand_local(source);
    }
    let dl = workdir.join("downloads");
    std::fs::create_dir_all(&dl)?;
    let mut cmd = yt_dlp()?;
    cmd.args(["-f", FORMAT, "--merge-output-format", "mp4", "-o"]).arg(dl.join("%(id)s.%(ext)s"));
    download(cmd, source, on_progress)
}

/// Download only `[start, start + len)` of a URL. The cut lands on the keyframe at or before
/// `start`, so the file can begin a few seconds early.
pub fn fetch_chunk(source: &str, start: f64, len: f64, dir: &Path, on_progress: &dyn Fn(DlProgress)) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let mut cmd = yt_dlp()?;
    cmd.args(["-f", FORMAT, "--merge-output-format", "mp4"])
        .args(["--download-sections", &format!("*{start:.2}-{:.2}", start + len)])
        .arg("-o")
        .arg(dir.join("chunk.%(ext)s"));
    let path = download(cmd, source, on_progress)?;
    if !path.is_file() {
        bail!("yt-dlp reported {} but no file is there", path.display());
    }
    Ok(path)
}

/// Audio track only (for scanning a long video before fetching any picture).
pub fn fetch_audio(url: &str, dir: &Path, on_progress: &dyn Fn(DlProgress)) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let mut cmd = yt_dlp()?;
    cmd.args(["-f", "ba[ext=m4a]/ba", "-o"]).arg(dir.join("audio_src.%(ext)s"));
    download(cmd, url, on_progress)
}

/// Title, length, live flag and replay heatmap, without downloading.
pub fn probe_meta(url: &str) -> Result<SourceMeta> {
    let out = yt_dlp()?
        .args(["--no-playlist", "--skip-download", "--print", "%(.{id,title,duration,is_live,heatmap})j"])
        .args(["--print", "chat=%(subtitles.live_chat.0.ext)s", url])
        .stderr(Stdio::piped())
        .output()?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        bail!("yt-dlp could not read the video: {}", err.chars().take(1500).collect::<String>());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{')).context("yt-dlp printed no metadata")?;
    let mut meta: SourceMeta = serde_json::from_str(line)?;
    meta.has_chat = stdout.lines().any(|l| l.strip_prefix("chat=").is_some_and(|e| !e.trim().is_empty() && e.trim() != "NA"));
    Ok(meta)
}

/// The live chat replay as yt-dlp's JSON lines (`<dir>/chat.live_chat.json`). YouTube serves it
/// in small sequential batches, so long streams take a while; callers run this in the background.
pub fn fetch_chat(url: &str, dir: &Path) -> Result<PathBuf> {
    let out = yt_dlp()?
        .args(["--no-playlist", "--skip-download", "--write-subs", "--sub-langs", "live_chat", "-o"])
        .arg(dir.join("chat"))
        .arg(url)
        .stderr(Stdio::piped())
        .output()?;
    let path = dir.join("chat.live_chat.json");
    if !out.status.success() || !path.is_file() {
        let err = String::from_utf8_lossy(&out.stderr);
        bail!("chat replay download failed: {}", err.chars().take(400).collect::<String>());
    }
    Ok(path)
}

/// Chat messages per second of video from a yt-dlp `live_chat.json`.
pub fn chat_rate(path: &Path, secs: usize) -> Result<Vec<f64>> {
    let mut rate = vec![0.0; secs.max(1)];
    let text = std::fs::read_to_string(path)?;
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let replay = &v["replayChatItemAction"];
        let Some(ms) = replay["videoOffsetTimeMsec"].as_str().and_then(|t| t.parse::<f64>().ok()) else { continue };
        let messages = replay["actions"]
            .as_array()
            .map_or(0, |a| a.iter().filter(|x| x.get("addChatItemAction").is_some()).count());
        let i = (ms / 1000.0).max(0.0) as usize;
        if let Some(r) = rate.get_mut(i) {
            *r += messages as f64;
        }
    }
    Ok(rate)
}

/// Direct URLs of the separate video and audio streams yt-dlp would download, when both are
/// plain HTTPS (what `fastdl` can range-fetch). None for single-file or HLS-only formats.
/// Also returns the video id (yt-dlp's download file name).
pub fn resolve_streams(url: &str) -> Result<Option<(String, crate::fastdl::Stream, crate::fastdl::Stream)>> {
    let out = yt_dlp()?
        .args(["--no-playlist", "-f", FORMAT, "--print", "id=%(id)s"])
        .args(["--print", "%(requested_formats.:.{url,ext,protocol,http_headers})j", url])
        .stderr(Stdio::piped())
        .output()?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        bail!("yt-dlp could not resolve streams: {}", err.chars().take(800).collect::<String>());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let id = stdout.lines().find_map(|l| l.strip_prefix("id=")).unwrap_or("").trim().to_string();
    let Some(line) = stdout.lines().rev().find(|l| l.trim_start().starts_with('[')) else { return Ok(None) };
    let streams: Vec<crate::fastdl::Stream> = serde_json::from_str(line)?;
    Ok(match <[_; 2]>::try_from(streams) {
        Ok([v, a]) if v.protocol == "https" && a.protocol == "https" => Some((id, v, a)),
        _ => None,
    })
}

/// Run a prepared yt-dlp download, report progress, return the final file path.
fn download(mut cmd: Command, source: &str, on_progress: &dyn Fn(DlProgress)) -> Result<PathBuf> {
    let mut child = cmd
        .args(["--newline", "--progress", "--progress-template", TEMPLATE])
        .args(["--print", "after_move:filepath", "--no-playlist", source])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stderr = child.stderr.take().context("no stderr")?;
    let err_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr.read_to_string(&mut buf);
        buf
    });
    let mut path: Option<String> = None;
    let mut last_emit = Instant::now() - Duration::from_secs(1);
    for line in BufReader::new(child.stdout.take().context("no stdout")?).lines().map_while(|l| l.ok()) {
        if let Some(p) = parse_progress(&line) {
            if last_emit.elapsed() >= Duration::from_millis(250) || p.percent >= 100.0 {
                on_progress(p);
                last_emit = Instant::now();
            }
        } else if !line.trim().is_empty() {
            path = Some(line.trim().to_string());
        }
    }
    let status = child.wait()?;
    let err = err_thread.join().unwrap_or_default();
    if !status.success() {
        bail!("yt-dlp failed: {}", err.chars().take(1500).collect::<String>());
    }
    Ok(PathBuf::from(path.context("yt-dlp printed no path")?))
}

fn media_seconds(path: &Path) -> Option<f64> {
    let out = Command::new(require("ffprobe").ok()?)
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Like `extract_audio`, but long files are decoded as parallel time ranges (AAC decoding is
/// single-threaded; 8 hours took ~60 s on one core) and joined into one WAV.
pub fn extract_audio_parallel(src: &Path, out_wav: &Path) -> Result<PathBuf> {
    let total = media_seconds(src).unwrap_or(0.0);
    let parts = std::thread::available_parallelism().map_or(2, |n| n.get()).clamp(2, 4);
    if total < 20.0 * 60.0 {
        return extract_audio(src, out_wav);
    }
    let dir = out_wav.with_extension("parts");
    std::fs::create_dir_all(&dir)?;
    let span = total / parts as f64;
    let results: Vec<Result<PathBuf>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..parts)
            .map(|i| {
                let raw = dir.join(format!("{i}.raw"));
                sc.spawn(move || {
                    let start = span * i as f64;
                    // the last part runs to the end so rounding never drops the tail
                    let mut args: Vec<String> = vec!["-y".into(), "-v".into(), "error".into(), "-ss".into(), format!("{start:.6}")];
                    if i + 1 < parts {
                        args.extend(["-t".into(), format!("{span:.6}")]);
                    }
                    args.extend(["-i".into(), src.to_string_lossy().into_owned()]);
                    args.extend(["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-c:a", "pcm_s16le"].map(String::from));
                    args.push(raw.to_string_lossy().into_owned());
                    run(&args, None).map(|_| raw)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("decoder thread panicked")))).collect()
    });
    let raws = results.into_iter().collect::<Result<Vec<_>>>();
    let written = raws.and_then(|raws| {
        let data: u64 = raws.iter().map(|r| std::fs::metadata(r).map_or(0, |m| m.len())).sum();
        let mut w = std::io::BufWriter::new(std::fs::File::create(out_wav)?);
        write_wav_header(&mut w, data)?;
        for r in &raws {
            std::io::copy(&mut std::fs::File::open(r)?, &mut w)?;
            // free each part as it lands, so disk never holds the parts and the full WAV at once
            let _ = std::fs::remove_file(r);
        }
        std::io::Write::flush(&mut w)?;
        Ok(())
    });
    let _ = std::fs::remove_dir_all(&dir);
    written?;
    Ok(out_wav.to_path_buf())
}

/// 44-byte header for 16 kHz mono 16-bit PCM.
fn write_wav_header(w: &mut impl std::io::Write, data_len: u64) -> Result<()> {
    let data = u32::try_from(data_len).context("audio too long for a WAV file")?;
    let (rate, ch, bits) = (16_000u32, 1u16, 16u16);
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?;
    w.write_all(&ch.to_le_bytes())?;
    w.write_all(&rate.to_le_bytes())?;
    w.write_all(&(rate * ch as u32 * bits as u32 / 8).to_le_bytes())?;
    w.write_all(&(ch * bits / 8).to_le_bytes())?;
    w.write_all(&bits.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data.to_le_bytes())?;
    Ok(())
}

/// 16 kHz mono WAV: what whisper wants and cheap to analyse.
pub fn extract_audio(video: &Path, out_wav: &Path) -> Result<PathBuf> {
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if let (Some(a), Some(v)) = (mtime(out_wav), mtime(video)) {
        if a >= v {
            return Ok(out_wav.to_path_buf());
        }
    }
    let args: Vec<String> = [
        "-y", "-v", "error", "-i", &video.to_string_lossy(), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", &out_wav.to_string_lossy(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    run(&args, None)?;
    Ok(out_wav.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_line() {
        let p = parse_progress("CFP|500|1000|NA|2048.5|3|avc1").unwrap();
        assert_eq!((p.percent, p.total, p.eta, p.kind.as_str()), (50.0, Some(1000), Some(3), "video"));
        let p = parse_progress("CFP|10|NA|40|NA|NA|none").unwrap();
        assert_eq!((p.total, p.kind.as_str()), (Some(40), "audio"));
        assert!(parse_progress("/tmp/out.mp4").is_none());
    }

    #[test]
    fn chat_rate_counts_messages_per_second() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.json");
        let msg = |ms: u64, n: usize| {
            let actions: Vec<String> = (0..n).map(|_| r#"{"addChatItemAction":{"item":{}}}"#.to_string()).collect();
            format!(r#"{{"replayChatItemAction":{{"actions":[{}],"videoOffsetTimeMsec":"{ms}"}}}}"#, actions.join(","))
        };
        let body = [msg(500, 1), msg(1200, 3), msg(1900, 2), "not json".into(), msg(99_000, 5)].join("\n");
        std::fs::write(&p, body).unwrap();
        let r = chat_rate(&p, 3).unwrap();
        assert_eq!(r, vec![1.0, 5.0, 0.0]);
    }

    #[test]
    fn wav_header_is_readable() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.wav");
        let mut f = std::fs::File::create(&p).unwrap();
        write_wav_header(&mut f, 8).unwrap();
        std::io::Write::write_all(&mut f, &[1, 0, 2, 0, 3, 0, 4, 0]).unwrap();
        drop(f);
        let mut r = hound::WavReader::open(&p).unwrap();
        assert_eq!((r.spec().sample_rate, r.spec().channels, r.duration()), (16_000, 1, 4));
        assert_eq!(r.samples::<i16>().map(|x| x.unwrap()).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn meta_with_and_without_heatmap() {
        let m: SourceMeta = serde_json::from_str(
            r#"{"id": "a", "title": "GTA | Stream: 3", "duration": 6880, "is_live": false}"#,
        )
        .unwrap();
        assert_eq!((m.title.as_str(), m.duration), ("GTA | Stream: 3", Some(6880.0)));
        assert!(m.heatmap.is_empty());
        let m: SourceMeta = serde_json::from_str(
            r#"{"id": "b", "title": "t", "duration": null, "is_live": true, "heatmap": null}"#,
        )
        .unwrap();
        assert_eq!((m.duration, m.is_live), (None, Some(true)));
        let m: SourceMeta = serde_json::from_str(
            r#"{"id": "c", "title": "t", "duration": 60, "heatmap": [{"start_time": 0, "end_time": 6, "value": 1.0}]}"#,
        )
        .unwrap();
        assert_eq!(m.heatmap.len(), 1);
    }
}

/// Returns the path to the best json3 file if found.
pub fn fetch_subtitles(url: &str, dir: &Path) -> Result<Option<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let mut cmd = yt_dlp()?;
    let _ = cmd
        .args([
            "--no-playlist",
            "--skip-download",
            "--write-auto-subs",
            "--write-subs",
            "--sub-langs",
            "hi-orig,en-orig,default,.*-orig,hi,en",
            "--sub-format",
            "json3",
            "-o",
        ])
        .arg(dir.join("subs.%(ext)s"))
        .arg(url)
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .status();

    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut files: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("subs.") && n.ends_with(".json3"))
            })
            .collect();
        // Prefer -orig
        files.sort_by_key(|p| if p.to_string_lossy().contains("-orig") { 0 } else { 1 });
        if let Some(first) = files.into_iter().next() {
            return Ok(Some(first));
        }
    }
    Ok(None)
}

/// Parse YouTube's json3 subtitle format into Word-level Segments.
pub fn parse_json3_subtitles(path: &Path) -> Result<Vec<crate::transcribe::Segment>> {
    let text = std::fs::read_to_string(path)?;
    let val: serde_json::Value = serde_json::from_str(&text)?;
    let events = val.get("events").and_then(|e| e.as_array()).context("no events")?;
    let mut segments = Vec::new();
    for ev in events {
        let t_start = ev.get("tStartMs").and_then(|t| t.as_f64()).unwrap_or(0.0) / 1000.0;
        let d_dur = ev.get("dDurationMs").and_then(|t| t.as_f64()).unwrap_or(0.0) / 1000.0;
        let Some(segs) = ev.get("segs").and_then(|s| s.as_array()) else { continue };

        let valid_segs: Vec<&serde_json::Value> = segs
            .iter()
            .filter(|s| {
                let u = s.get("utf8").and_then(|u| u.as_str()).unwrap_or("").trim();
                !u.is_empty() && u != "\n"
            })
            .collect();

        let mut words = Vec::new();
        for (idx, s) in valid_segs.iter().enumerate() {
            let w_text = s.get("utf8").and_then(|u| u.as_str()).unwrap_or("").trim().to_string();
            let offset = s.get("tOffsetMs").and_then(|o| o.as_f64()).unwrap_or(0.0) / 1000.0;
            let ws = t_start + offset;
            let we = if idx + 1 < valid_segs.len() {
                let next_off = valid_segs[idx + 1].get("tOffsetMs").and_then(|o| o.as_f64()).unwrap_or(offset + 0.3) / 1000.0;
                (t_start + next_off).max(ws + 0.1)
            } else if d_dur > 0.0 {
                (t_start + d_dur).max(ws + 0.2)
            } else {
                ws + 0.3
            };
            words.push(crate::transcribe::Word {
                w: w_text,
                s: (ws * 1000.0).round() / 1000.0,
                e: (we * 1000.0).round() / 1000.0,
            });
        }
        if !words.is_empty() {
            let seg_text = words.iter().map(|w| w.w.as_str()).collect::<Vec<_>>().join(" ");
            let s = words.first().map(|w| w.s).unwrap_or(t_start);
            let e = words.last().map(|w| w.e).unwrap_or(t_start + d_dur);
            segments.push(crate::transcribe::Segment { s, e, text: seg_text, words });
        }
    }
    Ok(segments)
}
