//! Word-level transcription types and helpers.
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Word {
    pub w: String,
    pub s: f64,
    pub e: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub s: f64,
    pub e: f64,
    pub text: String,
    pub words: Vec<Word>,
}

pub fn words_in(segments: &[Segment], start: f64, end: f64) -> Vec<Word> {
    segments
        .iter()
        .flat_map(|s| s.words.iter())
        .filter(|w| w.e > start && w.s < end)
        .cloned()
        .collect()
}

pub fn segments_in(segments: &[Segment], start: f64, end: f64) -> Vec<Segment> {
    segments
        .iter()
        .filter(|s| s.e > start && s.s < end)
        .cloned()
        .collect()
}

pub fn text_in(segments: &[Segment], start: f64, end: f64) -> String {
    words_in(segments, start, end).iter().map(|w| w.w.as_str()).collect::<Vec<_>>().join(" ")
}

fn model_file(name: &str) -> &'static str {
    match name {
        "large" => "ggml-large-v3.bin",
        "small" => "ggml-small.bin",
        "base" => "ggml-base.bin",
        _ => "ggml-large-v3-turbo.bin",
    }
}

pub fn model_path(name: &str) -> PathBuf {
    crate::paths::data_dir().join("models").join(model_file(name))
}

/// Download the ggml model once into ~/.clipforge/models.
pub fn ensure_model(name: &str, log: &dyn Fn(&str)) -> Result<PathBuf> {
    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}", model_file(name));
    download_once(&url, &model_path(name), log)
}

fn download_once(url: &str, path: &Path, log: &dyn Fn(&str)) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }
    std::fs::create_dir_all(path.parent().unwrap())?;
    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
    log(&format!("downloading {name}"));
    let mut resp = reqwest::blocking::Client::builder().timeout(None).build()?.get(url).send()?.error_for_status()?;
    let total = resp.content_length().unwrap_or(0);
    let part = path.with_extension("part");
    let mut file = std::fs::File::create(&part)?;
    let (mut done, mut next_pct, mut buf) = (0u64, 10u64, vec![0u8; 1 << 20]);
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        if total > 1 << 26 && done * 100 / total >= next_pct {
            log(&format!("{name} download {next_pct}%"));
            next_pct += 10;
        }
    }
    drop(file);
    std::fs::rename(&part, path)?;
    Ok(path.to_path_buf())
}

/// Join token bytes into words. Tokens are byte pieces: a Hindi (or any non-Latin) letter is
/// often split across two tokens, so decoding each token alone turns it into U+FFFD. A token
/// starting with a space starts a new word; each word is decoded once it is complete.
fn words_from_tokens(tokens: &[(Vec<u8>, f64, f64)]) -> Vec<Word> {
    let mut words = vec![];
    let mut cur: Option<(Vec<u8>, f64, f64)> = None;
    let mut finish = |w: Option<(Vec<u8>, f64, f64)>, words: &mut Vec<Word>| {
        if let Some((bytes, s, e)) = w {
            let text = String::from_utf8_lossy(&bytes).trim().to_string();
            if !text.is_empty() {
                words.push(Word { w: text, s, e });
            }
        }
    };
    for (bytes, s, e) in tokens {
        if bytes.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        match cur.as_mut() {
            Some(w) if bytes.first() != Some(&b' ') => {
                w.0.extend_from_slice(bytes);
                w.2 = e.max(w.2);
            }
            _ => {
                finish(cur.take(), &mut words);
                cur = Some((bytes.clone(), *s, *e));
            }
        }
    }
    finish(cur, &mut words);
    words
}

unsafe extern "C" fn abort_if_cancelled(_user_data: *mut std::ffi::c_void) -> bool {
    crate::cancel::requested()
}

/// whisper.cpp runs best on physical cores; hyperthreads add contention, not speed.
fn whisper_threads() -> i32 {
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
    sysinfo::System::physical_core_count().unwrap_or(logical).clamp(1, 8) as i32
}

/// Audio per whisper call. Bounds memory (10 min of 16 kHz f32 is ~38 MB) whatever the length.
const WINDOW_S: f64 = 600.0;

/// Samples `[start_s, end_s)` of a 16 kHz mono 16-bit WAV, as f32.
fn read_wav_range(wav: &Path, start_s: f64, end_s: f64) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(wav)?;
    let rate = r.spec().sample_rate as f64;
    let total = r.duration();
    let a = ((start_s.max(0.0) * rate) as u32).min(total);
    let b = ((end_s * rate) as u32).clamp(a, total);
    r.seek(a)?;
    Ok(r.samples::<i16>().take((b - a) as usize).filter_map(|s| s.ok()).map(|s| s as f32 / 32768.0).collect())
}

pub fn wav_seconds(wav: &Path) -> Result<f64> {
    let r = hound::WavReader::open(wav)?;
    Ok(r.duration() as f64 / r.spec().sample_rate as f64)
}

/// A loaded whisper model, reused across windows and ranges.
pub struct Whisper {
    ctx: WhisperContext,
    /// Model name ("base", "small", ...), for cache keys.
    pub model: String,
    /// Set by the caller or by `detect_language`. When None every window auto-detects, which
    /// roughly doubles the time on CPU.
    language: std::sync::Mutex<Option<String>>,
}

impl Whisper {
    pub fn load(model: &str, language: Option<&str>, log: &dyn Fn(&str)) -> Result<Self> {
        let model_path = ensure_model(model, log)?;
        log(&format!("transcribing with whisper.cpp ({}), model {model}", crate::platform::whisper_backend()));
        let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
            .map_err(|e| anyhow!("whisper init failed: {e}"))?;
        let language = language.filter(|l| !l.is_empty() && *l != "auto").map(str::to_string);
        Ok(Self { ctx, model: model.to_string(), language: std::sync::Mutex::new(language) })
    }

    /// Language given, or detected once from 30 s samples centred on `at` (one encoder pass
    /// each), stopping at the first confident one. Game audio and music fool the detector, so
    /// spread samples over the video where people talk. Without auto-detection on every window
    /// transcription is about twice as fast and music no longer comes out as foreign-script junk.
    pub fn detect_language(&self, wav: &Path, at: &[f64]) -> Option<String> {
        if let Some(l) = self.language.lock().unwrap().clone() {
            return Some(l);
        }
        let threads = whisper_threads() as usize;
        let mut state = self.ctx.create_state().ok()?;
        let mut votes: HashMap<i32, f32> = HashMap::new();
        for t in at {
            let audio = read_wav_range(wav, t - 15.0, t + 15.0).ok()?;
            if audio.len() < 16_000 * 5 || state.pcm_to_mel(&audio, threads).is_err() {
                continue;
            }
            let Ok((id, probs)) = state.lang_detect(0, threads) else { continue };
            let p = probs.get(id as usize).copied().unwrap_or(0.0);
            if p >= 0.8 {
                votes.clear();
                votes.insert(id, p);
                break;
            }
            *votes.entry(id).or_default() += p;
        }
        let (id, _) = votes.into_iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
        let lang = whisper_rs::get_lang_str(id)?.to_string();
        *self.language.lock().unwrap() = Some(lang.clone());
        Some(lang)
    }

    /// Transcribe `[start_s, end_s)` of the WAV in windows; timestamps are in WAV time.
    pub fn range(&self, wav: &Path, start_s: f64, end_s: f64) -> Result<Vec<Segment>> {
        let mut out = vec![];
        let mut t = start_s.max(0.0);
        while t < end_s {
            crate::cancel::check()?;
            let to = (t + WINDOW_S).min(end_s);
            let audio = read_wav_range(wav, t, to)?;
            if audio.len() < 16_000 / 2 {
                break;
            }
            out.extend(self.run(&audio, t)?);
            t = to;
        }
        Ok(out)
    }

    fn run(&self, audio: &[f32], offset: f64) -> Result<Vec<Segment>> {
        let eot = self.ctx.token_eot();
        let mut state = self.ctx.create_state().map_err(|e| anyhow!("whisper state failed: {e}"))?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let language = self.language.lock().unwrap().clone();
        params.set_language(language.as_deref().or(Some("auto")));
        params.set_token_timestamps(true);
        params.set_no_context(true);
        params.set_initial_prompt("Call of Duty Mobile, CODM, squad, rush, revive, enemy, gameplay, stream, Hindi, Hinglish, English, kill, loot, sniper.");
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_n_threads(whisper_threads());
        // Stop mid-pass when the job is cancelled. Not `set_abort_callback_safe`: in whisper-rs
        // 0.16 its trampoline reads the boxed closure as the wrong type and jumps to a bad
        // address (segfault inside whisper_encode_internal, seen on a real run).
        // SAFETY: a plain function that only reads an atomic flag and ignores the user-data
        // pointer, so it stays valid for the whole call.
        unsafe { params.set_abort_callback(Some(abort_if_cancelled)) };
        state.full(params, audio).map_err(|e| anyhow!("whisper failed: {e}"))?;
        crate::cancel::check()?;

        let mut segments = vec![];
        for i in 0..state.full_n_segments() {
            let Some(seg) = state.get_segment(i) else { continue };
            let (seg_s, seg_e) = (seg.start_timestamp() as f64 / 100.0, seg.end_timestamp() as f64 / 100.0);
            let mut pieces: Vec<(Vec<u8>, f64, f64)> = vec![];
            for t in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(t) else { continue };
                let d = tok.token_data();
                if d.id >= eot {
                    continue;
                }
                let ts = if d.t0 >= 0 { (d.t0 as f64 / 100.0).clamp(seg_s, seg_e) } else { seg_s };
                let te = if d.t1 >= 0 { (d.t1 as f64 / 100.0).clamp(ts, seg_e.max(ts)) } else { ts };
                pieces.push((tok.to_bytes().map(<[u8]>::to_vec).unwrap_or_default(), ts, te));
            }
            let mut words = words_from_tokens(&pieces);
            for w in &mut words {
                w.s = ((w.s + offset) * 1000.0).round() / 1000.0;
                w.e = ((w.e + offset) * 1000.0).round() / 1000.0;
            }
            let text = seg.to_str_lossy().map(|c| c.trim().to_string()).unwrap_or_default();
            // drop hallucinations on silence and music ("Thank you.", stray CJK, broken bytes)
            let speech = seg.no_speech_probability() < 0.6 && !text.contains('\u{FFFD}');
            if speech && !words.is_empty() && text.chars().count() > 1 {
                segments.push(Segment { s: seg_s + offset, e: seg_e + offset, text, words });
            }
        }
        Ok(segments)
    }
}

pub fn transcribe(
    wav: &Path, cache: &Path, model: &str, language: Option<&str>, log: &dyn Fn(&str),
) -> Result<Vec<Segment>> {
    if let Some(segs) = std::fs::read_to_string(cache).ok().and_then(|t| serde_json::from_str(&t).ok()) {
        return Ok(segs);
    }
    let segments = Whisper::load(model, language, log)?.range(wav, 0.0, wav_seconds(wav)?)?;
    std::fs::write(cache, serde_json::to_string(&segments)?)?;
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_hindi_letters_join_into_whole_words() {
        let hindi = "क्या है".as_bytes(); // multi-byte letters
        // split mid-letter, the way whisper's byte-level tokens do
        let toks = vec![
            ([b" ".as_slice(), &hindi[..2]].concat(), 0.0, 0.2),
            (hindi[2..7].to_vec(), 0.2, 0.4),
            (hindi[7..12].to_vec(), 0.4, 0.5),
            (hindi[12..].to_vec(), 0.5, 0.9),
            (b" go".to_vec(), 1.0, 1.2),
        ];
        let w = words_from_tokens(&toks);
        let text: Vec<&str> = w.iter().map(|x| x.w.as_str()).collect();
        assert_eq!(text, vec!["क्या", "है", "go"]);
        assert!(!w.iter().any(|x| x.w.contains('\u{FFFD}')));
        assert_eq!((w[0].s, w[2].e), (0.0, 1.2));
    }
}
