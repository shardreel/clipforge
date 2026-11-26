//! Find the best moments: excitement curve, candidate windows, dedupe, fallback titles.
use crate::llm::{Llm, parse_json};
use crate::media::{Encoder, require};
use crate::signals::{Signals, zscore};
use crate::transcribe::{Segment, text_in};
use base64::Engine;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::OnceLock;

fn hype_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(no way|oh my god|omg|let'?s go+|bro+|what the|wtf|holy|insane|crazy|clutch|run+|cops?|police|wanted|five stars?|5 stars?|crash(ed)?|boom|dead|died|look at (this|that)|did you see|yo+|wait+|nah+|bruh|sheesh|hahaha+|lmao)\b",
        )
        .unwrap()
    })
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Moment {
    pub start: f64,
    pub end: f64,
    pub score: f64,
    pub peak: f64,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub hook: String,
    #[serde(default)]
    pub hashtags: Vec<String>,
    #[serde(default)]
    pub reason: String,
}

impl Moment {
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

pub fn speech_rate(segments: &[Segment], secs: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut rate, mut hype) = (vec![0.0; secs], vec![0.0; secs]);
    for s in segments {
        for w in &s.words {
            if w.s >= 0.0 && (w.s as usize) < secs {
                rate[w.s as usize] += 1.0;
            }
        }
        let n = hype_re().find_iter(&s.text).count();
        let a = s.s.max(0.0) as usize;
        let b = (s.e.max(0.0) as usize).min(secs.saturating_sub(1));
        for _ in 0..n {
            for h in hype.iter_mut().take(b + 1).skip(a) {
                *h += 1.0;
            }
        }
    }
    (rate, hype)
}

pub fn excitement(sig: &Signals, segments: &[Segment]) -> Vec<f64> {
    let secs = sig.loud.len();
    let (rate, hype) = speech_rate(segments, secs);
    let has_speech = rate.iter().sum::<f64>() > 20.0;
    let (w_spike, w_act, w_rate, w_hype) =
        if has_speech { (0.45, 0.25, 0.15, 0.15) } else { (0.6, 0.4, 0.0, 0.0) };
    let (zs, za, zr) = (zscore(&sig.spike), zscore(&sig.activity), zscore(&rate));
    let mut curve: Vec<f64> = (0..secs)
        .map(|i| {
            let mut v = w_spike * zs[i] + w_act * za[i] + w_rate * zr[i] + w_hype * hype[i].clamp(0.0, 3.0);
            if sig.loud[i] < -50.0 {
                v -= 1.0;
            }
            v
        })
        .collect();
    let src = curve.clone();
    for i in 0..secs {
        let prev = if i > 0 { src[i - 1] } else { 0.0 };
        let next = if i + 1 < secs { src[i + 1] } else { 0.0 };
        curve[i] = (prev + src[i] + next) / 3.0;
    }
    curve
}

/// Starts only move earlier (more build-up), ends only move later (never clip the payoff).
fn snap(t: f64, anchors: &[f64], tol: f64, direction: f64) -> f64 {
    anchors
        .iter()
        .copied()
        .filter(|a| (a - t) * direction >= 0.0 && (a - t).abs() <= tol)
        .min_by(|a, b| (a - t).abs().partial_cmp(&(b - t).abs()).unwrap())
        .unwrap_or(t)
}

pub fn candidates(
    curve: &[f64], segments: &[Segment], cuts: &[f64], duration: f64, n: usize,
    min_len: f64, max_len: f64, target: f64,
) -> Vec<Moment> {
    let mut c = curve.to_vec();
    let seg_starts: Vec<f64> = segments.iter().map(|s| s.s).collect();
    let seg_ends: Vec<f64> = segments.iter().map(|s| s.e + 0.3).collect();
    let mut out = vec![];
    for _ in 0..n {
        let Some((peak, &pv)) = c.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) else { break };
        if pv < -0.5 || pv == f64::NEG_INFINITY {
            break;
        }
        let length = target;
        let mut start = (peak as f64 - 0.65 * length).max(0.0);
        let end = (start + length).min(duration);
        start = (end - length).max(0.0);
        let mut s2 = snap(start, &seg_starts, 3.0, -1.0);
        if s2 == start {
            s2 = snap(s2, cuts, 1.5, -1.0);
        }
        let mut e2 = snap(end, &seg_ends, 3.0, 1.0);
        if e2 == end {
            e2 = snap(e2, cuts, 1.5, 1.0);
        }
        e2 = e2.min(duration);
        let (start, end) = if (min_len..=max_len).contains(&(e2 - s2)) { (s2, e2) } else { (start, end) };
        let lo_i = (start as usize).min(curve.len());
        let hi_i = (end as usize + 1).min(curve.len());
        let window = &curve[lo_i..hi_i.max(lo_i)];
        let mean = if window.is_empty() { 0.0 } else { window.iter().sum::<f64>() / window.len() as f64 };
        out.push(Moment {
            start: round2(start), end: round2(end), score: ((mean + curve[peak]) * 1000.0).round() / 1000.0,
            peak: peak as f64, ..Default::default()
        });
        let lo = (start as usize).saturating_sub(5);
        let hi = (end as usize + 5).min(c.len());
        for v in &mut c[lo..hi] {
            *v = f64::NEG_INFINITY;
        }
    }
    out
}

/// Two windows share at least 40% of the shorter one.
pub fn overlapping(a: &Moment, b: &Moment) -> bool {
    a.end.min(b.end) - a.start.max(b.start) >= 0.4 * a.duration().min(b.duration())
}

/// The game from a niche like "Game: Call of Duty Mobile (CODM). Audience: ...": the shorter of
/// the name and its bracketed alias ("CODM"; "GTA 6" for the default niche). "Gaming" if absent.
pub fn game_name(niche: &str) -> String {
    let Some((name, alias)) = niche_game(niche) else { return "Gaming".into() };
    match alias {
        Some(a) if a.chars().count() < name.chars().count() => a,
        _ => name,
    }
}

/// Fallback hashtags: the game (name and alias, e.g. #codm #callofdutymobile) plus generic tags.
pub fn niche_hashtags(niche: &str) -> Vec<String> {
    let tag = |s: &str| {
        let t: String = s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
        (!t.is_empty()).then(|| format!("#{t}"))
    };
    let mut tags: Vec<String> = vec![];
    if let Some((name, alias)) = niche_game(niche) {
        tags.extend(alias.as_deref().and_then(tag));
        tags.extend(tag(&name));
    }
    tags.extend(["#gaming".to_string(), "#shorts".to_string()]);
    tags.dedup();
    tags
}

fn niche_game(niche: &str) -> Option<(String, Option<String>)> {
    let rest = &niche[niche.to_lowercase().find("game:")? + 5..];
    let line = rest.split(['.', '\n']).next()?.trim();
    let (name, alias) = match line.split_once('(') {
        Some((n, a)) => (n.trim(), a.split(')').next().map(|a| a.trim().to_string()).filter(|a| !a.is_empty())),
        None => (line, None),
    };
    (!name.is_empty()).then(|| (name.to_string(), alias))
}

pub fn dedupe(ms: Vec<Moment>) -> Vec<Moment> {
    let mut kept: Vec<Moment> = vec![];
    for m in ms {
        let clear = kept.iter().all(|k| !overlapping(&m, k));
        if clear {
            kept.push(m);
        }
    }
    kept
}

/// A placeholder title from `placeholder_title`, whatever its number.
pub fn is_placeholder(title: &str) -> bool {
    title.rsplit_once(" moment #").is_some_and(|(_, n)| n.parse::<usize>().is_ok())
}

/// Title for a moment with no speech and no model-written title.
pub fn placeholder_title(tag: &str, n: usize) -> String {
    format!("{tag} moment #{n}")
}

pub fn fallback_titles(ms: &mut [Moment], segments: &[Segment], tag: &str, hashtags: &[String]) {
    for (i, m) in ms.iter_mut().enumerate() {
        if m.title.is_empty() {
            let text = text_in(segments, m.peak - 3.0, m.peak + 3.0);
            let head = text.split_whitespace().take(8).collect::<Vec<_>>().join(" ");
            let head = head.trim_matches(|c| c == ' ' || c == ',' || c == '.').to_string();
            // a stray word or two ("of", "(music)") makes a worse title than the placeholder
            let words = head.split_whitespace().filter(|w| w.chars().any(char::is_alphanumeric)).count();
            m.title = if words < 3 { placeholder_title(tag, i + 1) } else { head };
        }
        if m.hashtags.is_empty() {
            m.hashtags = hashtags.to_vec();
        }
    }
}

const SYSTEM: &str = "You are a short-form video editor for gaming clips (the game and audience are given below).
You pick moments that will hold attention on TikTok / Reels / Shorts: chaos, chases, crashes,
fails, funny reactions, clutch escapes, surprising physics, strong one-liners.
A good clip has a hook in the first 2 seconds and a payoff. No slow intros, no menus/loading screens.";

pub struct PickOpts<'a> {
    pub max_clips: usize,
    pub min_len: f64,
    pub max_len: f64,
    pub target: f64,
    pub llm: Option<&'a Llm>,
    pub vision: bool,
    pub niche: &'a str,
    pub video: &'a Path,
    pub enc: &'a Encoder,
    /// Added to the excitement curve per second (replay heatmap, intro/outro penalty). May be empty.
    pub bias: &'a [f64],
}

fn biased_curve(sig: &Signals, segments: &[Segment], bias: &[f64]) -> Vec<f64> {
    let mut curve = excitement(sig, segments);
    for (c, b) in curve.iter_mut().zip(bias) {
        *c += b;
    }
    curve
}

/// Candidate windows from signals alone, wider than clips: where to spend transcription.
pub fn scout(sig: &Signals, bias: &[f64], duration: f64, n: usize, min_len: f64, max_len: f64, target: f64) -> Vec<Moment> {
    let curve = biased_curve(sig, &[], bias);
    candidates(&curve, &[], &[], duration, n, min_len, max_len, target.clamp(min_len, max_len))
}

fn frames_b64(video: &Path, m: &Moment, n: usize) -> Vec<String> {
    let times: Vec<f64> = if m.duration() > 3.0 && n > 1 {
        let (a, b) = (m.start + 1.0, m.end - 1.0);
        (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
    } else {
        vec![m.start]
    };
    let Ok(ffmpeg) = require("ffmpeg") else { return vec![] };
    times
        .into_iter()
        .filter_map(|t| {
            let out = std::process::Command::new(&ffmpeg)
                .args(["-v", "error", "-ss", &format!("{t:.2}"), "-i"])
                .arg(video)
                .args(["-frames:v", "1", "-vf", "scale=384:-2", "-q:v", "6", "-f", "image2", "-c:v", "mjpeg", "-"])
                .output()
                .ok()?;
            (out.status.success() && !out.stdout.is_empty())
                .then(|| base64::engine::general_purpose::STANDARD.encode(&out.stdout))
        })
        .collect()
}

fn refine_with_llm(
    llm: &Llm, moments: &[Moment], segments: &[Segment], duration: f64, o: &PickOpts,
) -> anyhow::Result<Vec<Moment>> {
    let mut content: Vec<Value> = vec![json!({"type": "text", "text": format!(
        "{}\nVideo length: {duration:.0}s. Below are {} candidate windows found from audio spikes and on-screen \
         action (signal score = how intense). Choose the best {}. You may shift start/end by up to 10s to tighten \
         the hook or land the payoff; length must stay within {:.0}-{:.0}s.\n",
        o.niche, moments.len(), o.max_clips, o.min_len, o.max_len)})];
    for (i, m) in moments.iter().enumerate() {
        let speech = text_in(segments, m.start, m.end);
        let speech = if speech.is_empty() { "(no speech - gameplay audio only)".to_string() } else { speech };
        content.push(json!({"type": "text", "text": format!(
            "\n### Candidate {i}: {:.1}s -> {:.1}s (peak at {:.0}s, signal {:.2})\nSpeech: {}",
            m.start, m.end, m.peak, m.score, speech.chars().take(900).collect::<String>())}));
        if o.vision {
            for b in frames_b64(o.video, m, 3) {
                content.push(json!({"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b}")}}));
            }
        }
    }
    content.push(json!({"type": "text", "text": "\nReply with ONLY a JSON array, best first:\n\
[{\"candidate\": 0, \"start\": 12.0, \"end\": 41.5, \"score\": 0-100,\n  \"title\": \"short punchy title (max 60 chars)\",\n  \
\"hook\": \"on-screen hook text, max 6 words\",\n  \"hashtags\": [\"#tag\", \"...\"], \"reason\": \"one sentence\"}]"}));
    let reply = llm.chat(json!([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]))?;
    let mut picks = parse_json(&reply)?;
    if picks.is_object() {
        picks = ["clips", "moments"]
            .iter()
            .find_map(|k| picks.get(*k).cloned())
            .or_else(|| picks.as_object().and_then(|m| m.values().next().cloned()))
            .unwrap_or(Value::Null);
    }
    let mut out = vec![];
    for p in picks.as_array().cloned().unwrap_or_default() {
        let idx = p["candidate"].as_i64().unwrap_or(0);
        let Some(base) = usize::try_from(idx).ok().and_then(|i| moments.get(i)) else { continue };
        let (mut s, mut e) = (p["start"].as_f64().unwrap_or(base.start), p["end"].as_f64().unwrap_or(base.end));
        // guard rails: stay near the candidate and within limits
        if !(base.start - 10.0..=base.peak - 1.0).contains(&s) {
            s = base.start;
        }
        if !(base.peak + 2.5..=base.end + 10.0).contains(&e) {
            e = base.end;
        }
        s = s.max(0.0);
        e = e.min(duration);
        if e - s > o.max_len {
            e = s + o.max_len;
        }
        if e - s < o.min_len {
            e = duration.min(s + o.min_len);
            s = (e - o.min_len).max(0.0);
        }
        let hashtags = p["hashtags"].as_array().map(|a| {
            a.iter()
                .filter_map(|h| h.as_str())
                .map(|h| if h.starts_with('#') { h.to_string() } else { format!("#{h}") })
                .take(8)
                .collect()
        });
        out.push(Moment {
            start: round2(s),
            end: round2(e),
            score: p["score"].as_f64().unwrap_or(base.score),
            peak: base.peak,
            title: p["title"].as_str().unwrap_or("").chars().take(80).collect(),
            hook: p["hook"].as_str().unwrap_or("").chars().take(48).collect(),
            hashtags: hashtags.unwrap_or_default(),
            reason: p["reason"].as_str().unwrap_or("").to_string(),
        });
    }
    let mut out = dedupe(out);
    out.truncate(o.max_clips);
    Ok(out)
}

pub fn pick(
    sig: &Signals, segments: &[Segment], duration: f64, o: &PickOpts, cache: Option<&Path>,
    log: &dyn Fn(&str),
) -> anyhow::Result<Vec<Moment>> {
    if let Some(text) = cache.and_then(|c| std::fs::read_to_string(c).ok()) {
        if let Ok(ms) = serde_json::from_str(&text) {
            return Ok(ms);
        }
    }
    let target = o.target.clamp(o.min_len, o.max_len);
    let mut ms = if duration <= o.min_len + 1.0 {
        vec![Moment { start: 0.0, end: round2(duration), score: 1.0, peak: duration / 2.0, ..Default::default() }]
    } else {
        let curve = biased_curve(sig, segments, o.bias);
        candidates(&curve, segments, &sig.cuts, duration, o.max_clips * 2, o.min_len, o.max_len, target)
    };
    match o.llm {
        Some(llm) if !ms.is_empty() => {
            log(&format!("ranking {} candidates with {}{}", ms.len(), llm.model, if o.vision { " + frames" } else { "" }));
            match refine_with_llm(llm, &ms, segments, duration, o) {
                Ok(v) if !v.is_empty() => ms = v,
                Ok(_) => ms.truncate(o.max_clips),
                // never lose a run because the LLM hiccuped
                Err(e) => {
                    log(&format!("LLM step failed ({e:#}); using signal ranking"));
                    ms.truncate(o.max_clips);
                }
            }
        }
        _ => ms.truncate(o.max_clips),
    }
    fallback_titles(&mut ms, segments, &game_name(o.niche), &niche_hashtags(o.niche));
    if let Some(c) = cache {
        std::fs::write(c, serde_json::to_string_pretty(&ms)?)?;
    }
    Ok(ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcribe::Word;

    fn seg(s: f64, text: &str) -> Segment {
        let words = text.split_whitespace().enumerate().map(|(i, w)| Word { w: w.into(), s: s + i as f64 * 0.3, e: s + i as f64 * 0.3 + 0.2 }).collect();
        Segment { s, e: s + 3.0, text: text.into(), words }
    }

    #[test]
    fn fallback_titles_need_three_words() {
        let mut ms = vec![Moment { start: 0.0, end: 30.0, peak: 10.0, ..Default::default() }; 2];
        ms[1].peak = 50.0;
        fallback_titles(&mut ms, &[seg(9.0, "of"), seg(49.0, "the only way through")], "GTA", &[]);
        assert_eq!(ms[0].title, "GTA moment #1");
        assert_eq!(ms[1].title, "the only way through");
    }
}

#[cfg(test)]
mod niche_tests {
    use super::*;

    #[test]
    fn game_and_tags_follow_the_niche() {
        let codm = "Game: Call of Duty Mobile (CODM). Audience: gaming Shorts. Commentary in Hindi.";
        assert_eq!(game_name(codm), "CODM");
        assert_eq!(niche_hashtags(codm), vec!["#codm", "#callofdutymobile", "#gaming", "#shorts"]);
        let gta = crate::settings::DEFAULT_NICHE;
        assert_eq!(game_name(gta), "GTA 6");
        assert_eq!(niche_hashtags(gta)[0], "#grandtheftautovi");
        assert_eq!(game_name("Funny moments"), "Gaming");
        assert_eq!(niche_hashtags("Funny moments"), vec!["#gaming", "#shorts"]);
        assert!(is_placeholder("CODM moment #12") && is_placeholder("GTA 6 moment #1"));
        assert!(!is_placeholder("the only way through"));
    }
}
