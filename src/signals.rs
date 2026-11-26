//! Non-speech signals: loudness spikes and on-screen action, reduced to per-second arrays.
use crate::media::{Encoder, require};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Signals {
    pub loud: Vec<f64>,
    pub spike: Vec<f64>,
    pub activity: Vec<f64>,
    pub cuts: Vec<f64>,
}

/// RMS dBFS per second (max over 250 ms sub-windows, so short bangs count).
pub fn loudness_per_second(wav: &Path) -> Result<Vec<f64>> {
    let mut reader = hound::WavReader::open(wav)?;
    let hop = ((reader.spec().sample_rate / 4) as usize).max(1);
    // streamed: hours of audio never sit in memory at once
    let (mut db, mut acc, mut k) = (Vec::new(), 0.0f64, 0usize);
    for s in reader.samples::<i16>().filter_map(|s| s.ok()) {
        let x = s as f64 / 32768.0;
        acc += x * x;
        k += 1;
        if k == hop {
            db.push(20.0 * (acc / hop as f64 + 1e-10).sqrt().log10());
            (acc, k) = (0.0, 0);
        }
    }
    if db.is_empty() {
        return Ok(vec![0.0]);
    }
    let secs = db.len() / 4;
    if secs == 0 {
        return Ok(vec![db.iter().cloned().fold(f64::MIN, f64::max)]);
    }
    Ok(db
        .chunks_exact(4)
        .take(secs)
        .map(|c| c.iter().cloned().fold(f64::MIN, f64::max))
        .collect())
}

pub fn rolling_median(x: &[f64], window: usize) -> Vec<f64> {
    let half = window / 2;
    let len = x.len();
    (0..len)
        .map(|i| {
            let mut w: Vec<f64> = (i as isize - half as isize..=(i + half) as isize)
                .map(|j| x[j.clamp(0, len as isize - 1) as usize])
                .collect();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w[w.len() / 2]
        })
        .collect()
}

/// Per-second mean scene-change score (about on-screen action) and hard-cut timestamps.
pub fn visual_activity(video: &Path, duration: f64, enc: &Encoder) -> Result<(Vec<f64>, Vec<f64>)> {
    crate::cancel::check()?;
    // Software decode: skip non-reference frames and the deblocking filter. We sample 4 fps at
    // 256 px, so the scores barely move, but decoding takes about half the CPU.
    let fast_decode: &[&str] =
        if enc.decode_args.is_empty() { &["-skip_frame:v", "noref", "-skip_loop_filter:v", "all"] } else { &[] };
    let out = Command::new(require("ffmpeg")?)
        .args(["-v", "error"])
        .args(&enc.decode_args)
        .args(fast_decode)
        .arg("-i")
        .arg(video)
        .args([
            "-an", "-vf",
            "fps=4,scale=256:-2,select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file=-",
            "-f", "null", "-",
        ])
        .output()?;
    let secs = (duration.ceil() as usize).max(1);
    let (mut sums, mut counts) = (vec![0.0f64; secs], vec![0.0f64; secs]);
    let mut cuts = vec![];
    let mut t = 0.0;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if let Some(v) = value_after(line, "pts_time:") {
            t = v;
        } else if let Some(score) = value_after(line, "scene_score=") {
            let i = (t as usize).min(secs - 1);
            sums[i] += score;
            counts[i] += 1.0;
            if score > 0.35 {
                cuts.push((t * 100.0).round() / 100.0);
            }
        }
    }
    let act = sums.iter().zip(&counts).map(|(s, c)| s / c.max(1.0)).collect();
    Ok((act, cuts))
}

fn value_after(line: &str, key: &str) -> Option<f64> {
    let rest = &line[line.find(key)? + key.len()..];
    let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

pub fn zscore(x: &[f64]) -> Vec<f64> {
    let n = x.len().max(1) as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    if sd > 1e-6 { x.iter().map(|v| (v - mean) / sd).collect() } else { vec![0.0; x.len()] }
}

fn fit(mut v: Vec<f64>, len: usize, fill: f64) -> Vec<f64> {
    v.resize(len, fill);
    v
}

pub fn compute(video: &Path, wav: Option<&Path>, duration: f64, enc: &Encoder, cache: &Path) -> Result<Signals> {
    if let Ok(text) = std::fs::read_to_string(cache) {
        if let Ok(s) = serde_json::from_str(&text) {
            return Ok(s);
        }
    }
    let secs = (duration.ceil() as usize).max(1);
    let mut out = audio_only(wav, duration)?;
    let (act, cuts) = visual_activity(video, duration, enc)?;
    (out.activity, out.cuts) = (fit(act, secs, 0.0), cuts);
    std::fs::write(cache, serde_json::to_string(&out)?)?;
    Ok(out)
}

/// Loudness and spikes only: no video needed. Activity is flat and there are no cuts.
pub fn audio_only(wav: Option<&Path>, duration: f64) -> Result<Signals> {
    let secs = (duration.ceil() as usize).max(1);
    let loud = match wav {
        Some(w) => fit(loudness_per_second(w)?, secs, -60.0),
        None => vec![-60.0; secs],
    };
    let med = rolling_median(&loud, 61);
    let spike = loud.iter().zip(&med).map(|(l, m)| (l - m).max(0.0)).collect();
    Ok(Signals { loud, spike, activity: vec![0.0; secs], cuts: vec![] })
}

/// Share of `[from, to)` (seconds into `video`) that looks like a pause menu or loading screen:
/// heavily blurred and nearly still. Pause menus in games blur the frame behind them, so blur
/// separates them from gameplay far better than motion alone (menus often animate a glow).
/// Measured on GTA VI: menu seconds score 7-13 on `blurdetect`, gameplay 3.7-6.2.
pub fn menu_share(video: &Path, from: f64, to: f64) -> Result<f64> {
    crate::cancel::check()?;
    let out = Command::new(require("ffmpeg")?)
        .args(["-v", "error", "-skip_frame:v", "noref", "-skip_loop_filter:v", "all"])
        .args(["-ss", &format!("{:.3}", from.max(0.0)), "-t", &format!("{:.3}", (to - from).max(1.0)), "-i"])
        .arg(video)
        .args([
            "-an", "-vf",
            "fps=2,scale=480:-2,format=gray,blurdetect=block_width=32:block_height=32,select='gte(scene,0)',metadata=print:file=-",
            "-f", "null", "-",
        ])
        .output()?;
    let (mut frames, mut menu) = (0usize, 0usize);
    let (mut blur, mut scene) = (None, None);
    let mut flush = |blur: &mut Option<f64>, scene: &mut Option<f64>| {
        if let (Some(b), Some(s)) = (blur.take(), scene.take()) {
            frames += 1;
            if b >= 7.0 && s < 0.02 {
                menu += 1;
            }
        }
    };
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if line.starts_with("frame:") {
            flush(&mut blur, &mut scene);
        } else if let Some(v) = value_after(line, "lavfi.blur=") {
            blur = Some(v);
        } else if let Some(v) = value_after(line, "lavfi.scene_score=") {
            scene = Some(v);
        }
    }
    flush(&mut blur, &mut scene);
    Ok(if frames == 0 { 0.0 } else { menu as f64 / frames as f64 })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs the sample pieces in /tmp/cf-bench/vis (downloaded during tuning); skipped otherwise.
    #[test]
    fn menu_share_separates_pause_menu_from_gameplay() {
        let dir = Path::new("/tmp/cf-bench/vis");
        if !dir.join("p14860.mp4").is_file() {
            return;
        }
        for (f, menu) in [("p14860", true), ("p550", true), ("p23851", false), ("p25966", false)] {
            let share = menu_share(&dir.join(format!("{f}.mp4")), 0.0, 30.0).unwrap();
            assert_eq!(share >= 0.3, menu, "{f}: {share:.2}");
        }
    }
}
