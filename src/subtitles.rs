//! Word-by-word karaoke captions + hook banner as an ASS file (rendered by libass in ffmpeg).
use crate::transcribe::Word;
use anyhow::Result;
use std::path::Path;

const HEADER: &str = "[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{cap_font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,{marginv},1
Style: Hook,{hook_font},70,&H00000000,&H00000000,&H0000E5FF,&H0000E5FF,-1,0,0,0,100,100,0,0,3,18,0,8,80,80,230,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
";

const HIGHLIGHT: &str = r"{\c&H0000E5FF&}";
const RESET: &str = r"{\c&H00FFFFFF&}";

fn ts(sec: f64) -> String {
    let sec = sec.max(0.0);
    let h = (sec / 3600.0) as u32;
    let m = ((sec % 3600.0) / 60.0) as u32;
    format!("{h}:{m:02}:{:05.2}", sec % 60.0)
}

fn esc(s: &str) -> String {
    s.replace('\\', "").replace('{', "(").replace('}', ")").replace('\n', " ")
}

fn has_indic(s: &str) -> bool {
    s.chars().any(|c| ('\u{0900}'..='\u{0D7F}').contains(&c))
}

fn norm_case(s: &str) -> String {
    if s.is_ascii() {
        s.to_ascii_uppercase()
    } else {
        s.to_string()
    }
}

fn chunk_words(words: &[Word]) -> Vec<Vec<Word>> {
    let (max_words, max_gap) = (3, 0.6);
    let mut chunks: Vec<Vec<Word>> = vec![];
    let mut cur: Vec<Word> = vec![];
    for w in words {
        let split = cur.last().is_some_and(|l| {
            cur.len() >= max_words
                || w.s - l.e > max_gap
                || l.w.chars().last().is_some_and(|c| ".?!,".contains(c))
        });
        if split {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push(w.clone());
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

pub struct AssOpts<'a> {
    pub hook: &'a str,
    pub margin_v: u32,
}

pub fn build_ass(words: &[Word], clip_start: f64, clip_end: f64, out: &Path, o: &AssOpts) -> Result<()> {
    let dur = clip_end - clip_start;
    let hook_secs: f64 = 3.5;
    let cap_font = if words.iter().any(|w| has_indic(&w.w)) {
        "Noto Sans Devanagari ExtraBold"
    } else {
        "Arial Black"
    };
    let hook_font = if has_indic(o.hook) {
        "Noto Sans Devanagari ExtraBold"
    } else {
        "Arial Black"
    };
    let header = HEADER
        .replace("{cap_font}", cap_font)
        .replace("{hook_font}", hook_font)
        .replace("{size}", "84")
        .replace("{marginv}", &o.margin_v.to_string());
    let mut lines = vec![header];
    if !o.hook.is_empty() {
        lines.push(format!(
            "Dialogue: 1,{},{},Hook,,0,0,0,,{{\\fad(120,200)}}{}",
            ts(0.0),
            ts(hook_secs.min(dur)),
            esc(&norm_case(o.hook))
        ));
    }
    let rel: Vec<Word> = words
        .iter()
        .filter(|w| w.e > clip_start && w.s < clip_end)
        .map(|w| Word { w: esc(&norm_case(&w.w)), s: w.s - clip_start, e: w.e - clip_start })
        .collect();
    for chunk in chunk_words(&rel) {
        for (i, w) in chunk.iter().enumerate() {
            let end = chunk.get(i + 1).map_or((w.s + 0.25).max(w.e), |n| n.s);
            let parts: Vec<String> = chunk
                .iter()
                .enumerate()
                .map(|(j, x)| if j == i { format!("{HIGHLIGHT}{}{RESET}", x.w) } else { x.w.clone() })
                .collect();
            lines.push(format!(
                "Dialogue: 0,{},{},Cap,,0,0,0,,{}",
                ts(w.s),
                ts(end.min(dur)),
                parts.join(" ")
            ));
        }
    }
    std::fs::write(out, lines.join("\n") + "\n")?;
    Ok(())
}
