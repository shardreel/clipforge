use crate::platform;
use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

const EXTRA_BIN_DIRS: [&str; 2] = ["/opt/homebrew/bin", "/usr/local/bin"];
/// Keg-only ffmpeg build with libass (captions). Preferred when installed.
const FFMPEG_FULL_BIN: &str = "/opt/homebrew/opt/ffmpeg-full/bin";

pub fn require(binary: &str) -> Result<PathBuf> {
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    let mac = cfg!(target_os = "macos");
    let mut dirs: Vec<PathBuf> = if mac { vec![PathBuf::from(FFMPEG_FULL_BIN)] } else { vec![] };
    dirs.extend(std::env::split_paths(&path_var));
    if mac {
        dirs.extend(EXTRA_BIN_DIRS.iter().map(PathBuf::from));
    }
    dirs.into_iter()
        .map(|d| d.join(binary))
        .find(|p| p.is_file())
        .with_context(|| format!("'{binary}' not found on PATH ({})", platform::install_hint(binary)))
}

fn capture(binary: &str, args: &[&str]) -> String {
    require(binary)
        .and_then(|b| Ok(Command::new(b).args(args).output()?))
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

fn ffmpeg_encoders() -> &'static str {
    static C: OnceLock<String> = OnceLock::new();
    C.get_or_init(|| capture("ffmpeg", &["-hide_banner", "-encoders"]))
}

fn ffmpeg_filters() -> &'static str {
    static C: OnceLock<String> = OnceLock::new();
    C.get_or_init(|| capture("ffmpeg", &["-hide_banner", "-filters"]))
}

pub fn has_libass() -> bool {
    ffmpeg_filters().contains(" subtitles ")
}

#[derive(Clone, Debug, Serialize)]
pub struct Encoder {
    pub name: String,
    pub video_args: Vec<String>,
    pub decode_args: Vec<String>,
    /// Global args that must come before the inputs (the VAAPI device).
    pub device_args: Vec<String>,
    /// Filter appended to the graph to hand frames to the encoder (VAAPI upload).
    pub upload: Option<String>,
    /// Software encoder to retry a clip with when this hardware encoder fails on it.
    #[serde(skip)]
    pub software: Option<Box<Encoder>>,
}

impl Encoder {
    pub fn hardware(&self) -> bool {
        ["videotoolbox", "nvenc", "vaapi"].iter().any(|h| self.name.contains(h))
    }

    /// Frames are filtered and encoded on the CPU: parallel renders compete for the same cores.
    pub fn cpu_bound(&self) -> bool {
        !self.hardware() || self.name.contains("vaapi")
    }

    fn new(name: String, video_args: Vec<String>, decode_args: Vec<String>) -> Self {
        Self { name, video_args, decode_args, device_args: vec![], upload: None, software: None }
    }
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn bitrate(quality: &str) -> &'static str {
    match quality {
        "low" => "5M",
        "medium" => "8M",
        _ => "12M",
    }
}

/// ffmpeg lists encoders it was built with, not ones the driver supports (e.g. hevc_vaapi on
/// older Intel iGPUs, NVENC with a stale driver). Encode a few frames once per encoder to be sure.
fn encoder_works(enc: &Encoder) -> bool {
    static SEEN: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    if let Some(ok) = seen.lock().unwrap().get(&enc.name) {
        return *ok;
    }
    let ok = require("ffmpeg").is_ok_and(|ffmpeg| {
        let mut cmd = Command::new(ffmpeg);
        cmd.args(["-hide_banner", "-v", "error"]).args(&enc.device_args);
        cmd.args(["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30", "-frames:v", "5"]);
        if let Some(u) = &enc.upload {
            cmd.args(["-vf", u]);
        }
        cmd.args(&enc.video_args).args(["-f", "null", "-"]);
        cmd.output().is_ok_and(|o| o.status.success())
    });
    seen.lock().unwrap().insert(enc.name.clone(), ok);
    ok
}

/// First DRM render node (Intel/AMD integrated or discrete GPU), or CLIPFORGE_VAAPI_DEVICE.
fn vaapi_device() -> Option<PathBuf> {
    if let Some(dev) = std::env::var_os("CLIPFORGE_VAAPI_DEVICE") {
        return Some(PathBuf::from(dev));
    }
    let mut nodes: Vec<PathBuf> = std::fs::read_dir("/dev/dri")
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("renderD")))
        .collect();
    nodes.sort();
    nodes.into_iter().next()
}

/// VAAPI on Intel/AMD GPUs, including integrated ones. Constant QP: the rate-control mode
/// every VAAPI driver supports (older Intel iGPUs reject bitrate targets).
fn vaapi_encoder(codec: &str, quality: &str) -> Option<Encoder> {
    let name = format!("{codec}_vaapi");
    if !ffmpeg_encoders().contains(&name) {
        return None;
    }
    let dev = vaapi_device()?;
    let qp = match quality {
        "low" => "27",
        "medium" => "24",
        _ => "21",
    };
    let mut args = strs(&["-c:v", &name, "-qp", qp]);
    if codec == "hevc" {
        args.extend(strs(&["-tag:v", "hvc1"]));
    }
    let mut enc = Encoder::new(name, args, vec![]);
    enc.device_args = vec!["-vaapi_device".into(), dev.to_string_lossy().into_owned()];
    enc.upload = Some("format=nv12,hwupload".into());
    Some(enc)
}

fn hardware_encoder(codec: &str, quality: &str) -> Option<Encoder> {
    let rate = bitrate(quality);
    if cfg!(target_os = "macos") {
        let vt = format!("{codec}_videotoolbox");
        if !ffmpeg_encoders().contains(&vt) {
            return None;
        }
        let profile = if codec == "h264" { "high" } else { "main" };
        let mut args = strs(&[
            "-c:v", &vt, "-b:v", rate, "-maxrate", rate, "-bufsize", "24M",
            "-profile:v", profile, "-allow_sw", "1", "-realtime", "0",
        ]);
        if codec == "hevc" {
            args.extend(strs(&["-tag:v", "hvc1"]));
        }
        return Some(Encoder::new(vt, args, strs(&["-hwaccel", "videotoolbox"])));
    }
    let nv = format!("{codec}_nvenc");
    if ffmpeg_encoders().contains(&nv) && platform::has_nvidia() {
        let mut args = strs(&["-c:v", &nv, "-preset", "p5", "-b:v", rate, "-maxrate", rate, "-bufsize", "24M"]);
        if codec == "hevc" {
            args.extend(strs(&["-tag:v", "hvc1"]));
        }
        let enc = Encoder::new(nv, args, strs(&["-hwaccel", "cuda"]));
        if encoder_works(&enc) {
            return Some(enc);
        }
    }
    vaapi_encoder(codec, quality).filter(encoder_works)
}

/// Hardware encoder when available: VideoToolbox on Mac, else NVENC, else VAAPI (Intel/AMD,
/// integrated GPUs too). On macOS hardware is always preferred (`force_software` ignored).
pub fn pick_encoder(codec: &str, quality: &str, force_software: bool) -> Encoder {
    let crf = match quality {
        "low" => "25",
        "medium" => "22",
        _ => "19",
    };
    let lib = if codec == "hevc" { "libx265" } else { "libx264" };
    let sw = Encoder::new(lib.into(), strs(&["-c:v", lib, "-preset", "veryfast", "-crf", crf]), vec![]);
    if !force_software || cfg!(target_os = "macos") {
        if let Some(mut enc) = hardware_encoder(codec, quality) {
            enc.software = Some(Box::new(sw));
            return enc;
        }
    }
    sw
}

#[derive(Clone, Debug, Serialize)]
pub struct VideoInfo {
    pub path: PathBuf,
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub has_audio: bool,
}

pub fn probe(path: &Path) -> Result<VideoInfo> {
    let out = Command::new(require("ffprobe")?)
        .args(["-v", "error", "-print_format", "json", "-show_format", "-show_streams"])
        .arg(path)
        .output()?;
    if !out.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let data: Value = serde_json::from_slice(&out.stdout)?;
    let streams = data["streams"].as_array().context("no streams")?;
    let kind = |s: &&Value, k: &str| s["codec_type"].as_str() == Some(k);
    let v = streams.iter().find(|s| kind(s, "video")).context("no video stream")?;
    let has_audio = streams.iter().any(|s| kind(&s, "audio"));
    let rate = v["avg_frame_rate"].as_str().unwrap_or("30/1");
    let (num, den) = rate.split_once('/').unwrap_or((rate, "1"));
    let (num, den): (f64, f64) = (num.parse().unwrap_or(30.0), den.parse().unwrap_or(1.0));
    let fps = if den > 0.0 && num > 0.0 { num / den } else { 30.0 };
    let (mut w, mut h) = (v["width"].as_u64().unwrap_or(0) as u32, v["height"].as_u64().unwrap_or(0) as u32);
    let rot = v["side_data_list"]
        .as_array()
        .and_then(|l| l.iter().filter_map(|d| d["rotation"].as_i64()).last())
        .unwrap_or(0);
    if rot.abs() == 90 || rot.abs() == 270 {
        std::mem::swap(&mut w, &mut h);
    }
    let dur = |x: &Value| x.as_str().and_then(|s| s.parse::<f64>().ok());
    let duration = dur(&data["format"]["duration"]).or_else(|| dur(&v["duration"])).unwrap_or(0.0);
    Ok(VideoInfo { path: path.to_path_buf(), duration, width: w, height: h, fps, has_audio })
}

pub fn run(args: &[String], cwd: Option<&Path>) -> Result<()> {
    // a cancelled job must not start new work after its running programs were killed
    crate::cancel::check()?;
    let mut cmd = Command::new(require("ffmpeg")?);
    cmd.args(args);
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let out = cmd.output()?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: String = err.chars().rev().take(2500).collect::<Vec<_>>().into_iter().rev().collect();
        bail!("ffmpeg failed ({}):\n{tail}", out.status);
    }
    Ok(())
}
