//! Platform detection: macOS, Linux, or Linux under WSL. Default dirs, install hints, GPU detect.
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    Mac,
    Linux,
    Wsl,
}

impl Os {
    pub fn as_str(self) -> &'static str {
        match self {
            Os::Mac => "macos",
            Os::Linux => "linux",
            Os::Wsl => "wsl",
        }
    }
}

pub fn os() -> Os {
    if cfg!(target_os = "macos") {
        Os::Mac
    } else if is_wsl() {
        Os::Wsl
    } else {
        Os::Linux
    }
}

fn is_wsl() -> bool {
    std::fs::read_to_string("/proc/version").map(|v| v.to_lowercase().contains("microsoft")).unwrap_or(false)
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

/// Where rendered clips go by default.
pub fn default_out_dir() -> PathBuf {
    let folder = if os() == Os::Mac { "Movies" } else { "Videos" };
    home().join(folder).join("clipforge")
}

/// True when `path` is on a Windows drive mounted into WSL (slow file I/O).
pub fn on_windows_mount(path: &Path) -> bool {
    os() == Os::Wsl && path.starts_with("/mnt")
}

/// How to install a missing tool on this OS.
pub fn install_hint(binary: &str) -> String {
    match os() {
        Os::Mac => format!("brew install {binary}"),
        _ if binary == "yt-dlp" => "pipx install yt-dlp".into(),
        _ => format!("sudo apt install {binary}"),
    }
}

pub fn has_nvidia() -> bool {
    Command::new("nvidia-smi").arg("-L").output().map(|o| o.status.success()).unwrap_or(false)
}

/// No Apple Silicon and no NVIDIA GPU: whisper runs on the CPU, so default to light settings.
pub fn lite_recommended() -> bool {
    os() != Os::Mac && !has_nvidia() && !cfg!(feature = "cuda")
}

/// Whisper backend compiled into this binary.
pub fn whisper_backend() -> &'static str {
    if cfg!(target_os = "macos") {
        "Metal"
    } else if cfg!(feature = "cuda") {
        "CUDA"
    } else {
        "CPU"
    }
}
