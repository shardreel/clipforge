//! Preflight: does this machine have the disk, RAM and CPU to run a job?
use crate::{paths, platform};
use serde::Serialize;
use std::path::Path;
use sysinfo::{Disks, System};

const GB: u64 = 1024 * 1024 * 1024;
const MIN_DISK_GB: u64 = 10;
const REC_DISK_GB: u64 = 30;
const MIN_RAM_GB: u64 = 4;
const REC_RAM_GB: u64 = 16;
const MIN_CORES: usize = 4;
const REC_CORES: usize = 8;

#[derive(Serialize)]
pub struct Check {
    pub name: &'static str,
    /// pass | warn | fail
    pub level: &'static str,
    pub detail: String,
}

#[derive(Serialize)]
pub struct Report {
    pub platform: &'static str,
    pub whisper_backend: &'static str,
    pub ram_gb: u64,
    pub free_disk_gb: u64,
    pub cores: usize,
    pub checks: Vec<Check>,
    /// False when any check failed; jobs are blocked.
    pub ok: bool,
}

fn free_bytes(path: &Path) -> u64 {
    Disks::new_with_refreshed_list()
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
        .unwrap_or(0)
}

fn level(value: u64, min: u64, rec: u64, block: bool) -> &'static str {
    if value < min {
        if block { "fail" } else { "warn" }
    } else if value < rec {
        "warn"
    } else {
        "pass"
    }
}

pub fn run() -> Report {
    let mut sys = System::new();
    sys.refresh_memory();
    let ram_gb = sys.total_memory() / GB;
    let cores = System::physical_core_count().unwrap_or(1);
    let free_gb = free_bytes(&paths::data_dir()) / GB;
    let os = platform::os();

    let mut checks = vec![
        Check {
            name: "Free disk",
            level: level(free_gb, MIN_DISK_GB, REC_DISK_GB, true),
            detail: format!("{free_gb} GB free in {} (need {MIN_DISK_GB}, recommended {REC_DISK_GB})", paths::data_dir().display()),
        },
        Check {
            name: "Memory",
            level: level(ram_gb, MIN_RAM_GB, REC_RAM_GB, true),
            detail: format!(
                "{ram_gb} GB RAM (need {MIN_RAM_GB}, recommended {REC_RAM_GB}; under 8 GB use the small whisper model and light mode)"
            ),
        },
        Check {
            name: "CPU cores",
            level: level(cores as u64, MIN_CORES as u64, REC_CORES as u64, false),
            detail: format!("{cores} physical cores (need {MIN_CORES}, recommended {REC_CORES})"),
        },
    ];

    if os == platform::Os::Mac {
        let arm = cfg!(target_arch = "aarch64");
        checks.push(Check {
            name: "Apple Silicon",
            level: if arm { "pass" } else { "fail" },
            detail: if arm { "arm64 Mac".into() } else { "Intel Mac is not supported".into() },
        });
    } else {
        let gpu = platform::has_nvidia();
        let enc = crate::media::pick_encoder("h264", "high", false);
        checks.push(Check {
            name: "GPU",
            level: if gpu { "pass" } else { "warn" },
            detail: if gpu {
                "NVIDIA GPU found".into()
            } else if enc.hardware() {
                format!("no NVIDIA GPU; encoding uses the integrated GPU ({}), transcription runs on CPU. Light mode is on by default", enc.name)
            } else {
                "no NVIDIA GPU or VAAPI device; encoding and transcription run on CPU. Light mode is on by default".into()
            },
        });
    }

    if os == platform::Os::Wsl {
        let slow = platform::on_windows_mount(&paths::data_dir());
        checks.push(Check {
            name: "WSL storage",
            level: if slow { "warn" } else { "pass" },
            detail: if slow {
                "data dir is on a Windows drive (/mnt); use the Linux filesystem for speed".into()
            } else {
                "data dir is on the Linux filesystem".into()
            },
        });
    }

    let ok = checks.iter().all(|c| c.level != "fail");
    Report {
        platform: os.as_str(),
        whisper_backend: platform::whisper_backend(),
        ram_gb,
        free_disk_gb: free_gb,
        cores,
        checks,
        ok,
    }
}

/// First failing check as an error message, or None when jobs may run.
pub fn blocker() -> Option<String> {
    let r = run();
    r.checks.into_iter().find(|c| c.level == "fail").map(|c| format!("{}: {}", c.name, c.detail))
}
