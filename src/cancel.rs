//! Cancelling the running job. Jobs run one at a time, so one process-wide flag is enough.
//! Long steps check it (whisper's abort callback, download and render loops, stage changes),
//! and the external programs doing the heavy work (ffmpeg, yt-dlp) are killed outright.
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

static REQUESTED: AtomicBool = AtomicBool::new(false);

/// Returned (inside anyhow) when a step stops because the user cancelled.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// Called when a job starts.
pub fn reset() {
    REQUESTED.store(false, Ordering::SeqCst);
}

pub fn requested() -> bool {
    REQUESTED.load(Ordering::SeqCst)
}

pub fn check() -> anyhow::Result<()> {
    if requested() { Err(Cancelled.into()) } else { Ok(()) }
}

/// Flag the running job and stop the programs it started.
pub fn request() {
    REQUESTED.store(true, Ordering::SeqCst);
    kill_descendants(std::process::id());
}

/// SIGTERM every process below `pid` (yt-dlp starts ffmpeg, so walk the whole tree).
fn kill_descendants(pid: u32) {
    let Ok(out) = Command::new("pgrep").arg("-P").arg(pid.to_string()).output() else { return };
    for child in String::from_utf8_lossy(&out.stdout).split_whitespace().filter_map(|p| p.parse::<u32>().ok()) {
        kill_descendants(child);
        let _ = Command::new("kill").arg("-TERM").arg(child.to_string()).status();
    }
}
