//! Fast time-range downloads from YouTube-style fragmented MP4 (DASH over plain HTTPS).
//!
//! yt-dlp's `--download-sections` hands the URL to ffmpeg, which reads it over one open-ended
//! HTTP request. YouTube throttles those to about playback speed (~1 MB/s). Bounded 10 MB range
//! requests are not throttled the same way, and several in parallel fill the link. So: read the
//! file's `sidx` index once, map the wanted time range to byte ranges, fetch those in parallel
//! and stitch init segment + fragments into a playable file.
use crate::ingest::DlProgress;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Parallel connections per download. Four fill most home links without tripping YouTube.
pub const CONNECTIONS: usize = 4;
/// Bytes per range request: small enough to dodge the open-ended-request throttle.
const PIECE: u64 = 10 << 20;
const ATTEMPTS: usize = 4;

/// One media stream yt-dlp resolved (video-only or audio-only).
#[derive(Clone, Debug, Deserialize)]
pub struct Stream {
    pub url: String,
    #[serde(default)]
    pub ext: String,
    #[serde(default)]
    pub protocol: String,
    #[serde(default)]
    pub http_headers: HashMap<String, String>,
}

/// One entry of the `sidx` box: a self-contained fragment starting on a keyframe.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub offset: u64,
    pub size: u64,
}

/// Init segment (ftyp + moov) plus the time-to-byte map of a fragmented MP4.
#[derive(Clone, Debug)]
pub struct Index {
    pub init: Vec<u8>,
    pub segments: Vec<Segment>,
}

impl Index {
    /// Contiguous segments covering `[start, end)`.
    pub fn covering(&self, start: f64, end: f64) -> &[Segment] {
        let first = self.segments.iter().position(|s| s.end > start).unwrap_or(self.segments.len());
        let last = self.segments.iter().rposition(|s| s.start < end).map_or(first, |i| i + 1);
        &self.segments[first..last.max(first)]
    }
}

fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(120))
        .build()?)
}

fn get_range(c: &reqwest::blocking::Client, s: &Stream, from: u64, to_incl: u64) -> Result<reqwest::blocking::Response> {
    let mut req = c.get(&s.url).header("Range", format!("bytes={from}-{to_incl}"));
    for (k, v) in &s.http_headers {
        req = req.header(k.as_str(), v.as_str());
    }
    let resp = req.send()?;
    if resp.status().as_u16() != 206 {
        bail!("HTTP {} for a range request", resp.status().as_u16());
    }
    Ok(resp)
}

fn read_range(c: &reqwest::blocking::Client, s: &Stream, from: u64, len: u64) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(len as usize);
    get_range(c, s, from, from + len - 1)?.read_to_end(&mut buf)?;
    Ok(buf)
}

fn be_u32(b: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_be_bytes(b.get(at..at + 4).context("truncated box")?.try_into()?))
}

fn be_u64(b: &[u8], at: usize) -> Result<u64> {
    Ok(u64::from_be_bytes(b.get(at..at + 8).context("truncated box")?.try_into()?))
}

/// Parse a `sidx` box body (after the 8-byte header). `media_start` is the file offset of the
/// first byte after the box.
pub fn parse_sidx(body: &[u8], media_start: u64) -> Result<Vec<Segment>> {
    let version = *body.first().context("empty sidx")?;
    let timescale = be_u32(body, 8)? as f64;
    if timescale <= 0.0 {
        bail!("sidx timescale is 0");
    }
    let (ept, first_offset, mut at) = if version == 0 {
        (be_u32(body, 12)? as u64, be_u32(body, 16)? as u64, 20)
    } else {
        (be_u64(body, 12)?, be_u64(body, 20)?, 28)
    };
    let count = u16::from_be_bytes(body.get(at + 2..at + 4).context("truncated sidx")?.try_into()?) as usize;
    at += 4;
    let (mut t, mut off) = (ept, media_start + first_offset);
    let mut segs = Vec::with_capacity(count);
    for _ in 0..count {
        let r = be_u32(body, at)?;
        if r & 0x8000_0000 != 0 {
            bail!("nested sidx not supported");
        }
        let size = (r & 0x7fff_ffff) as u64;
        let dur = be_u32(body, at + 4)? as u64;
        segs.push(Segment { start: t as f64 / timescale, end: (t + dur) as f64 / timescale, offset: off, size });
        t += dur;
        off += size;
        at += 12;
    }
    Ok(segs)
}

/// Read top-level boxes until `sidx`; everything before it except `sidx` is the init segment.
pub fn read_index(s: &Stream) -> Result<Index> {
    let c = client()?;
    let mut head = read_range(&c, s, 0, 256 << 10)?;
    let (mut pos, mut saw_moov) = (0usize, false);
    loop {
        if pos + 8 > head.len() {
            bail!("no sidx in the first {} KB", head.len() >> 10);
        }
        let mut size = be_u32(&head, pos)? as u64;
        let kind: [u8; 4] = head[pos + 4..pos + 8].try_into()?;
        let mut hdr = 8;
        if size == 1 {
            size = be_u64(&head, pos + 8)?;
            hdr = 16;
        }
        if size < hdr as u64 {
            bail!("bad box size");
        }
        let end = pos as u64 + size;
        if end as usize > head.len() {
            if end > 8 << 20 {
                bail!("index box too large");
            }
            let more = read_range(&c, s, head.len() as u64, end - head.len() as u64)?;
            head.extend_from_slice(&more);
        }
        if &kind == b"sidx" {
            if !saw_moov {
                bail!("moov does not precede sidx");
            }
            let segments = parse_sidx(&head[pos + hdr..end as usize], end)?;
            if segments.is_empty() {
                bail!("empty sidx");
            }
            return Ok(Index { init: head[..pos].to_vec(), segments });
        }
        if &kind == b"moof" || &kind == b"mdat" {
            bail!("media before any sidx");
        }
        saw_moov |= &kind == b"moov";
        pos = end as usize;
    }
}

/// Byte ranges to fetch for one stream, written to `out` at `dst` offsets.
struct Job<'a> {
    stream: &'a Stream,
    out: &'a File,
    pieces: Vec<(u64, u64, u64)>, // (src offset, len, dst offset)
}

fn plan<'a>(stream: &'a Stream, segs: &[Segment], dst_base: u64, out: &'a File) -> Job<'a> {
    let (src, len) = (segs[0].offset, segs.iter().map(|s| s.size).sum::<u64>());
    let pieces = (0..len.div_ceil(PIECE))
        .map(|i| {
            let o = i * PIECE;
            (src + o, PIECE.min(len - o), dst_base + o)
        })
        .collect();
    Job { stream, out, pieces }
}

/// Fetch every piece of every job over `CONNECTIONS` parallel range requests.
fn fetch_all(jobs: &[Job], on_progress: &dyn Fn(DlProgress)) -> Result<()> {
    let total: u64 = jobs.iter().flat_map(|j| &j.pieces).map(|p| p.1).sum();
    let work: Vec<(usize, (u64, u64, u64))> =
        jobs.iter().enumerate().flat_map(|(j, job)| job.pieces.iter().map(move |p| (j, *p))).collect();
    let next = AtomicUsize::new(0);
    let done = AtomicU64::new(0);
    let running = AtomicUsize::new(CONNECTIONS.min(work.len()));
    let error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let started = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..CONNECTIONS.min(work.len()) {
            sc.spawn(|| {
                let c = match client() {
                    Ok(c) => c,
                    Err(e) => {
                        *error.lock().unwrap() = Some(e);
                        running.fetch_sub(1, Ordering::SeqCst);
                        return;
                    }
                };
                while error.lock().unwrap().is_none() && !crate::cancel::requested() {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(&(j, (src, len, dst))) = work.get(i) else { break };
                    let job = &jobs[j];
                    let mut last = None;
                    for attempt in 0..ATTEMPTS {
                        if attempt > 0 {
                            std::thread::sleep(Duration::from_millis(500 << attempt));
                        }
                        let mut got = 0u64;
                        let r = (|| -> Result<()> {
                            let mut resp = get_range(&c, job.stream, src, src + len - 1)?;
                            let mut buf = vec![0u8; 256 << 10];
                            while got < len {
                                let n = resp.read(&mut buf)?;
                                if n == 0 {
                                    bail!("connection closed after {got} of {len} bytes");
                                }
                                let n = n.min((len - got) as usize);
                                job.out.write_all_at(&buf[..n], dst + got)?;
                                got += n as u64;
                                done.fetch_add(n as u64, Ordering::Relaxed);
                            }
                            Ok(())
                        })();
                        // a failed attempt restarts the piece; take back its partial bytes
                        if r.is_err() {
                            done.fetch_sub(got, Ordering::Relaxed);
                        }
                        match r {
                            Ok(()) => {
                                last = None;
                                break;
                            }
                            Err(e) => last = Some(e),
                        }
                    }
                    if let Some(e) = last {
                        error.lock().unwrap().get_or_insert(e);
                        break;
                    }
                }
                running.fetch_sub(1, Ordering::SeqCst);
            });
        }
        while running.load(Ordering::SeqCst) > 0 {
            std::thread::sleep(Duration::from_millis(250));
            let d = done.load(Ordering::Relaxed);
            let secs = started.elapsed().as_secs_f64().max(0.1);
            let speed = d as f64 / secs;
            on_progress(DlProgress {
                percent: (d as f64 / total.max(1) as f64 * 100.0).min(100.0),
                downloaded: d,
                total: Some(total),
                speed: Some(speed),
                eta: (speed > 0.0).then(|| (total.saturating_sub(d) as f64 / speed) as u64),
                kind: "video".into(),
            });
        }
    });
    crate::cancel::check()?;
    match error.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Write init + the fragments covering the range into `path`; returns the fetch job.
fn prepare<'a>(stream: &'a Stream, index: &Index, segs: &[Segment], out: &'a File) -> Result<Job<'a>> {
    out.set_len(index.init.len() as u64 + segs.iter().map(|s| s.size).sum::<u64>())?;
    out.write_all_at(&index.init, 0)?;
    Ok(plan(stream, segs, index.init.len() as u64, out))
}

/// Download `[start, start + len)` of separate video and audio streams and mux them into
/// `dest` (stream copy). Returns the time in the source where the file starts (a keyframe at
/// or before `start`).
pub fn fetch_range(
    video: (&Stream, &Index), audio: (&Stream, &Index), start: f64, len: f64, dir: &Path, dest: &Path,
    on_progress: &dyn Fn(DlProgress),
) -> Result<f64> {
    let vsegs = video.1.covering(start, start + len);
    let (Some(vfirst), Some(vlast)) = (vsegs.first(), vsegs.last()) else { bail!("range is outside the video") };
    let asegs = audio.1.covering(vfirst.start, vlast.end);
    if asegs.is_empty() {
        bail!("range is outside the audio");
    }
    let vpath = dir.join(format!("v.{}", if video.0.ext.is_empty() { "mp4" } else { &video.0.ext }));
    let apath = dir.join(format!("a.{}", if audio.0.ext.is_empty() { "m4a" } else { &audio.0.ext }));
    let (vf, af) = (File::create(&vpath)?, File::create(&apath)?);
    let jobs = [prepare(video.0, video.1, vsegs, &vf)?, prepare(audio.0, audio.1, asegs, &af)?];
    fetch_all(&jobs, on_progress)?;
    drop((vf, af));
    // Align audio and video to the exact same start timestamp (vfirst.start).
    // Stream copy (-c copy) in MP4 ignores negative itsoffset and drops edit-list sync.
    // If audio started before video, trim the lead-in; if video started before, delay audio.
    let diff = vfirst.start - asegs[0].start;
    let mut args: Vec<String> = vec![
        "-y".into(), "-v".into(), "error".into(),
        "-i".into(), vpath.to_string_lossy().into_owned(),
    ];
    if diff.abs() > 0.005 {
        if diff > 0.0 {
            args.extend([
                "-ss".into(), format!("{diff:.4}"),
                "-i".into(), apath.to_string_lossy().into_owned(),
                "-map".into(), "0:v:0".into(), "-map".into(), "1:a:0".into(),
                "-c:v".into(), "copy".into(), "-c:a".into(), "aac".into(), "-b:a".into(), "192k".into(),
            ]);
        } else {
            let ms = (-diff * 1000.0).round() as u64;
            args.extend([
                "-i".into(), apath.to_string_lossy().into_owned(),
                "-af".into(), format!("adelay={ms}|{ms}"),
                "-map".into(), "0:v:0".into(), "-map".into(), "1:a:0".into(),
                "-c:v".into(), "copy".into(), "-c:a".into(), "aac".into(), "-b:a".into(), "192k".into(),
            ]);
        }
    } else {
        args.extend([
            "-i".into(), apath.to_string_lossy().into_owned(),
            "-map".into(), "0:v:0".into(), "-map".into(), "1:a:0".into(),
            "-c".into(), "copy".into(),
        ]);
    }
    args.extend([
        "-shortest".into(), "-movflags".into(), "+faststart".into(),
        dest.to_string_lossy().into_owned(),
    ]);
    let muxed = crate::media::run(&args, None);
    let _ = std::fs::remove_file(&vpath);
    let _ = std::fs::remove_file(&apath);
    muxed?;
    Ok(vfirst.start)
}

/// Download a whole stream (init, index and every fragment) over parallel range requests.
pub fn fetch_whole(stream: &Stream, index: &Index, dest: &Path, on_progress: &dyn Fn(DlProgress)) -> Result<()> {
    let last = index.segments.last().context("empty index")?;
    let size = last.offset + last.size;
    let out = File::create(dest)?;
    out.set_len(size)?;
    let whole = [Segment { start: 0.0, end: last.end, offset: 0, size }];
    fetch_all(&[plan(stream, &whole, 0, &out)], on_progress)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sidx_v0(timescale: u32, ept: u32, first_offset: u32, refs: &[(u32, u32)]) -> Vec<u8> {
        let mut b = vec![0, 0, 0, 0];
        b.extend(1u32.to_be_bytes());
        b.extend(timescale.to_be_bytes());
        b.extend(ept.to_be_bytes());
        b.extend(first_offset.to_be_bytes());
        b.extend(0u16.to_be_bytes());
        b.extend((refs.len() as u16).to_be_bytes());
        for (size, dur) in refs {
            b.extend(size.to_be_bytes());
            b.extend(dur.to_be_bytes());
            b.extend(0x9000_0000u32.to_be_bytes());
        }
        b
    }

    #[test]
    fn sidx_maps_time_to_bytes() {
        let body = sidx_v0(1000, 0, 0, &[(100, 5000), (200, 5000), (300, 4000)]);
        let segs = parse_sidx(&body, 1000).unwrap();
        assert_eq!(segs[0], Segment { start: 0.0, end: 5.0, offset: 1000, size: 100 });
        assert_eq!(segs[2], Segment { start: 10.0, end: 14.0, offset: 1300, size: 300 });
        let idx = Index { init: vec![], segments: segs };
        let c = idx.covering(6.0, 11.0);
        assert_eq!((c.len(), c[0].start, c[1].end), (2, 5.0, 14.0));
        assert_eq!(idx.covering(0.0, 100.0).len(), 3);
        assert!(idx.covering(20.0, 30.0).is_empty());
    }

    #[test]
    fn nested_sidx_is_rejected() {
        let body = sidx_v0(1000, 0, 0, &[(0x8000_0010, 5000)]);
        assert!(parse_sidx(&body, 0).is_err());
    }

    #[test]
    fn pieces_tile_the_range() {
        let f = tempfile::tempfile().unwrap();
        let s = Stream { url: String::new(), ext: String::new(), protocol: String::new(), http_headers: HashMap::new() };
        let segs = [Segment { start: 0.0, end: 1.0, offset: 50, size: PIECE * 2 + 7 }];
        let job = plan(&s, &segs, 10, &f);
        assert_eq!(job.pieces, vec![(50, PIECE, 10), (50 + PIECE, PIECE, 10 + PIECE), (50 + 2 * PIECE, 7, 10 + 2 * PIECE)]);
    }
}

