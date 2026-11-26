//! Job store: queue (one at a time), live event broadcast for SSE, JSON persistence.
use crate::ingest::DlProgress;
use crate::paths;
use crate::picker::Moment;
use crate::pipeline::{self, ClipMeta, Event, Options};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::{Semaphore, broadcast};

const MAX_LOG_LINES: usize = 2000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub source: String,
    pub options: Options,
    pub status: String, // queued | running | review | done | failed | cancelled
    pub stage: String,
    #[serde(default)]
    pub progress: Option<DlProgress>,
    pub log: Vec<String>,
    pub clips: Vec<ClipMeta>,
    pub out_dir: Option<PathBuf>,
    pub error: Option<String>,
    pub created_at: u64,
    /// Moments waiting for the user while status is "review".
    #[serde(default)]
    pub review: Option<Vec<Moment>>,
    /// How many of `review` to tick by default (the clip count asked for).
    #[serde(default)]
    pub review_preselect: usize,
}

#[derive(Default)]
struct State {
    jobs: HashMap<String, Job>,
    tx: HashMap<String, broadcast::Sender<String>>,
    /// Queued jobs the user cancelled before they started.
    cancelled: HashSet<String>,
    /// Where to send the approved moments of a job in review.
    review_tx: HashMap<String, std::sync::mpsc::Sender<Vec<Moment>>>,
    /// Last download progress step (10% buckets) printed to the terminal, per job.
    printed_pct: HashMap<String, u32>,
}

/// Mirror job events to the terminal, so a run can be followed (and errors seen) without the UI.
fn print_event(st: &mut State, job: &Job, ev: &Value) {
    let id = &job.id;
    let text = |k: &str| ev[k].as_str().unwrap_or("").to_string();
    match ev["type"].as_str() {
        Some("stage") => tracing::info!("[{id}] stage: {}", text("stage")),
        Some("log") => tracing::info!("[{id}] {}", text("line")),
        Some("review") => tracing::info!(
            "[{id}] waiting for review in the browser: {} moments",
            ev["moments"].as_array().map_or(0, |a| a.len())
        ),
        // only the main download; per-clip pieces are short and would flood the terminal
        Some("progress") if job.stage == "fetch" => {
            let p = &ev["progress"];
            let pct = p["percent"].as_f64().unwrap_or(0.0);
            let bucket = (pct / 10.0) as u32;
            if st.printed_pct.insert(id.clone(), bucket) != Some(bucket) {
                let mb = p["downloaded"].as_f64().unwrap_or(0.0) / 1e6;
                let speed = p["speed"].as_f64().map(|s| format!(" at {:.1} MB/s", s / 1e6)).unwrap_or_default();
                tracing::info!("[{id}] downloading {}: {pct:.0}% ({mb:.0} MB){speed}", p["kind"].as_str().unwrap_or(""));
            }
        }
        Some("status") => match ev["status"].as_str().unwrap_or("") {
            "failed" => tracing::error!("[{id}] FAILED: {}", text("error")),
            "cancelled" => tracing::warn!("[{id}] cancelled"),
            "done" => tracing::info!(
                "[{id}] done: {} clips in {}",
                job.clips.len(),
                job.out_dir.as_ref().map_or(String::new(), |d| d.display().to_string())
            ),
            s => tracing::info!("[{id}] status: {s}"),
        },
        _ => {}
    }
}

pub struct Jobs {
    state: Mutex<State>,
    gate: Semaphore,
}

fn jobs_dir() -> PathBuf {
    paths::data_dir().join("jobs")
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Jobs {
    pub fn load() -> Arc<Self> {
        let mut state = State::default();
        for e in std::fs::read_dir(jobs_dir()).into_iter().flatten().flatten() {
            let Some(mut job) = std::fs::read_to_string(e.path()).ok().and_then(|t| serde_json::from_str::<Job>(&t).ok())
            else {
                continue;
            };
            if ["queued", "running", "review"].contains(&job.status.as_str()) {
                job.status = "failed".into();
                job.error = Some("interrupted by restart".into());
            }
            state.jobs.insert(job.id.clone(), job);
        }
        Arc::new(Self { state: Mutex::new(state), gate: Semaphore::new(1) })
    }

    pub fn list(&self) -> Vec<Job> {
        let mut v: Vec<Job> = self.state.lock().unwrap().jobs.values().cloned().collect();
        v.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        v
    }

    pub fn busy(&self) -> bool {
        self.state.lock().unwrap().jobs.values().any(|j| ["queued", "running", "review"].contains(&j.status.as_str()))
    }

    pub fn get(&self, id: &str) -> Option<Job> {
        self.state.lock().unwrap().jobs.get(id).cloned()
    }

    /// Snapshot plus a receiver for later events, taken under one lock so none are missed.
    pub fn subscribe(&self, id: &str) -> Option<(Job, broadcast::Receiver<String>)> {
        let mut st = self.state.lock().unwrap();
        let job = st.jobs.get(id)?.clone();
        let rx = st.tx.entry(id.to_string()).or_insert_with(|| broadcast::channel(512).0).subscribe();
        Some((job, rx))
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut Job), ev: Value) {
        let mut st = self.state.lock().unwrap();
        let Some(job) = st.jobs.get_mut(id) else { return };
        f(job);
        let snapshot = job.clone();
        print_event(&mut st, &snapshot, &ev);
        if let Some(tx) = st.tx.get(id) {
            let _ = tx.send(ev.to_string());
        }
        drop(st);
        if ["done", "failed", "cancelled"].contains(&snapshot.status.as_str()) {
            let _ = std::fs::create_dir_all(jobs_dir());
            let _ = serde_json::to_string_pretty(&snapshot)
                .map(|t| std::fs::write(jobs_dir().join(format!("{id}.json")), t));
        }
    }

    /// Cancel a queued or running job. Returns false when it already finished.
    pub fn cancel(&self, id: &str) -> Option<bool> {
        let mut st = self.state.lock().unwrap();
        let status = st.jobs.get(id)?.status.clone();
        Some(match status.as_str() {
            "queued" => {
                st.cancelled.insert(id.to_string());
                true
            }
            "running" | "review" => {
                st.review_tx.remove(id);
                drop(st);
                crate::cancel::request();
                true
            }
            _ => false,
        })
    }

    fn begin_review(&self, id: &str, moments: Vec<Moment>, preselect: usize, reply: std::sync::mpsc::Sender<Vec<Moment>>) {
        self.state.lock().unwrap().review_tx.insert(id.to_string(), reply);
        let ev = json!({"type": "review", "moments": moments, "preselect": preselect});
        self.update(
            id,
            |j| {
                j.status = "review".into();
                j.review = Some(moments);
                j.review_preselect = preselect;
                j.progress = None;
            },
            ev,
        );
    }

    /// The user's approved moments for a job in review. Err when it isn't waiting for one.
    pub fn submit_review(&self, id: &str, moments: Vec<Moment>) -> Result<(), &'static str> {
        if moments.is_empty() {
            return Err("approve at least one moment");
        }
        let tx = {
            let mut st = self.state.lock().unwrap();
            match st.jobs.get(id) {
                None => return Err("job not found"),
                Some(j) if j.status != "review" => return Err("job is not waiting for review"),
                _ => {}
            }
            st.review_tx.remove(id).ok_or("job is not waiting for review")?
        };
        tx.send(moments).map_err(|_| "job is no longer waiting")?;
        self.update(
            id,
            |j| {
                j.status = "running".into();
                j.review = None;
            },
            json!({"type": "status", "status": "running"}),
        );
        Ok(())
    }

    fn mark_cancelled(&self, id: &str) {
        self.update(
            id,
            |j| {
                j.status = "cancelled".into();
                j.error = Some("cancelled".into());
                j.progress = None;
            },
            json!({"type": "status", "status": "cancelled", "error": "cancelled"}),
        );
    }

    pub fn submit(self: &Arc<Self>, source: String, options: Options) -> Job {
        let id = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
        let job = Job {
            id: id.clone(), source: source.clone(), options: options.clone(), status: "queued".into(),
            stage: "queued".into(), progress: None, log: vec![], clips: vec![], out_dir: None, error: None, created_at: now(),
            review: None, review_preselect: 0,
        };
        self.state.lock().unwrap().jobs.insert(id.clone(), job.clone());
        tracing::info!("[{id}] new job: {source}");
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let Ok(_permit) = this.gate.acquire().await else { return };
            if this.state.lock().unwrap().cancelled.remove(&id) {
                this.mark_cancelled(&id);
                return;
            }
            crate::cancel::reset();
            this.update(&id, |j| j.status = "running".into(), json!({"type": "status", "status": "running"}));
            let runner = Arc::clone(&this);
            let rid = id.clone();
            let result = tokio::task::spawn_blocking(move || {
                let emit = |ev: Event| match ev {
                    Event::Stage(s) => runner.update(
                        &rid,
                        |j| {
                            j.stage = s.to_string();
                            j.progress = None;
                        },
                        json!({"type": "stage", "stage": s}),
                    ),
                    Event::Progress(p) => runner.update(
                        &rid,
                        |j| j.progress = Some(p.clone()),
                        json!({"type": "progress", "progress": p}),
                    ),
                    Event::Review { moments, preselect, reply } => runner.begin_review(&rid, moments, preselect, reply),
                    Event::Log(line) => runner.update(
                        &rid,
                        |j| {
                            j.log.push(line.clone());
                            if j.log.len() > MAX_LOG_LINES {
                                j.log.remove(0);
                            }
                        },
                        json!({"type": "log", "line": line}),
                    ),
                };
                pipeline::process(&source, &options, &emit)
            })
            .await;
            this.state.lock().unwrap().review_tx.remove(&id);
            match result {
                Ok(Ok(out)) => this.update(
                    &id,
                    |j| {
                        j.status = "done".into();
                        j.stage = "done".into();
                        j.out_dir = Some(out.out_dir);
                        j.clips = out.clips;
                    },
                    json!({"type": "status", "status": "done"}),
                ),
                Ok(Err(e)) if crate::cancel::requested() || e.downcast_ref::<crate::cancel::Cancelled>().is_some() => {
                    this.mark_cancelled(&id)
                }
                other => {
                    let msg = match other {
                        Ok(Err(e)) => format!("{e:#}"),
                        Err(e) => format!("job crashed: {e}"),
                        Ok(Ok(_)) => unreachable!(),
                    };
                    this.update(
                        &id,
                        |j| {
                            j.status = "failed".into();
                            j.error = Some(msg.clone());
                        },
                        json!({"type": "status", "status": "failed", "error": msg}),
                    )
                }
            }
        });
        job
    }
}
