use crate::jobs::Jobs;
use crate::pipeline::Options;
use crate::settings::Settings;
use crate::storage;
use crate::{media, paths};
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::{Stream, StreamExt};
use tower::ServiceExt;
use tower_http::services::ServeFile;

#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<Jobs>,
}

pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.0.is_server_error() {
            tracing::error!("API {}: {}", self.0.as_u16(), self.1);
        } else {
            tracing::warn!("API {}: {}", self.0.as_u16(), self.1);
        }
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{:#}", e.into()))
    }
}

fn not_found(what: &str) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, format!("{what} not found"))
}

fn bad_request(msg: &str) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.to_string())
}

type ApiResult<T> = Result<T, ApiError>;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/jobs", get(list_jobs).post(create_job))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/jobs/{id}/events", get(job_events))
        .route("/api/jobs/{id}/rerender", post(rerender))
        .route("/api/jobs/{id}/cancel", post(cancel_job))
        .route("/api/jobs/{id}/review", post(review_job))
        .route("/api/jobs/{id}/files/{name}", get(job_file))
        .route("/api/uploads", post(upload).layer(DefaultBodyLimit::disable()))
        .route("/api/storage", get(storage_report))
        .route("/api/storage/{kind}", axum::routing::delete(storage_clear))
        .route("/api/defaults", get(defaults))
        .route("/api/settings", get(get_settings).put(put_settings))
        .route("/api/probe", get(probe))
        .route("/api/frame", get(frame))
        .with_state(state)
}

async fn health() -> Json<Value> {
    let enc = media::pick_encoder("h264", "high", false);
    Json(json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "platform": crate::platform::os().as_str(),
        "preflight": crate::preflight::run(),
        "ffmpeg": media::require("ffmpeg").is_ok(),
        "ffprobe": media::require("ffprobe").is_ok(),
        "yt_dlp": media::require("yt-dlp").is_ok(),
        "libass": media::has_libass(),
        "encoder": enc.name,
        "hardware_encoder": enc.hardware(),
        "whisper_model": (["turbo", "large", "small", "base"].iter().any(|m| crate::transcribe::model_path(m).exists())),
        "lite_recommended": crate::platform::lite_recommended(),
    }))
}

fn settings_view(s: &Settings) -> Value {
    json!({
        "llm": s.llm,
        "llm_model": s.llm_model,
        "base_url": s.base_url,
        "niche": s.niche,
        "has_gemini_key": s.gemini_key().is_some(),
        "has_openrouter_key": s.openrouter_key().is_some(),
    })
}

async fn storage_report() -> ApiResult<Json<Value>> {
    let r = tokio::task::spawn_blocking(storage::report).await?;
    Ok(Json(json!(r)))
}

async fn storage_clear(State(s): State<AppState>, Path(kind): Path<String>) -> ApiResult<Json<Value>> {
    if s.jobs.busy() {
        return Err(ApiError(StatusCode::CONFLICT, "a job is running; wait for it to finish".into()));
    }
    let freed = tokio::task::spawn_blocking(move || storage::clear(&kind))
        .await?
        .map_err(|e| bad_request(&e.to_string()))?;
    Ok(Json(json!({ "freed": freed })))
}

async fn defaults() -> Json<Value> {
    let s = Settings::load();
    let base = Options { llm: s.llm, llm_model: s.llm_model, niche: s.niche, ..Options::default() };
    // CPU-only machines: the base whisper model is ~3x faster than small and ~10x than turbo, and
    // light mode keeps renders to one at a time at 30 fps.
    let lite = tokio::task::spawn_blocking(crate::platform::lite_recommended).await.unwrap_or(false);
    let opts = if lite { Options { lite: true, asr_model: "base".into(), workers: 1, ..base } } else { base };
    Json(json!(opts))
}

async fn get_settings() -> Json<Value> {
    Json(settings_view(&Settings::load()))
}

#[derive(Deserialize)]
struct SettingsUpdate {
    llm: Option<String>,
    llm_model: Option<String>,
    base_url: Option<String>,
    niche: Option<String>,
    gemini_api_key: Option<String>,
    openrouter_api_key: Option<String>,
}

async fn put_settings(Json(b): Json<SettingsUpdate>) -> ApiResult<Json<Value>> {
    let mut s = Settings::load();
    if let Some(v) = b.llm {
        s.llm = v;
    }
    if let Some(v) = b.llm_model {
        s.llm_model = Some(v).filter(|m| !m.is_empty());
    }
    if let Some(v) = b.base_url {
        s.base_url = Some(v).filter(|m| !m.is_empty());
    }
    if let Some(v) = b.niche {
        s.niche = v;
    }
    if let Some(v) = b.gemini_api_key {
        s.gemini_api_key = v;
    }
    if let Some(v) = b.openrouter_api_key {
        s.openrouter_api_key = v;
    }
    s.save()?;
    Ok(Json(settings_view(&s)))
}

#[derive(Deserialize)]
struct NewJob {
    source: String,
    #[serde(default)]
    options: Options,
}

async fn create_job(State(s): State<AppState>, Json(b): Json<NewJob>) -> ApiResult<Json<Value>> {
    if b.source.trim().is_empty() {
        return Err(bad_request("source is required"));
    }
    check_preflight()?;
    Ok(Json(json!(s.jobs.submit(b.source.trim().to_string(), b.options))))
}

fn check_preflight() -> ApiResult<()> {
    match crate::preflight::blocker() {
        Some(msg) => Err(ApiError(StatusCode::UNPROCESSABLE_ENTITY, format!("this machine cannot run jobs: {msg}"))),
        None => Ok(()),
    }
}

async fn list_jobs(State(s): State<AppState>) -> Json<Value> {
    let jobs: Vec<_> = s.jobs.list().into_iter().map(|mut j| { j.log.clear(); j }).collect();
    Json(json!(jobs))
}

async fn get_job(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    Ok(Json(json!(s.jobs.get(&id).ok_or_else(|| not_found("job"))?)))
}

#[derive(Deserialize)]
struct Rerender {
    options: Options,
}

async fn rerender(State(s): State<AppState>, Path(id): Path<String>, Json(b): Json<Rerender>) -> ApiResult<Json<Value>> {
    let job = s.jobs.get(&id).ok_or_else(|| not_found("job"))?;
    check_preflight()?;
    Ok(Json(json!(s.jobs.submit(job.source, b.options))))
}

#[derive(Deserialize)]
struct Review {
    moments: Vec<crate::picker::Moment>,
}

async fn review_job(State(s): State<AppState>, Path(id): Path<String>, Json(b): Json<Review>) -> ApiResult<Json<Value>> {
    s.jobs.submit_review(&id, b.moments).map_err(bad_request)?;
    Ok(Json(json!({ "ok": true })))
}

async fn cancel_job(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    // killing child processes runs a few short commands; keep them off the async runtime
    let jobs = Arc::clone(&s.jobs);
    let found = tokio::task::spawn_blocking(move || jobs.cancel(&id)).await?;
    match found {
        Some(cancelling) => Ok(Json(json!({ "cancelling": cancelling }))),
        None => Err(not_found("job")),
    }
}

type EventStream = Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

async fn job_events(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    let (job, rx) = s.jobs.subscribe(&id).ok_or_else(|| not_found("job"))?;
    let terminal = ["done", "failed", "cancelled"].contains(&job.status.as_str());
    let first = tokio_stream::once(Ok(Event::default().data(json!({"type": "snapshot", "job": job}).to_string())));
    let stream: EventStream = if terminal {
        Box::pin(first)
    } else {
        let live = BroadcastStream::new(rx).filter_map(|r| r.ok()).map(|d| Ok(Event::default().data(d)));
        Box::pin(first.chain(live))
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn job_file(State(s): State<AppState>, Path((id, name)): Path<(String, String)>, req: Request) -> ApiResult<Response> {
    if name.contains('/') || name.contains("..") {
        return Err(bad_request("bad file name"));
    }
    let dir = s.jobs.get(&id).and_then(|j| j.out_dir).ok_or_else(|| not_found("job output"))?;
    let path = dir.join(&name);
    if !path.is_file() {
        return Err(not_found("file"));
    }
    Ok(ServeFile::new(path).oneshot(req).await?.into_response())
}

async fn upload(mut mp: Multipart) -> ApiResult<Json<Value>> {
    let dir = paths::data_dir().join("uploads");
    tokio::fs::create_dir_all(&dir).await?;
    while let Some(mut field) = mp.next_field().await.map_err(|e| bad_request(&e.to_string()))? {
        if field.name() != Some("file") {
            continue;
        }
        let name = field
            .file_name()
            .and_then(|n| std::path::Path::new(n).file_name())
            .map_or_else(|| "upload.mp4".to_string(), |n| n.to_string_lossy().into_owned());
        let path = dir.join(format!("{}-{name}", &uuid::Uuid::new_v4().simple().to_string()[..6]));
        let mut f = tokio::fs::File::create(&path).await?;
        while let Some(chunk) = field.chunk().await.map_err(|e| bad_request(&e.to_string()))? {
            f.write_all(&chunk).await?;
        }
        f.flush().await?;
        return Ok(Json(json!({ "path": path, "name": name })));
    }
    Err(bad_request("multipart field 'file' missing"))
}

#[derive(Deserialize)]
struct PathQuery {
    path: String,
    t: Option<f64>,
}

fn local_video(path: &str) -> ApiResult<std::path::PathBuf> {
    let p = std::path::PathBuf::from(path);
    if p.is_file() { Ok(p) } else { Err(not_found("file")) }
}

async fn probe(Query(q): Query<PathQuery>) -> ApiResult<Json<Value>> {
    let p = local_video(&q.path)?;
    let info = tokio::task::spawn_blocking(move || media::probe(&p)).await??;
    Ok(Json(json!(info)))
}

async fn frame(Query(q): Query<PathQuery>) -> ApiResult<Response> {
    let p = local_video(&q.path)?;
    let t = q.t.unwrap_or(1.0).max(0.0);
    let out = tokio::task::spawn_blocking(move || {
        std::process::Command::new(media::require("ffmpeg")?)
            .args(["-v", "error", "-ss", &format!("{t:.2}"), "-i"])
            .arg(p)
            .args(["-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "4", "-f", "image2", "-c:v", "mjpeg", "-"])
            .output()
            .map_err(anyhow::Error::from)
    })
    .await??;
    if out.stdout.is_empty() {
        return Err(bad_request("could not read frame"));
    }
    Ok(([(header::CONTENT_TYPE, "image/jpeg")], out.stdout).into_response())
}
