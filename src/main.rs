mod api;
mod cancel;
mod fastdl;
mod ingest;
mod jobs;
mod llm;
mod media;
mod paths;
mod picker;
mod pipeline;
mod platform;
mod preflight;
mod render;
mod settings;
mod signals;
mod storage;
mod subtitles;
mod transcribe;

use std::net::SocketAddr;
use std::path::PathBuf;
use tower_http::services::{ServeDir, ServeFile};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    // whisper.cpp/ggml print model-loading details to stderr on every load; no logging backend
    // is compiled into whisper-rs, so this routes them nowhere (errors still surface as Results)
    whisper_rs::install_logging_hooks();
    let addr: SocketAddr = std::env::var("CLIPFORGE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8787".into())
        .parse()?;
    let state = api::AppState { jobs: jobs::Jobs::load() };
    let mut app = api::router(state);
    let dist = std::env::var_os("CLIPFORGE_WEB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("web/dist"));
    let index = dist.join("index.html");
    if index.is_file() {
        app = app.fallback_service(ServeDir::new(&dist).fallback(ServeFile::new(index)));
    } else {
        tracing::warn!("web UI not built ({}). Run: cd web && bun run build", dist.display());
    }
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("clipforge listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
