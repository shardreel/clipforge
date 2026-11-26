import { useEffect, useRef, useState } from "react";
import { api, fileUrl, frameUrl, isUrl, jobEvents } from "../api";
import { OptionsForm } from "../OptionsForm";
import type { Clip, Job, Moment, Options, Progress } from "../types";
import { Card, CopyButton, Icon, Logo, NumInput, baseName, fmtBytes, fmtTime } from "../ui";

const STAGES = [
  { id: "fetch", label: "Fetch" },
  { id: "audio", label: "Audio" },
  { id: "transcribe", label: "Transcribe" },
  { id: "analyse", label: "Analyse" },
  { id: "pick", label: "Pick" },
  { id: "render", label: "Render" },
];

function Stepper({ job }: { job: Job }) {
  const now = STAGES.findIndex((s) => s.id === job.stage);
  const finished = job.status === "done";
  return (
    <div className="steps">
      {STAGES.map((s, i) => {
        const done = finished || (now >= 0 && i < now);
        const active = !finished && i === now && (job.status === "running" || job.status === "review");
        return (
          <div key={s.id} className={`step ${done ? "done" : ""} ${active ? "now" : ""}`}>
            <span className={`dot ${done ? "on" : ""} ${active ? "pulse" : ""}`} />
            <span className="trunc">{s.label}</span>
          </div>
        );
      })}
    </div>
  );
}

function DownloadBar({ p }: { p: Progress }) {
  const pct = Math.round(p.percent);
  return (
    <div className="card section">
      <div className="card-body stack" style={{ gap: 10 }}>
        <div className="row between wrap">
          <span className="label">Downloading {p.kind}</span>
          <span className="mono">{p.total ? `${pct}%` : fmtBytes(p.downloaded)}</span>
        </div>
        <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
          <i style={{ width: `${p.total ? p.percent : 0}%` }} />
        </div>
        <div className="row between wrap hint mono">
          <span>{p.total ? `${fmtBytes(p.downloaded)} / ${fmtBytes(p.total)}` : fmtBytes(p.downloaded)}</span>
          <span>
            {p.speed ? `${fmtBytes(p.speed)}/s` : ""}
            {p.eta !== null ? `  ETA ${fmtTime(p.eta)}` : ""}
          </span>
        </div>
      </div>
    </div>
  );
}

/** Link to the moment in the source video (YouTube and most sites accept `t=`). */
function watchUrl(source: string, t: number): string | null {
  if (!isUrl(source)) return null;
  try {
    const u = new URL(source);
    u.searchParams.set("t", `${Math.max(0, Math.floor(t))}s`);
    return u.toString();
  } catch {
    return null;
  }
}

type ReviewItem = Moment & { on: boolean };

function ReviewPanel({ job }: { job: Job }) {
  const moments = job.review ?? [];
  const preselect = job.review_preselect || moments.length;
  const [items, setItems] = useState<ReviewItem[]>(() => moments.map((m, i) => ({ ...m, on: i < preselect })));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const local = !isUrl(job.source);
  const chosen = items.filter((x) => x.on);
  const patch = (i: number, p: Partial<ReviewItem>) => setItems((xs) => xs.map((x, k) => (k === i ? { ...x, ...p } : x)));

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      await api.review(job.id, chosen.map(({ on: _on, ...m }) => m));
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <Card title="Review moments" right={<span className="muted">{chosen.length} selected</span>}>
      <p className="hint" style={{ marginTop: 0 }}>
        Untick weak picks, fix titles, nudge start and end. Only ticked moments are rendered.
      </p>
      <div className="stack" style={{ gap: 12 }}>
        {items.map((m, i) => {
          const link = watchUrl(job.source, m.start);
          return (
            <div key={i} className="row wrap" style={{ gap: 12, alignItems: "center", opacity: m.on ? 1 : 0.5 }}>
              <input type="checkbox" checked={m.on} onChange={(e) => patch(i, { on: e.target.checked })} aria-label="Render this moment" />
              {local ? (
                <img src={frameUrl(job.source, m.peak)} alt="" width={128} style={{ borderRadius: 6 }} loading="lazy" />
              ) : link ? (
                <a className="btn sm" href={link} target="_blank" rel="noreferrer">
                  Watch {fmtTime(m.start)}
                </a>
              ) : null}
              <input type="text" style={{ flex: "1 1 220px" }} value={m.title} onChange={(e) => patch(i, { title: e.target.value })} />
              <div style={{ width: 96 }}>
                <NumInput value={m.start} step={1} min={0} onChange={(v) => patch(i, { start: v })} />
              </div>
              <div style={{ width: 96 }}>
                <NumInput value={m.end} step={1} min={0} onChange={(v) => patch(i, { end: v })} />
              </div>
              <span className="mono faint">{Number.isFinite(m.end - m.start) ? `${(m.end - m.start).toFixed(0)}s` : "-"}</span>
            </div>
          );
        })}
      </div>
      {error && <div className="alert" style={{ marginTop: 12 }}>{error}</div>}
      <div className="row" style={{ justifyContent: "flex-end", marginTop: 16 }}>
        <button className="btn primary" disabled={busy || chosen.length === 0} onClick={submit}>
          {busy ? "Starting..." : `Render ${chosen.length} clip${chosen.length === 1 ? "" : "s"}`}
        </button>
      </div>
    </Card>
  );
}

function ClipCard({ jobId, clip }: { jobId: string; clip: Clip }) {
  if (!clip.file) {
    return (
      <div className="clip">
        <div className="clip-body">
          <div className="clip-title">{clip.title}</div>
          <span className="badge fail">render failed</span>
        </div>
      </div>
    );
  }
  const src = fileUrl(jobId, clip.file);
  return (
    <div className="clip">
      <video src={src} poster={fileUrl(jobId, clip.file.replace(/\.mp4$/, ".jpg"))} controls preload="none" playsInline />
      <div className="clip-body">
        <div className="clip-title">{clip.title}</div>
        {clip.hook && <div className="muted">{clip.hook}</div>}
        <div className="tags">{clip.hashtags.map((h) => <span key={h}>{h}</span>)}</div>
        <div className="clip-meta">
          <span>{fmtTime(clip.start)} - {fmtTime(clip.end)}</span>
          <span>{(clip.end - clip.start).toFixed(0)}s</span>
        </div>
        {clip.reason && <div className="hint">{clip.reason}</div>}
        <div className="clip-actions">
          <CopyButton text={clip.caption} label="Caption" />
          <a className="btn sm" href={src} download={clip.file}>
            <Icon name="download" size={13} /> Save
          </a>
        </div>
      </div>
    </div>
  );
}

export function JobPage({ id, go }: { id: string; go: (hash: string) => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [opts, setOpts] = useState<Options | null>(null);
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setJob(null);
    setError(null);
    return jobEvents(id, (e) => {
      if (e.type === "snapshot") setJob(e.job);
      else if (e.type === "stage") setJob((j) => j && { ...j, stage: e.stage, progress: null });
      else if (e.type === "progress") setJob((j) => j && { ...j, progress: e.progress });
      else if (e.type === "log") setJob((j) => j && { ...j, log: [...j.log, e.line] });
      else if (e.type === "review")
        setJob((j) => j && { ...j, status: "review", review: e.moments, review_preselect: e.preselect, progress: null });
      else if (e.type === "status") {
        setJob((j) => j && { ...j, status: e.status, error: e.error ?? j.error });
        if (e.status === "done" || e.status === "failed" || e.status === "cancelled") api.job(id).then(setJob).catch(() => undefined);
      }
    });
  }, [id]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [job?.log.length]);

  useEffect(() => {
    if (job && !opts) setOpts(job.options);
  }, [job, opts]);

  async function cancel() {
    setBusy(true);
    setError(null);
    try {
      await api.cancel(id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function resume() {
    if (!job) return;
    setBusy(true);
    setError(null);
    try {
      const next = await api.rerender(id, job.options);
      go(`#/job/${next.id}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function rerender() {
    if (!opts) return;
    setBusy(true);
    setError(null);
    try {
      const next = await api.rerender(id, opts);
      setOpts(null);
      go(`#/job/${next.id}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (!job) {
    return <div className="page"><div className="empty"><Logo size={28} />Loading job...</div></div>;
  }

  const running = job.status === "queued" || job.status === "running" || job.status === "review";
  const badge =
    job.status === "done" ? <span className="badge solid">Done</span>
    : job.status === "failed" ? <span className="badge fail">Failed</span>
    : job.status === "cancelled" ? <span className="badge">Cancelled</span>
    : job.status === "review" ? <span className="badge solid">Waiting for review</span>
    : <span className="badge"><span className="dot pulse" /> {job.status === "queued" ? "Queued" : "Running"}</span>;

  return (
    <div className="page">
      <div className="page-head row between wrap">
        <div style={{ minWidth: 0 }}>
          <h1 className="trunc">{baseName(job.source)}</h1>
          <p className="mono faint trunc">{job.source}</p>
        </div>
        <div className="row" style={{ gap: 10 }}>
          {badge}
          {running && (
            <button className="btn sm" disabled={busy} onClick={cancel}>
              {busy ? "Cancelling..." : "Cancel"}
            </button>
          )}
          {(job.status === "failed" || job.status === "cancelled") && (
            <button
              className="btn sm primary"
              disabled={busy}
              onClick={resume}
              title="Run again with the same settings. Downloads, analysis and finished clips are reused."
            >
              Resume
            </button>
          )}
        </div>
      </div>

      <div className="section"><Stepper job={job} /></div>

      {job.stage === "fetch" && job.progress && <DownloadBar p={job.progress} />}

      {job.status === "review" && job.review && <div className="section"><ReviewPanel key={job.review.length} job={job} /></div>}

      {job.error && job.status !== "cancelled" && <div className="alert section"><b>Error: </b>{job.error}</div>}

      {(running || job.status === "failed" || job.status === "cancelled") && (
        <div className="section">
          <h3>Log</h3>
          <div className="log" ref={logRef}>
            {job.log.length === 0 ? <div>Waiting to start...</div> : job.log.map((l, i) => <div key={i}>{l}</div>)}
          </div>
        </div>
      )}

      {job.status === "done" && (
        <>
          <div className="row between wrap section">
            <div>
              <h2>{job.clips.length} clips</h2>
              {job.out_dir && <p className="mono faint">{job.out_dir}</p>}
            </div>
            {job.out_dir && <CopyButton text={job.out_dir} label="Copy folder path" />}
          </div>
          <div className="clips section">
            {job.clips.map((c, i) => <ClipCard key={i} jobId={job.id} clip={c} />)}
          </div>
        </>
      )}

      {!running && opts && (
        <details className="card section">
          <summary className="card-head row between">
            <h2>Re-render with different settings</h2>
            <span className="muted">Reuses cached analysis</span>
          </summary>
          <div className="card-body">
            <OptionsForm options={opts} onChange={setOpts} source={job.source} />
            <div className="row" style={{ justifyContent: "flex-end" }}>
              <button className="btn primary" disabled={busy} onClick={rerender}>
                {busy ? "Starting..." : "Re-render"}
              </button>
            </div>
          </div>
        </details>
      )}
      {error && <div className="alert">{error}</div>}
      {job.status === "done" && (
        <Card title="Log">
          <div className="log">{job.log.map((l, i) => <div key={i}>{l}</div>)}</div>
        </Card>
      )}
    </div>
  );
}
