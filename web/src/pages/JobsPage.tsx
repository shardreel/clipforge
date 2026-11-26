import { useEffect, useState } from "react";
import { api } from "../api";
import type { Job } from "../types";
import { Logo, baseName, fmtDate } from "../ui";

function Status({ job }: { job: Job }) {
  if (job.status === "done") return <span className="badge solid">Done</span>;
  if (job.status === "failed") return <span className="badge fail">Failed</span>;
  if (job.status === "cancelled") return <span className="badge">Cancelled</span>;
  if (job.status === "review") return <span className="badge solid">Review</span>;
  return (
    <span className="badge">
      <span className="dot pulse" /> {job.status === "queued" ? "Queued" : job.stage}
    </span>
  );
}

export function JobsPage({ go }: { go: (hash: string) => void }) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .jobs()
        .then((j) => alive && (setJobs(j), setError(null)))
        .catch((e: Error) => alive && setError(e.message));
    load();
    const id = setInterval(load, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="page">
      <div className="page-head row between wrap">
        <div>
          <h1>Jobs</h1>
          <p>Every run, newest first.</p>
        </div>
        <a className="btn primary" href="#/">New job</a>
      </div>
      {error && <div className="alert section">{error}</div>}
      <div className="card">
        {jobs === null ? (
          <div className="empty"><Logo size={28} />Loading...</div>
        ) : jobs.length === 0 ? (
          <div className="empty"><Logo size={28} />No jobs yet. Start one from New job.</div>
        ) : (
          <table className="table">
            <thead>
              <tr><th>Video</th><th>Status</th><th>Clips</th><th>Started</th></tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} onClick={() => go(`#/job/${j.id}`)}>
                  <td>
                    <div className="trunc" style={{ maxWidth: 420 }}>{baseName(j.source)}</div>
                    <div className="hint mono">{j.id}</div>
                  </td>
                  <td><Status job={j} /></td>
                  <td className="mono">{j.status === "done" ? j.clips.length : "-"}</td>
                  <td className="muted">{fmtDate(j.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
