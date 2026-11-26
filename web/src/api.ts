import type { Health, Job, JobEvent, Moment, Options, SettingsUpdate, SettingsView, StorageReport, VideoInfo } from "./types";

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
  return body as T;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req<Health>("/api/health"),
  storage: () => req<StorageReport>("/api/storage"),
  clearStorage: (kind: string) => req<{ freed: number }>(`/api/storage/${encodeURIComponent(kind)}`, { method: "DELETE" }),
  defaults: () => req<Options>("/api/defaults"),
  settings: () => req<SettingsView>("/api/settings"),
  saveSettings: (s: SettingsUpdate) => req<SettingsView>("/api/settings", json("PUT", s)),
  jobs: () => req<Job[]>("/api/jobs"),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  createJob: (source: string, options: Options) => req<Job>("/api/jobs", json("POST", { source, options })),
  review: (id: string, moments: Moment[]) => req<{ ok: boolean }>(`/api/jobs/${id}/review`, json("POST", { moments })),
  cancel: (id: string) => req<{ cancelling: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  rerender: (id: string, options: Options) => req<Job>(`/api/jobs/${id}/rerender`, json("POST", { options })),
  probe: (path: string) => req<VideoInfo>(`/api/probe?path=${encodeURIComponent(path)}`),
};

export const isUrl = (s: string) => /^https?:\/\//i.test(s.trim());

export const frameUrl = (path: string, t = 1) => `/api/frame?path=${encodeURIComponent(path)}&t=${t.toFixed(2)}`;

export const fileUrl = (id: string, name: string) => `/api/jobs/${id}/files/${encodeURIComponent(name)}`;

export function upload(file: File, onProgress: (pct: number) => void): Promise<{ path: string; name: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress((e.loaded / e.total) * 100);
    xhr.onerror = () => reject(new Error("upload failed"));
    xhr.onload = () => {
      let body: { path?: string; name?: string; error?: string } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300 && body.path) resolve({ path: body.path, name: body.name ?? file.name });
      else reject(new Error(body.error ?? `HTTP ${xhr.status}`));
    };
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export function jobEvents(id: string, onEvent: (e: JobEvent) => void): () => void {
  const es = new EventSource(`/api/jobs/${id}/events`);
  es.onmessage = (m) => {
    try {
      onEvent(JSON.parse(m.data) as JobEvent);
    } catch {
      /* ignore malformed event */
    }
  };
  return () => es.close();
}
