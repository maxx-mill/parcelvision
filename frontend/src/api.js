async function json(resp) {
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      detail = (await resp.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return resp.json();
}

export const createJob = async (bbox) =>
  json(
    await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bbox }),
    })
  );

export const getJob = async (id) => json(await fetch(`/api/jobs/${id}`));
export const cancelJob = async (id) => json(await fetch(`/api/jobs/${id}`, { method: "DELETE" }));
export const listJobs = async () => json(await fetch("/api/jobs?limit=50"));
export const getBuildings = async (id) => json(await fetch(`/api/jobs/${id}/buildings`));
export const exportUrl = (id, format) => `/api/jobs/${id}/export?format=${format}`;

export const startValidation = async (id) =>
  json(await fetch(`/api/jobs/${id}/validate`, { method: "POST" }));
export const getValidation = async (id) => json(await fetch(`/api/jobs/${id}/validation`));
