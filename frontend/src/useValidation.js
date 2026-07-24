import { useCallback, useEffect, useRef, useState } from "react";
import { getJob, getValidation, startValidation } from "./api.js";

const POLL_MS = 2000;
const TERMINAL = new Set(["done", "failed"]);

/** Trigger parcel validation for a job and poll its (separate) validation
 *  lifecycle; load the parcel layer + summary when it completes. `job` may
 *  carry an existing validation_status (e.g. the demo job), in which case
 *  finished results are loaded immediately without re-running. */
export function useValidation(job) {
  const jobId = job?.status === "done" ? job.id : null;
  const priorStatus = job?.validation_status ?? null;
  const [status, setStatus] = useState(null); // null | loading_parcels | validating | done | failed
  const [result, setResult] = useState(null); // { summary, parcels }
  const [error, setError] = useState(null);
  const timer = useRef(null);

  const stop = () => {
    if (timer.current) {
      clearInterval(timer.current);
      timer.current = null;
    }
  };

  const load = useCallback(async (id) => {
    const v = await getValidation(id);
    setResult({ summary: v.summary, parcels: v.parcels });
    setStatus(v.status);
  }, []);

  // On job change: reset, and if it was already validated, show it straight away.
  useEffect(() => {
    stop();
    setResult(null);
    setError(null);
    if (jobId && priorStatus === "done") {
      setStatus("done");
      load(jobId).catch((e) => setError(e.message));
    } else {
      setStatus(null);
    }
  }, [jobId, priorStatus, load]);

  useEffect(() => stop, []);

  const validate = useCallback(async () => {
    if (!jobId) return;
    setError(null);
    try {
      const j = await startValidation(jobId);
      setStatus(j.validation_status ?? "loading_parcels");
      stop();
      timer.current = setInterval(async () => {
        try {
          const jj = await getJob(jobId);
          setStatus(jj.validation_status);
          if (TERMINAL.has(jj.validation_status)) {
            stop();
            if (jj.validation_status === "done") await load(jobId);
            else setError(jj.validation_error || "validation failed");
          }
        } catch (e) {
          stop();
          setError(e.message);
        }
      }, POLL_MS);
    } catch (e) {
      setError(e.message);
    }
  }, [jobId, load]);

  return { status, result, error, validate };
}
