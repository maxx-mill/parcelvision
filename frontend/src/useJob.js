import { useCallback, useEffect, useRef, useState } from "react";
import { cancelJob, createJob, getBuildings, getJob } from "./api.js";

const POLL_MS = 2000;
const TERMINAL = new Set(["done", "failed", "canceled"]);

/** Submit a job and poll its status until it settles; loads the result layer
 *  when it completes. Also handles adopting an existing job (demo AOI). */
export function useJob() {
  const [job, setJob] = useState(null);
  const [buildings, setBuildings] = useState(null);
  const [error, setError] = useState(null);
  const timer = useRef(null);

  const stopPolling = () => {
    if (timer.current) {
      clearInterval(timer.current);
      timer.current = null;
    }
  };

  const loadResults = useCallback(async (jobId) => {
    setBuildings(await getBuildings(jobId));
  }, []);

  const watch = useCallback(
    (initial) => {
      stopPolling();
      setJob(initial);
      setBuildings(null);
      setError(null);
      if (TERMINAL.has(initial.status)) {
        if (initial.status === "done") loadResults(initial.id).catch((e) => setError(e.message));
        return;
      }
      timer.current = setInterval(async () => {
        try {
          const j = await getJob(initial.id);
          setJob(j);
          if (TERMINAL.has(j.status)) {
            stopPolling();
            if (j.status === "done") await loadResults(j.id);
          }
        } catch (e) {
          stopPolling();
          setError(e.message);
        }
      }, POLL_MS);
    },
    [loadResults]
  );

  const submit = useCallback(
    async (bbox) => {
      setError(null);
      try {
        watch(await createJob(bbox));
      } catch (e) {
        setError(e.message);
      }
    },
    [watch]
  );

  const cancel = useCallback(async () => {
    if (!job) return;
    try {
      const j = await cancelJob(job.id);
      stopPolling();
      setJob(j);
    } catch (e) {
      setError(e.message);
    }
  }, [job]);

  const reset = useCallback(() => {
    stopPolling();
    setJob(null);
    setBuildings(null);
    setError(null);
  }, []);

  useEffect(() => stopPolling, []);

  return { job, buildings, error, submit, watch, cancel, reset };
}
