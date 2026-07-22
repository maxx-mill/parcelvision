"""Orchestration contract: run_extraction walks the documented status sequence
and cleans up after itself. DB and pipeline stages are stubbed; the real
end-to-end path is covered by the compose smoke test."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from worker import jobs, status

JOB_ID = "11111111-2222-3333-4444-555555555555"
BBOX = [-90.32, 38.64, -90.31, 38.65]


@pytest.fixture
def env(monkeypatch, tmp_path):
    statuses: list[tuple[str, dict]] = []

    def record(engine, job_id, st, error=None, building_count=None):
        statuses.append((st, {"error": error, "count": building_count}))

    row = SimpleNamespace(bbox=BBOX, backend="fake")
    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.one_or_none.return_value = row
    engine = MagicMock()
    engine.connect.return_value = conn

    monkeypatch.setattr(jobs, "get_engine", lambda: engine)
    monkeypatch.setattr(jobs.status, "set_status", record)
    monkeypatch.setattr(jobs, "load_buildings", lambda eng, jid, gdf: len(gdf))
    monkeypatch.setattr(jobs.config, "imagery_dir", lambda: str(tmp_path))
    return SimpleNamespace(statuses=statuses, row=row, monkeypatch=monkeypatch)


def test_fake_backend_lifecycle_skips_imagery(env):
    result = jobs.run_extraction(JOB_ID)
    seq = [s for s, _ in env.statuses]
    assert seq == [status.RUNNING_INFERENCE, status.VECTORIZING, status.WRITING_DB, status.DONE]
    assert result["building_count"] > 0
    assert env.statuses[-1][1]["count"] == result["building_count"]


def test_imagery_backend_reports_fetch_stage_and_failure(env):
    env.row.backend = "local_cpu"

    def boom(bbox, workdir, year=None):
        raise RuntimeError("no NAIP imagery found")

    env.monkeypatch.setattr(jobs, "fetch_imagery", boom)
    with pytest.raises(RuntimeError):
        jobs.run_extraction(JOB_ID)
    seq = [s for s, _ in env.statuses]
    assert seq == [status.FETCHING_IMAGERY, status.FAILED]
    assert "no NAIP imagery" in env.statuses[-1][1]["error"]
