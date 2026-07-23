"""Endpoint-contract tests that run without Postgres/Redis: the DB session and
queue are stubbed out. Full-stack behavior is covered by the compose smoke test
(see README / Makefile)."""

from unittest.mock import MagicMock

import pytest
from app.db import get_session
from app.main import app
from app.routers import jobs as jobs_module
from fastapi.testclient import TestClient

STL_BBOX = [-90.31, 38.61, -90.30, 38.62]


@pytest.fixture
def client(monkeypatch, _no_db_init):
    session = MagicMock()
    app.dependency_overrides[get_session] = lambda: session
    enqueued: list[str] = []
    monkeypatch.setattr(jobs_module, "enqueue_extraction", enqueued.append)
    # lifespan would try to connect to Postgres; TestClient without context
    # manager skips startup only if we don't use `with` — be explicit instead.
    with TestClient(app, raise_server_exceptions=True) as c:
        c.enqueued = enqueued  # type: ignore[attr-defined]
        c.app_session = session  # type: ignore[attr-defined]
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def _no_db_init(monkeypatch):
    monkeypatch.setattr("app.main.init_db", lambda: None)


def test_create_job_returns_id_and_enqueues(client):
    resp = client.post("/api/jobs", json={"bbox": STL_BBOX})
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    assert body["bbox"] == STL_BBOX
    assert client.enqueued == [body["id"]]


def test_create_job_rejects_malformed_bbox(client):
    resp = client.post("/api/jobs", json={"bbox": [1, 2, 3]})
    assert resp.status_code == 422
    assert client.enqueued == []


def test_create_job_rejects_oversized_bbox(client):
    resp = client.post("/api/jobs", json={"bbox": [-90.6, 38.4, -90.1, 38.9]})
    assert resp.status_code == 413
    assert "km²" in resp.json()["detail"]
    assert client.enqueued == []


def test_export_rejects_unknown_format(client):
    resp = client.get("/api/jobs/00000000-0000-0000-0000-000000000000/export?format=dwg")
    assert resp.status_code == 400


def _job(status: str):
    import uuid
    from datetime import UTC, datetime

    from app.models import Job

    return Job(
        id=uuid.uuid4(), status=status, bbox=STL_BBOX, backend="local_cpu",
        is_seed=False, created_at=datetime.now(UTC),
    )


def test_cancel_running_job(client, monkeypatch):
    canceled: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        jobs_module, "cancel_extraction", lambda jid, running: canceled.append((jid, running))
    )
    job = _job("running_inference")
    client.app_session.get.return_value = job
    resp = client.delete(f"/api/jobs/{job.id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "canceled"
    assert canceled == [(str(job.id), True)]


def test_cancel_terminal_job_conflicts(client, monkeypatch):
    monkeypatch.setattr(jobs_module, "cancel_extraction", lambda *a, **k: 1 / 0)
    job = _job("done")
    client.app_session.get.return_value = job
    resp = client.delete(f"/api/jobs/{job.id}")
    assert resp.status_code == 409
