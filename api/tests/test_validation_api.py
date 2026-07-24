"""Contract tests for the parcel-validation endpoints (Chapter 3). DB session
and queue are stubbed; the real spatial joins are exercised by the live run
documented in the README."""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from app.db import get_session
from app.main import app
from app.models import Job
from app.routers import validation as validation_module
from fastapi.testclient import TestClient

STL_BBOX = [-90.31, 38.61, -90.30, 38.62]


@pytest.fixture(autouse=True)
def _no_db_init(monkeypatch):
    monkeypatch.setattr("app.main.init_db", lambda: None)


@pytest.fixture
def client(monkeypatch):
    session = MagicMock()
    app.dependency_overrides[get_session] = lambda: session
    enqueued: list[str] = []
    monkeypatch.setattr(validation_module, "enqueue_validation", enqueued.append)
    with TestClient(app) as c:
        c.app_session = session  # type: ignore[attr-defined]
        c.enqueued = enqueued  # type: ignore[attr-defined]
        yield c
    app.dependency_overrides.clear()


def _job(**kw):
    base = dict(
        id=uuid.uuid4(),
        status="done",
        bbox=STL_BBOX,
        backend="local_cpu",
        is_seed=False,
        validation_status=None,
        validation_error=None,
        created_at=datetime.now(UTC),
    )
    base.update(kw)
    return Job(**base)


def test_validate_enqueues_for_done_job(client):
    job = _job()
    client.app_session.get.return_value = job
    resp = client.post(f"/api/jobs/{job.id}/validate")
    assert resp.status_code == 202
    assert resp.json()["validation_status"] == "loading_parcels"
    assert client.enqueued == [str(job.id)]


def test_validate_rejects_unfinished_job(client):
    job = _job(status="running_inference")
    client.app_session.get.return_value = job
    resp = client.post(f"/api/jobs/{job.id}/validate")
    assert resp.status_code == 409
    assert client.enqueued == []


def test_validate_rejects_when_already_running(client):
    job = _job(validation_status="validating")
    client.app_session.get.return_value = job
    resp = client.post(f"/api/jobs/{job.id}/validate")
    assert resp.status_code == 409


def test_get_validation_409_before_started(client):
    job = _job(validation_status=None)
    client.app_session.get.return_value = job
    resp = client.get(f"/api/jobs/{job.id}/validation")
    assert resp.status_code == 409
