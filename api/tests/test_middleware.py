"""Request-id middleware + job rate-limit contract (no Postgres/Redis needed)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from app import ratelimit
from app.db import get_session
from app.main import app
from app.routers import jobs as jobs_module
from fastapi.testclient import TestClient

STL_BBOX = [-90.31, 38.61, -90.30, 38.62]


@pytest.fixture(autouse=True)
def _no_db_init(monkeypatch):
    monkeypatch.setattr("app.main.init_db", lambda: None)


@pytest.fixture
def client(monkeypatch):
    app.dependency_overrides[get_session] = lambda: MagicMock()
    monkeypatch.setattr(jobs_module, "enqueue_extraction", lambda *_: None)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_response_carries_request_id(client):
    r = client.get("/api/health")
    assert r.headers.get("x-request-id")


def test_incoming_request_id_is_echoed(client):
    r = client.get("/api/health", headers={"X-Request-ID": "trace-42"})
    assert r.headers["x-request-id"] == "trace-42"


def test_job_creation_is_rate_limited(client, monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.n = 0

        def incr(self, _key):
            self.n += 1
            return self.n

        def expire(self, *_):
            pass

    monkeypatch.setattr(ratelimit, "get_redis", lambda fake=FakeRedis(): fake)
    monkeypatch.setattr(
        ratelimit, "get_settings", lambda: SimpleNamespace(job_rate_limit_per_min=2)
    )

    codes = [client.post("/api/jobs", json={"bbox": STL_BBOX}).status_code for _ in range(4)]
    assert codes[:2] == [201, 201]  # first two allowed
    assert codes[2] == 429 and codes[3] == 429  # then throttled
