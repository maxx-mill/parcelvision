#!/usr/bin/env bash
# Full-stack smoke test on the fake backend: boots db/redis/api/worker(slim),
# submits a job, walks it to `done`, and asserts footprints come back out.
# Used by CI and `make smoke`. Roughly 2 minutes cold.
set -euo pipefail
cd "$(dirname "$0")/.."

# Own project name + no shared host ports: safe to run beside a dev stack.
COMPOSE="docker compose -p parcelvision-smoke -f docker-compose.yml -f docker-compose.ci.yml"
API="http://localhost:8001"
[ -f .env ] || cp .env.example .env

$COMPOSE up -d --build --force-recreate db redis api worker
trap '$COMPOSE down -v >/dev/null 2>&1' EXIT

echo "waiting for api…"
for _ in $(seq 1 45); do
  curl -sf "$API/api/health" | grep -q '"status":"ok"' && break
  sleep 2
done
curl -sf "$API/api/health" | grep -q '"status":"ok"'

# sed/grep JSON parsing keeps this runnable on runners and Git Bash alike.
JID=$(curl -sf -X POST "$API/api/jobs" \
  -H 'Content-Type: application/json' \
  -d '{"bbox": [-90.32, 38.64, -90.31, 38.65]}' | sed -n 's/.*"id":"\([0-9a-f-]*\)".*/\1/p')
[ -n "$JID" ] || { echo "job submit failed"; exit 1; }
echo "job $JID submitted"

ST=queued
for _ in $(seq 1 60); do
  ST=$(curl -sf "$API/api/jobs/$JID" | sed -n 's/.*"status":"\([a-z_]*\)".*/\1/p')
  case "$ST" in
    done) break ;;
    failed|canceled) echo "job ended $ST"; curl -s "$API/api/jobs/$JID"; exit 1 ;;
  esac
  sleep 2
done
[ "$ST" = done ] || { echo "timed out in status $ST"; exit 1; }

COUNT=$(curl -sf "$API/api/jobs/$JID/buildings" | grep -o '"geometry"' | wc -l | tr -d ' ')
echo "smoke OK: job done with $COUNT synthetic footprints end to end"
[ "$COUNT" -gt 10 ]
