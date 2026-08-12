# HRIS Check-in Service — Build Plan

> **STATUS: v1 COMPLETE & VERIFIED LIVE.** All steps below are done; this file remains
> the single source of truth for any AI CLI continuing this project.

> This file is the single source of truth for building the HRIS check-in service.
> Any AI CLI (DeepSeek, Gemini, Claude, ...) should read THIS file first and follow
> the steps in order. Do not rely on chat history.

## Goal

A simple HRIS check-in/check-out system that uses the DeepFace Docker service (repo root)
as a face-identification helper via pure HTTP (Option B). Employees enroll their face once;
daily check-in/out is a photo match that writes an attendance record.

## Architecture

```
browser/kiosk ──► HRIS app (Flask, port 8000) ──► deepface (port 5005)
                        │
                        ▼
                 SQLite (stdlib, file-based)
                 employees + embeddings + punches
```

- HRIS **never imports deepface** — it calls `POST localhost:5005/represent` over HTTP
- Embeddings (128 floats per photo, Facenet) stored in SQLite as JSON — no vector DB at this scale
- Identity logic (cosine match + threshold) lives in HRIS, not deepface

## Dependencies

`HRIS/requirements.txt`: `flask`, `requests`, `numpy` (sqlite3 is Python stdlib).

## Steps (build in order)

### Step 1 — Scaffold
- Create `HRIS/` folder, `requirements.txt`, `templates/` folder.

### Step 2 — `HRIS/db.py`
Schema:
```
employees(id INTEGER PK, name TEXT UNIQUE, embeddings TEXT)  -- JSON list of embedding lists
punches(id INTEGER PK, employee_id INTEGER, punch_type TEXT CHECK IN/OUT, at TEXT ISO)
```
Helpers: `add_embedding`, `all_embeddings`, `punch(employee_id, type)`,
`list_employees`, `list_punches`. sqlite3 stdlib, ~40 lines.

### Step 3 — `HRIS/face.py`
- `represent(photo_bytes)` → POST to `http://localhost:5005/represent` (multipart, `model_name=Facenet`) → embedding list
- `match(embedding)` → cosine distance vs ALL stored embeddings → best employee + distance
- Rule: `distance < 0.40` → identified; else → unknown (no punch)

### Step 4 — `HRIS/app.py` (Flask routes)
| Route | What it does |
|---|---|
| `GET /` | kiosk page (HTML) |
| `POST /enroll` | name + photo → represent → save embedding |
| `POST /checkin` | photo → match → write punch IN |
| `POST /checkout` | photo → match → write punch OUT |
| `GET /employees` | roster + enrollment count |
| `GET /punches` | attendance log |

### Step 5 — `HRIS/templates/index.html`
One-page kiosk: photo upload, name box (enrollment), Check-in / Check-out buttons,
live punches log.

### Step 6 — `HRIS/test_check.py`
Runnable self-check (no camera needed): enroll 2 fake employees using deepface's public
test images (repo: `tests/unit/dataset/img*.jpg`), assert matcher picks the right person and
rejects an unknown face. Run: `python test_check.py`.

### Step 7 — `HRIS/README.md`
Run order: deepface up (repo root) → HRIS up → enroll → check in/out. Curl examples.

## Locked decisions

1. **SQLite** — file-based, zero ops. Upgrade path: swap schema to Postgres later.
2. **Threshold 0.40** — deepface Facenet cosine standard. Tune after real-photo testing.
3. **HRIS on port 8000**, deepface stays on 5005 (repo root compose).
4. **Official deepface image, no clone** — HRIS talks HTTP only.
5. **Detector: mtcnn** (`DEEPFACE_DETECTOR` env in `face.py`) — opencv default fails on
   angles/light; mtcnn is installed in the image and handles pose. retinaface = upgrade.
6. **Camera auto-snapshot** — Check-in/Check-out/Enroll grab a fresh frame from the
   live camera; Capture/Upload is only a fallback. Camera requires localhost/HTTPS.

## Open questions (answered during build)

- Camera capture on kiosk page → **YES**, getUserMedia + auto-snapshot on button click.
- Enrollments: 1 photo or 2–4? → **2–4** photos per employee, appended per click.
  Use the exact same name spelling on every enroll click.

## Done = acceptance criteria (ALL MET, verified live 2026-08-12)

- [x] `docker compose up -d` at repo root → deepface healthy, `/represent` returns 128-dim embeddings
- [x] Enroll Alice (2 photos) + Bob with real test photos (mtcnn detector)
- [x] Check-in with enrolled face → punch row IN; check-out → OUT
- [x] Check-in with unenrolled face → rejected `unknown face`, no punch
- [x] `python test_check.py` → ALL CHECKS PASSED (incl. live deepface tier)
- [x] Kiosk: colored status (green/red), plain-language match labels, Recent punches table

## Session corrections (v1, keep these in mind)

1. Facenet embeddings are **128-dim**, not 512 (that's Facenet512).
2. Repo test images moved to `tests/unit/dataset/` — README examples were stale.
3. `round(None)` crash on unknown face → rejected responses now carry `distance: null`.
4. opencv detector → mtcnn after "Face could not be detected" on tilted faces.
5. CSS specificity bug: `#status` overrode `.ok/.err` colors → use `#status.ok`.
6. Never reuse a cached snapshot for consecutive actions (caused `dist 0` and identical
   enroll photos) — every action takes a fresh camera frame.

## Upgrade paths (NOT now — YAGNI)

- Vector DB / pgvector when embeddings exceed tens of thousands
- GPU deepface when throughput demands
- Anti-spoofing (torch, custom build) when spoofing becomes a business risk
- Camera capture = small `getUserMedia` addition to index.html
