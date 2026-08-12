# HRIS Check-in Service

Face-based employee check-in/check-out. Uses the DeepFace Docker service (repo root)
as a pure-HTTP face-embedding helper. See `PLAN.md` for the design and build steps.
**Status: v1 complete and verified live.** `hris.db` and `hris.log` are gitignored
(biometric data — do not commit).

## Run

```bash
# 1. DeepFace service (repo root) — must be up and healthy first
cd .. && cp .env.example .env && docker compose up -d
curl localhost:5005/          # expect <h1>Welcome to DeepFace API...

# 2. HRIS app
cd HRIS
pip install -r requirements.txt
python app.py                 # kiosk at http://localhost:8000
```

## How it works

1. **Enroll** (once per employee, 2–4 photos): name → `Enroll this photo` ×3–4,
   moving your head between clicks. Each click appends an embedding to the employee
   row in `hris.db`. Use the **exact same name** every click (case/space matters).
2. **Check-in / Check-out**: one click — the kiosk snapshots the live camera
   automatically (no pre-capture needed), deepface converts the face to an embedding,
   HRIS compares it against **all registered photos**, and if the closest match is
   below the threshold it writes an `IN`/`OUT` punch — otherwise `unknown face`.
3. **Verify**: `GET /employees` (roster + photo count), `GET /punches` (log).

## Reading the match distance

| Distance | Meaning | Verdict |
|---|---|---|
| 0.00 – 0.15 | nearly identical | ✅ strong match |
| 0.15 – 0.30 | same person, normal variation | ✅ good match |
| 0.30 – 0.40 | same person, different light/angle | ⚠️ weak (accepted) |
| 0.40 – 0.80 | different person or extreme angle | ❌ rejected |
| 0.80+ | strangers | ❌ rejected |

`0.40` is the cutoff (`THRESHOLD` in `face.py`). The kiosk shows
`match X of 0.40 — <plain-language label>` on every check.

## API

| Route | Body | Result |
|---|---|---|
| `POST /enroll` | multipart `photo` + `name` | `{ok, name, photos}` |
| `POST /checkin` | multipart `photo` | `{ok, name, punch_type, distance, threshold, label, at}` or `{ok:false, reason:"unknown face", distance, label}` |
| `POST /checkout` | multipart `photo` | same as checkin |
| `GET /employees` | – | roster + photo count |
| `GET /punches` | – | attendance log (newest first) |

Curl example:

```bash
curl -X POST http://localhost:8000/checkin -F photo=@alice.jpg
```

## Config (`face.py`)

- `DEEPFACE_URL` — deepface endpoint (default `http://localhost:5005`)
- `DEEPFACE_DETECTOR` — face detector (default `mtcnn`; opencv fails on angles/light,
  `retinaface` = most accurate, slower)
- `THRESHOLD` — match cutoff (default `0.40`, deepface Facenet cosine standard)

## Kiosk notes

- Camera works only on **http://localhost:8000** (or HTTPS) — browsers block it on LAN IPs.
  No camera → use **Upload instead**.
- Buttons auto-snapshot the live camera; **Capture/Upload** is only a fallback/override.

## Test

```bash
python test_check.py    # matcher + db (synthetic) + live deepface integration when up
```

## Constraints / upgrade paths (see PLAN.md)

- 1 gunicorn worker in deepface → scale with more containers, not workers
- Embeddings in SQLite → swap to pgvector when they exceed tens of thousands
- Anti-spoofing off → custom build with torch when spoofing becomes a real risk
