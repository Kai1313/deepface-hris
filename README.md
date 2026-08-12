# DeepFace Docker Service

Face recognition (verify), facial attributes (analyze) and embeddings (represent) as a
standalone HTTP service. Built with the official `serengil/deepface` image — no deepface
code lives here; the consumer service just makes HTTP calls.

> **`HRIS/`** — a working consumer built on this service: face check-in/check-out kiosk.
> See `HRIS/README.md` for its docs, `HRIS/PLAN.md` for its build plan.

> `.env.example` is committed (no secrets; model config only). Before first run:
> `cp .env.example .env` — the real `.env` is gitignored, so it can safely hold
> `DEEPFACE_AUTH_TOKEN` later.

```
consumer service ──HTTP──► deepface:5005 (docker network) / localhost:5005 (host)
```

## Quick start

```bash
docker compose up -d          # first boot downloads ~100MB model weights (takes a few min)
docker compose ps             # wait until health shows healthy
curl -s localhost:5005/       # expect: <h1>Welcome to DeepFace API...
```

Test with the repo's sample images:

```bash
curl -s -X POST http://localhost:5005/verify \
  -d '{"img1":"https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img1.jpg",
       "img2":"https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img2.jpg"}'
# {"verified": true, "distance": 0.36, ...}  (same person)

curl -s -X POST http://localhost:5005/verify \
  -d '{"img1":"https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img1.jpg",
       "img2":"https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img5.jpg"}'
# {"verified": false, "distance": 0.98, ...}  (different person)

curl -s -X POST http://localhost:5005/analyze \
  -d '{"img":"https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img1.jpg",
       "actions":"age,gender"}'

curl -s -X POST http://localhost:5005/represent \
  -d '{"img":"https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img1.jpg",
       "model_name":"Facenet"}' > embedding.json
```

Images also arrive as multipart file uploads (`-F img1=@a.jpg -F img2=@b.jpg`) or base64.

## Everyday commands

| Do this | Run this |
|---|---|
| Start | `docker compose up -d` |
| Stop (keep data) | `docker compose down` |
| Stop + delete weights | `docker compose down -v` |
| Logs | `docker compose logs -f deepface` |
| Rebuild config after `.env` change | `docker compose up -d` |

## How the pieces fit

| File | Role |
|---|---|
| `docker-compose.yml` | service def: port, healthcheck, restart, weights volume |
| `.env.example` | model choice + optional auth token (copy to `.env`, which is gitignored) |
| `openapi.yaml` | machine-readable contract — consumer devs/client-generators code against this |
| `README.md` | this file: humans + AI handoff |

## Configuration

- **Model**: `DEEPFACE_FACE_RECOGNITION_MODELS=Facenet` in `.env` (preloads at startup so the
  first request is fast). Others: `VGG-Face` (default, accurate, heavy/slow), `Facenet512`,
  `ArcFace`...
- **Auth**: set `DEEPFACE_AUTH_TOKEN` in `.env`, then every request needs
  `Authorization: Bearer <token>`.
- **Port**: host `5005` → container `5000` (the image's internal port).

## Hard constraints (do not "fix" casually)

1. **Single worker.** The image runs gunicorn `--workers=1` — models are memory-heavy and not
   concurrency-safe. Scale by running more containers, not more workers.
2. **First boot downloads weights** into the named volume `deepface-weights`. `down -v` deletes
   them → next boot re-downloads (~100MB for Facenet).
3. **`/register` and `/search` are inactive** without a vector DB
   (`DEEPFACE_CONNECTION_DETAILS`, e.g. postgres/pgvector). Default deployment is
   verify/analyze/represent only — by design, YAGNI.

## AI handoff brief (for any model/CLI)

Context: DeepFace API in Docker; consumer does HTTP only. Commands: table above. Files: the
table above. Gotchas: the constraints section — especially the port mapping (5005→5000) and
that the image has no `curl` (use `python -c` or host-side curl). Don't add a custom
Dockerfile; the official image is the maintained path. Don't add vector DBs, GPU, or
anti-spoofing (torch) unless asked — upgrade path: enable via `.env`/compose flags, not code.

## Upgrades (when the time comes)

- **Vector DB** (many-people lookup): add postgres+pgvector service, set
  `DEEPFACE_DATABASE_TYPE` + `DEEPFACE_CONNECTION_DETAILS`, activate `/register` `/search`.
- **GPU**: run a TF-GPU variant, add `gpus: all` to the service.
- **Anti-spoofing**: needs `torch` — not in the official image; would require a custom build.
