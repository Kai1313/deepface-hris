"""DeepFace HTTP client + cosine matcher. HRIS never imports deepface itself."""
import os
import requests
import numpy as np

DEEPFACE_URL = "http://localhost:5005"
THRESHOLD = 0.40  # deepface Facenet cosine-distance threshold
# opencv (default) is weak: fails on angles/light. mtcnn handles pose much better.
DETECTOR_BACKEND = os.getenv("DEEPFACE_DETECTOR", "mtcnn")


def represent(photo_bytes: bytes, filename: str = "photo.jpg") -> list:
    """POST a photo to deepface /represent, return the 128-dim Facenet embedding."""
    resp = requests.post(
        f"{DEEPFACE_URL}/represent",
        files={"img": (filename, photo_bytes, "image/jpeg")},
        data={"model_name": "Facenet", "detector_backend": DETECTOR_BACKEND},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"deepface /represent failed ({resp.status_code}): {resp.text[:200]}")
    return resp.json()["results"][0]["embedding"]


def cosine_distance(a: list, b: list) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match(embedding: list, employees: list) -> tuple:
    """Best employee for an embedding. employees = db.all_embeddings().

    Returns (id, name, distance) if below THRESHOLD, else (None, None, None).
    """
    best = None
    for eid, name, embeddings in employees:
        for emb in embeddings:
            d = cosine_distance(embedding, emb)
            if best is None or d < best[2]:
                best = (eid, name, d)
    if best is not None and best[2] < THRESHOLD:
        return best
    # above threshold: no identity, but keep the closest distance so the UI can say how far off
    return (None, None, best[2] if best is not None else None)
