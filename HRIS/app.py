"""HRIS check-in service: kiosk UI + enroll/checkin/checkout API."""
import base64
from flask import Flask, jsonify, render_template, request

import db
import face

app = Flask(__name__)


def get_photo() -> bytes:
    """Photo from multipart 'photo' field, or JSON/base64 'img' data URI."""
    if "photo" in request.files:
        return request.files["photo"].read()
    data = request.get_json(silent=True) or {}
    img = data.get("img")
    if img and img.startswith("data:image"):
        return base64.b64decode(img.split(",", 1)[1])
    raise ValueError("no photo: send multipart 'photo' or JSON base64 'img'")


@app.get("/")
def index():
    return render_template("index.html")


def deepface_err(err: Exception):
    """Map deepface failures to friendly responses."""
    msg = str(err)
    if "Face could not be detected" in msg:
        return jsonify(ok=False, reason="no face detected — look directly at the camera and try again")
    return jsonify(error=msg), 503


@app.post("/enroll")
def enroll():
    data = request.get_json(silent=True) or {}
    name = (request.form.get("name") or data.get("name") or "").strip()
    if not name:
        return jsonify(error="name required"), 400
    try:
        emb = face.represent(get_photo())
    except Exception as err:
        return deepface_err(err)
    _, photos = db.add_embedding(name, emb)
    return jsonify(ok=True, name=name, photos=photos)


def dist_label(dist):
    """Turn a cosine distance into a human explanation (0 = identical, 0.40 = cutoff)."""
    if dist is None:
        return "no usable match"
    if dist < 0.15:
        return "very close match — clearly the same face"
    if dist < 0.30:
        return "good match — same person"
    if dist < face.THRESHOLD:
        return "weak match — recognized, but photo conditions differ a lot"
    return "too different — not the same person"


def punch_flow(punch_type: str):
    try:
        emb = face.represent(get_photo())
    except Exception as err:
        return deepface_err(err)
    eid, name, dist = face.match(emb, db.all_embeddings())
    if eid is None:
        return jsonify(ok=False, reason="unknown face", distance=dist, label=dist_label(dist))
    at = db.punch(eid, punch_type)
    return jsonify(ok=True, name=name, punch_type=punch_type, distance=round(dist, 3),
                   threshold=face.THRESHOLD, label=dist_label(dist), at=at)


@app.post("/checkin")
def checkin():
    return punch_flow("IN")


@app.post("/checkout")
def checkout():
    return punch_flow("OUT")


@app.get("/employees")
def employees():
    return jsonify(db.list_employees())


@app.get("/punches")
def punches():
    return jsonify(db.list_punches())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
