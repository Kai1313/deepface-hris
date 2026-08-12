"""Self-check for HRIS. No camera needed.

Tier 1 (always runs): db + matcher with synthetic embeddings.
Tier 2 (if deepface container is up): live /represent integration.

Run: python test_check.py  (from the HRIS folder)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import db
import face

rng = np.random.default_rng(42)

# fresh database
if os.path.exists(db.DB):
    os.remove(db.DB)

# enroll Alice (3 photos) and Bob (2 photos) with synthetic embeddings
for _ in range(3):
    db.add_embedding("Alice", rng.standard_normal(128).tolist())
for _ in range(2):
    db.add_embedding("Bob", rng.standard_normal(128).tolist())

employees = db.all_embeddings()
assert len(employees) == 2

# 1) same-person probe (tiny perturbation of Alice's first embedding) -> Alice
probe = db.all_embeddings()[0][2][0] + 0.01 * rng.standard_normal(128)
eid, name, dist = face.match(probe.tolist(), employees)
assert name == "Alice", f"expected Alice, got {name}"
print(f"[1] same-person probe -> {name} (dist {dist:.3f}) OK")

# 2) unknown face -> rejected (no punch)
eid, name, dist = face.match(rng.standard_normal(128).tolist(), employees)
assert name is None, f"expected unknown, got {name}"
print(f"[2] random probe -> rejected (dist {dist if dist is None else f'{dist:.3f}'}) OK")

# 3) punch flow + log
db.punch(1, "IN")
db.punch(1, "OUT")
punches = db.list_punches()
assert len(punches) == 2 and punches[0]["punch_type"] == "OUT" and punches[1]["name"] == "Alice"
print("[3] punch IN/OUT recorded OK")

# 4) live deepface integration (skipped when container not running)
try:
    import requests
    requests.get("http://localhost:5005/", timeout=3).raise_for_status()
except Exception:
    print("[4] deepface not reachable on :5005 — integration skipped")
else:
    img = requests.get(
        "https://raw.githubusercontent.com/serengil/deepface/master/tests/unit/dataset/img1.jpg",
        timeout=30,
    ).content
    emb = face.represent(img)
    assert len(emb) == 128, f"expected 128-dim Facenet embedding, got {len(emb)}"
    eid, name, dist = face.match(emb, employees)
    # img1 is a stock photo, not Alice/Bob -> must be rejected
    assert name is None, f"stock photo should be unknown, got {name}"
    print(f"[4] live deepface -> 128-dim embedding, unknown face rejected (dist {dist if dist is None else f'{dist:.3f}'}) OK")

print("ALL CHECKS PASSED")
