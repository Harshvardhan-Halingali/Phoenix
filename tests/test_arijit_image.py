import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

def test_arijit_context():
    client = TestClient(app)
    arijit_img_path = "/Users/hithin7/.gemini/antigravity/brain/0ef0a38e-1e8e-4bc5-b753-49533f010637/.user_uploaded/media_1795115127141_dad3c0d4.jpg"
    
    print("Uploading Arijit Concert image to /api/v1/investigate...")
    with open(arijit_img_path, "rb") as f:
        resp = client.post("/api/v1/investigate", files={"file": ("arijit_concert.jpg", f, "image/jpeg")})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    print(f"Job launched: {job_id}")

    # Wait for execution
    time.sleep(3.0)

    # Fetch dossier
    r_resp = client.get(f"/api/v1/report/{job_id}")
    assert r_resp.status_code == 200
    dossier = r_resp.json()

    print("\n" + "=" * 70)
    print("           ARIJIT CONCERT IMAGE — FORENSIC VERDICT & OSINT")
    print("=" * 70)
    print("Verdict Type:", dossier["verdict"]["verdict_type"])
    print("Authenticity Score:", dossier["verdict"]["authenticity_score"], "%")
    print("Tamper / Synthetic Risk:", dossier["verdict"]["tamper_score"], "%")
    print("Summary:", dossier["verdict"]["summary"])
    print("\nKey Indicators:")
    for k in dossier["verdict"]["key_indicators"]:
        print("  -", k)
    print("\nOSINT Module Findings:")
    print(json.dumps(dossier.get("context_osint"), indent=2))
    print("=" * 70)

if __name__ == "__main__":
    test_arijit_context()
