import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app
from app.modules.efficientnet_detector import efficientnet_detector
from app.modules.authenticity_combiner import compute_combined_authenticity_score

def run_tests():
    print("=================================================================")
    print("STEP 1: Verify EfficientNet-B0 Model Loading & Architecture")
    print("=================================================================")
    detector = efficientnet_detector
    detector._lazy_init()
    assert detector is not None, "Detector is None!"
    assert detector.is_loaded(), "Model is not loaded!"
    print(f"✓ Model loaded: {detector.architecture} on {detector.device}")
    print(f"✓ Checkpoint: {detector.checkpoint_path}")
    print(f"✓ Class to index: {detector.class_to_idx}")
    print(f"✓ Test Metrics: {detector.metrics}")

    print("\n=================================================================")
    print("STEP 2: Inference on Sample Genuine Image")
    print("=================================================================")
    test_img = "data/uploads/fd05e475-3963-44f2-bbcb-7dd3aade08e3_sample_clean.jpg"
    if not os.path.exists(test_img):
        import glob
        imgs = glob.glob("data/uploads/*.jpg") + glob.glob("data/uploads/*.png")
        test_img = imgs[0] if imgs else None

    print(f"Testing on image: {test_img}")
    res = detector.predict(test_img)
    print(f"Classification: {res['classification']}")
    print(f"Fake %: {res['fake_percentage']:.2f}% | Real %: {res['real_percentage']:.2f}%")
    print(f"Sum check: {res['fake_percentage'] + res['real_percentage']:.2f}%")
    assert abs((res['fake_percentage'] + res['real_percentage']) - 100.0) < 0.01, "Scores do not sum to 100%!"

    print("\n=================================================================")
    print("STEP 3: Combined Authenticity Score Logic Verification")
    print("=================================================================")
    sample_prog = {
        "human_real_percentage": 65.0,
        "ai_fake_percentage": 35.0,
        "classification": "AUTHENTIC / INTACT"
    }
    sample_ai = {
        "human_real_percentage": 30.0,
        "ai_fake_percentage": 70.0,
        "classification": "Fake"
    }
    combined = compute_combined_authenticity_score(sample_prog, sample_ai)
    print(f"Prog: Real {sample_prog['human_real_percentage']}%, Fake {sample_prog['ai_fake_percentage']}%")
    print(f"AI:   Real {sample_ai['human_real_percentage']}%, Fake {sample_ai['ai_fake_percentage']}%")
    print(f"Combined Real: {combined['overall_human_real_percentage']}%")
    print(f"Combined Fake: {combined['overall_ai_fake_percentage']}%")
    print(f"Discrepancy Delta: {combined['discrepancy_delta']}% (Disagree: {combined['detectors_disagree']})")
    print(f"Advisory: {combined['uncertainty_advisory']}")
    assert abs((combined['overall_human_real_percentage'] + combined['overall_ai_fake_percentage']) - 100.0) < 0.01, "Combined scores do not sum to 100%!"

    print("\n=================================================================")
    print("STEP 4: TestClient Dedicated Endpoint POST /api/v1/detect/image")
    print("=================================================================")
    client = TestClient(app)
    with open(test_img, "rb") as f:
        response = client.post("/api/v1/detect/image", files={"file": ("test.jpg", f, "image/jpeg")})
    
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}: {response.text}"
    data = response.json()
    
    prog_out = data["programmatic_detection"]
    ai_out = data["ai_model_detection"]
    comb_out = data["overall_combined_assessment"]

    print("\nVerification Checks:")
    print(f"1. Programmatic: Real={prog_out['human_real_percentage']}%, Fake={prog_out['ai_fake_percentage']}% -> Sum={prog_out['human_real_percentage'] + prog_out['ai_fake_percentage']:.2f}%")
    assert abs((prog_out['human_real_percentage'] + prog_out['ai_fake_percentage']) - 100.0) < 0.01

    print(f"2. AI Model: Real={ai_out['human_real_percentage']}%, Fake={ai_out['ai_fake_percentage']}% -> Sum={ai_out['human_real_percentage'] + ai_out['ai_fake_percentage']:.2f}%")
    assert abs((ai_out['human_real_percentage'] + ai_out['ai_fake_percentage']) - 100.0) < 0.01

    print(f"3. Combined: Real={comb_out['overall_human_real_percentage']}%, Fake={comb_out['overall_ai_fake_percentage']}% -> Sum={comb_out['overall_human_real_percentage'] + comb_out['overall_ai_fake_percentage']:.2f}%")
    assert abs((comb_out['overall_human_real_percentage'] + comb_out['overall_ai_fake_percentage']) - 100.0) < 0.01

    print("\n=================================================================")
    print("STEP 5: Full Investigation Pipeline POST /api/v1/investigate")
    print("=================================================================")
    with open(test_img, "rb") as f:
        inv_res = client.post("/api/v1/investigate", files={"file": ("test.jpg", f, "image/jpeg")})
    assert inv_res.status_code in (200, 202), f"Investigate failed: {inv_res.text}"
    job_id = inv_res.json()["job_id"]
    print(f"Job created: {job_id}")

    import time
    for _ in range(40):
        rep = client.get(f"/api/v1/report/{job_id}")
        if rep.status_code == 200:
            dossier = rep.json()
            if "authenticity_assessment" in dossier:
                print("✓ Dossier contains authenticity_assessment!")
                aa = dossier["authenticity_assessment"]
                print(f"Programmatic: {aa['programmatic_detection']['human_real_percentage']}% Real / {aa['programmatic_detection']['ai_fake_percentage']}% Fake")
                print(f"AI Model:     {aa['ai_model_detection']['human_real_percentage']}% Real / {aa['ai_model_detection']['ai_fake_percentage']}% Fake")
                print(f"Overall:      {aa['overall_combined_assessment']['overall_human_real_percentage']}% Real / {aa['overall_combined_assessment']['overall_ai_fake_percentage']}% Fake")
                break
        time.sleep(0.5)

    # Invalid upload test
    bad_res = client.post("/api/v1/detect/image", files={"file": ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream")})
    assert bad_res.status_code == 400, f"Expected 400 for bad file but got {bad_res.status_code}"
    print("✓ Graceful error handling for invalid/unsupported file validated (HTTP 400).")

    print("\n=================================================================")
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
