import os
import sys
import json
from pathlib import Path
from starlette.testclient import TestClient

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.config import settings

client = TestClient(app)

def test_01_health_endpoints():
    print("\n--- Testing Health Endpoints ---")
    res = client.get("/health")
    assert res.status_code == 200, f"/health returned {res.status_code}: {res.text}"
    data = res.json()
    assert data["status"] == "healthy"
    print("Health response:", json.dumps(data, indent=2))

    res_v1 = client.get("/api/v1/health")
    assert res_v1.status_code == 200
    assert res_v1.json()["status"] == "healthy"
    print("✓ Health endpoints verified")

def test_02_cors_headers():
    print("\n--- Testing CORS Configuration ---")
    origin = "https://phoenix-app.pages.dev"
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    res = client.options("/api/v1/investigate", headers=headers)
    print("Preflight response code:", res.status_code)
    print("Preflight CORS header:", res.headers.get("access-control-allow-origin"))
    assert res.headers.get("access-control-allow-origin") in ["*", origin]

    # Test actual GET with Origin
    res_get = client.get("/health", headers={"Origin": origin})
    assert res_get.headers.get("access-control-allow-origin") in ["*", origin]
    print("✓ CORS headers verified for Cloudflare Pages origin")

def test_03_text_detector():
    print("\n--- Testing RoBERTa AI Text Detection ---")
    # Human-like sample
    human_text = "I went to the store today to buy fresh groceries. The weather was unusually brisk for late November."
    res = client.post("/api/v1/detect/text", json={"text": human_text})
    assert res.status_code == 200, f"Error: {res.text}"
    data = res.json()
    print("Human sample inference:", json.dumps(data, indent=2))
    assert "prediction" in data
    assert "ai_score" in data
    assert "human_score" in data

    # AI-like sample
    ai_text = "In conclusion, the migration towards decentralized cloud environments requires a multifaceted paradigm shift that optimizes container virtualization while mitigating latency across distributed microservice architectures."
    res_ai = client.post("/api/v1/detect/text", json={"text": ai_text})
    assert res_ai.status_code == 200
    data_ai = res_ai.json()
    print("AI sample inference:", json.dumps(data_ai, indent=2))
    assert data_ai["ai_score"] is not None
    print("✓ Text detector model inference verified")

def test_04_image_detector():
    print("\n--- Testing Image Detection (EfficientNet-B0 + Programmatic) ---")
    sample_img = Path("data/uploads/sample_clean.jpg")
    if not sample_img.exists():
        # pick any jpeg/png in data/uploads
        candidates = list(Path("data/uploads").glob("*.jpg")) + list(Path("data/uploads").glob("*.jpeg"))
        assert len(candidates) > 0, "No sample image found"
        sample_img = candidates[0]

    with open(sample_img, "rb") as f:
        res = client.post("/api/v1/detect/image", files={"file": ("sample.jpg", f, "image/jpeg")})
    
    assert res.status_code == 200, f"Image detection failed: {res.text}"
    data = res.json()
    print("Image detection response keys:", list(data.keys()))
    assert "ai_model_detection" in data
    assert "programmatic_detection" in data
    assert "overall_combined_assessment" in data
    print(f"EfficientNet-B0 result: {data['ai_model_detection']}")
    print(f"Overall verdict: {data['overall_combined_assessment'].get('verdict') or data['overall_combined_assessment'].get('overall_verdict')}")
    print("✓ Image detection model inference verified")

def test_05_audio_detector():
    print("\n--- Testing Audio Detection (DF Arena 1B) ---")
    sample_audio = Path("data/uploads/audio/speech_like_sim.wav")
    if not sample_audio.exists():
        sample_audio = Path("data/uploads/audio/tts_generated_sample.wav")
    assert sample_audio.exists(), "Audio test file not found"

    with open(sample_audio, "rb") as f:
        res = client.post("/api/v1/detect/audio", files={"file": ("audio_sample.wav", f, "audio/wav")})

    assert res.status_code == 200, f"Audio detection failed: {res.text}"
    data = res.json()
    print("Audio detection response:", json.dumps(data, indent=2))
    assert data.get("success") is True
    assert "prediction" in data
    assert "spoof_probability" in data
    assert "bonafide_probability" in data
    print("✓ Audio detector model inference verified")

def test_06_frontend_static_assets():
    print("\n--- Testing Frontend Assets for Cloudflare Pages ---")
    static_dir = Path("app/static")
    assert (static_dir / "index.html").exists(), "index.html missing"
    assert (static_dir / "js" / "app.js").exists(), "app.js missing"
    assert (static_dir / "config.js").exists(), "config.js missing"
    
    # Check that config.js has API_BASE_URL
    config_content = (static_dir / "config.js").read_text()
    assert "window.API_BASE_URL" in config_content
    
    # Check that app.js uses apiUrl helper
    app_js_content = (static_dir / "js" / "app.js").read_text()
    assert "function apiUrl(" in app_js_content
    assert "apiUrl(\"/api/v1/detect/text\")" in app_js_content
    assert "apiUrl(\"/api/v1/detect/audio\")" in app_js_content
    assert "apiUrl(\"/api/v1/detect/video\")" in app_js_content
    assert "apiUrl(\"/api/v1/connectors/status\")" in app_js_content
    assert "apiUrl(\"/api/v1/investigate\")" in app_js_content
    print("✓ Frontend static assets verified for Cloudflare Pages")

if __name__ == "__main__":
    test_01_health_endpoints()
    test_02_cors_headers()
    test_03_text_detector()
    test_04_image_detector()
    test_05_audio_detector()
    test_06_frontend_static_assets()
    print("\nALL 6 VERIFICATION SUITES PASSED SUCCESSFULLY!")
