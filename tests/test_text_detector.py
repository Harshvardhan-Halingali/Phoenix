import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.modules.text_detector import text_detector, TextDetector
from detector import predict, predict_batch

client = TestClient(app)

def test_model_loading_and_architecture():
    """Verify RoBERTa-base checkpoint loads with exact 2-class head and metadata."""
    detector = text_detector
    detector._lazy_init()
    assert detector.is_loaded(), "Model failed to load!"
    assert detector.model.num_labels == 2, f"Expected 2 classes, got {detector.model.num_labels}"
    assert detector.tokenizer is not None, "Tokenizer is None"
    assert "model_name" in detector.metadata, "Metadata missing model_name"
    print(f"✓ Model verified: {detector.metadata.get('model_name')} on device: {detector.device}")

def test_human_text_inference():
    """Verify inference on authentic human text produces high Human-written score."""
    human_sample = (
        "Hey everyone! Just wanted to share a quick update on our weekend road trip. "
        "The weather was a bit unpredictable with sudden rain showers, but we found "
        "this amazing little coffee shop near the coastline that served the best scones."
    )
    res = text_detector.predict(human_sample)
    
    assert res["status"] == "success"
    assert res["prediction"] == "Human-written"
    assert res["label"] == "Human"
    assert res["human_score"] > 50.0, f"Expected Human > 50%, got {res['human_score']}%"
    assert res["ai_score"] < 50.0
    assert abs((res["human_score"] + res["ai_score"]) - 100.0) < 0.01, "Scores do not sum to 100%"
    assert res["truncated"] is False
    print(f"✓ Human text test passed: Human={res['human_score']}%, AI={res['ai_score']}%")

def test_ai_text_inference():
    """Verify inference on synthetic AI text produces high AI-generated score."""
    ai_sample = (
        "The integration of deep neural architectures into automated natural language processing "
        "represents a transformative evolution in contemporary artificial intelligence research. "
        "By leveraging multi-head self-attention mechanisms and hierarchical representations, "
        "transformer models systematically achieve unprecedented semantic comprehension."
    )
    res = text_detector.predict(ai_sample)
    
    assert res["status"] == "success"
    assert res["prediction"] == "AI-generated"
    assert res["label"] == "AI"
    assert res["ai_score"] > 50.0, f"Expected AI > 50%, got {res['ai_score']}%"
    assert res["human_score"] < 50.0
    assert abs((res["human_score"] + res["ai_score"]) - 100.0) < 0.01, "Scores do not sum to 100%"
    assert res["truncated"] is False
    print(f"✓ AI text test passed: AI={res['ai_score']}%, Human={res['human_score']}%")

def test_long_text_truncation():
    """Verify texts exceeding 256 tokens are flagged as truncated."""
    long_sample = "Artificial intelligence models analyze contextual language embeddings across vast domains. " * 40
    res = text_detector.predict(long_sample)
    
    assert res["truncated"] is True
    assert res["raw_token_count"] > 256
    assert abs((res["human_score"] + res["ai_score"]) - 100.0) < 0.01
    print(f"✓ Long text truncation test passed: raw_tokens={res['raw_token_count']}, truncated={res['truncated']}")

def test_chunked_prediction():
    """Verify chunked inference averages sliding windows across long text."""
    long_sample = "The machine learning algorithm iteratively optimizes gradient weights. " * 35
    res = text_detector.predict_chunks(long_sample)
    
    assert res["status"] == "success"
    assert res["truncated"] is True
    assert "chunks_evaluated" in res
    assert abs((res["human_score"] + res["ai_score"]) - 100.0) < 0.01
    print(f"✓ Chunked prediction test passed: chunks_evaluated={res['chunks_evaluated']}")

def test_batch_prediction():
    """Verify batch prediction helper."""
    batch = [
        "Hey bro, meet me at the library around noon.",
        "Deep learning frameworks utilize backpropagation to minimize objective loss functions."
    ]
    results = predict_batch(batch)
    assert len(results) == 2
    assert results[0]["prediction"] == "Human-written"
    assert results[1]["prediction"] == "AI-generated"
    print("✓ Batch prediction test passed.")

def test_empty_input_validation():
    """Verify empty or whitespace-only inputs are rejected with HTTP 400."""
    # Test through detector directly
    empty_caught = False
    try:
        text_detector.predict("")
    except ValueError:
        empty_caught = True
    assert empty_caught, "Failed to raise ValueError on empty string"

    whitespace_caught = False
    try:
        text_detector.predict("    \n\t  ")
    except ValueError:
        whitespace_caught = True
    assert whitespace_caught, "Failed to raise ValueError on whitespace string"

    # Test through FastAPI endpoint
    resp1 = client.post("/api/v1/detect/text", json={"text": ""})
    assert resp1.status_code == 400
    assert "empty" in resp1.json()["error"].lower()

    resp2 = client.post("/api/v1/detect/text", json={"text": "   "})
    assert resp2.status_code == 400

    print("✓ Empty input validation passed (HTTP 400).")

def test_fastapi_endpoints():
    """Verify /api/v1/detect/text, /api/detect/text, and /detect endpoints."""
    sample = "This is a simple human written note for testing purposes."
    
    for path in ["/api/v1/detect/text", "/api/detect/text", "/detect"]:
        resp = client.post(path, json={"text": sample})
        assert resp.status_code == 200, f"Failed on {path}: {resp.text}"
        data = resp.json()
        assert data["task"] == "ai_authorship_detection"
        assert "human_score" in data
        assert "ai_score" in data
        assert abs((data["human_score"] + data["ai_score"]) - 100.0) < 0.01
        print(f"✓ Endpoint {path} verified successfully.")

def test_existing_pipelines_preservation():
    """Verify existing GET / and audio/video modules are unharmed."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Text Forensics" in resp.text
    assert "RoBERTa-base" in resp.text
    assert "Image Forensics" in resp.text
    assert "Audio Deepfake" in resp.text
    assert "Video Deepfake" in resp.text
    print("✓ Existing UI & multi-modal routes fully preserved.")

if __name__ == "__main__":
    test_model_loading_and_architecture()
    test_human_text_inference()
    test_ai_text_inference()
    test_long_text_truncation()
    test_chunked_prediction()
    test_batch_prediction()
    test_empty_input_validation()
    test_fastapi_endpoints()
    test_existing_pipelines_preservation()
    print("\n=======================================================")
    print("ALL ROBERTA TEXT DETECTOR TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")
