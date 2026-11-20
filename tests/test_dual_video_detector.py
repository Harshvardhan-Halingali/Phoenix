from pathlib import Path
from app.modules.video_detector import video_detector

def test_dual_video_detector():
    test_video = Path("data/uploads/video/test_real_sim_video.mp4")
    assert test_video.exists(), "Test video does not exist"
    
    result = video_detector.predict(str(test_video), max_sample_frames=10, num_clips=2)
    
    assert result["success"] is True
    assert "spoof_ai_percentage" in result
    assert "bonafide_ai_percentage" in result
    assert "spoof_deepfake_percentage" in result
    assert "bonafide_deepfake_percentage" in result
    assert "prediction" in result
    assert "verdict_badge" in result
    assert "models_used" in result
    
    print(f"\n[TEST PASS] Dual Video Detector returned:")
    print(f"  Spoof AI: {result['spoof_ai_percentage']}%")
    print(f"  Bonafide AI: {result['bonafide_ai_percentage']}%")
    print(f"  Spoof Deepfake: {result['spoof_deepfake_percentage']}%")
    print(f"  Bonafide Deepfake: {result['bonafide_deepfake_percentage']}%")
    print(f"  Verdict: {result['verdict_badge']}")

if __name__ == "__main__":
    test_dual_video_detector()
