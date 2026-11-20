import os
import sys

def download_models():
    print("Pre-caching AI models for offline forensic inference...")
    try:
        from transformers import pipeline
        print("1. Downloading Hugging Face AI detection model (Organika/sdxl-detector)...")
        pipeline("image-classification", model="Organika/sdxl-detector", device="cpu")
        print("   [SUCCESS] Organika/sdxl-detector cached locally.")
    except Exception as e:
        print(f"   [NOTICE] Model download skipped: {e}")

    try:
        import open_clip
        print("2. Downloading OpenCLIP weights (ViT-B-32 laion2b_s34b_b79k)...")
        open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")
        print("   [SUCCESS] OpenCLIP ViT-B-32 cached locally.")
    except Exception as e:
        print(f"   [NOTICE] OpenCLIP download skipped: {e}")

if __name__ == "__main__":
    download_models()
