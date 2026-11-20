import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, Any

from app.config import settings


class EfficientNetDeepfakeDetector:
    """
    EfficientNet-B0 Deepfake & Synthetic Image Classifier
    Trained on 10,905 Fake/Real images:
    - Test Accuracy: 74.13%
    - Macro F1: 0.7387
    - ROC-AUC: 0.8292
    Classes: index 0 = Fake, index 1 = Real
    """

    def __init__(
        self,
        checkpoint_path: Path = settings.EFFICIENTNET_CHECKPOINT_PATH,
        device: str = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self._initialized = False

        # Official preprocessing pipeline specified in training checkpoint
        self.transform = transforms.Compose([
            transforms.Resize((settings.EFFICIENTNET_IMAGE_SIZE, settings.EFFICIENTNET_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _lazy_init(self):
        if self._initialized:
            return

        t0 = time.perf_counter()
        print(f"[EFFICIENTNET-B0] Initializing deepfake detector on device: {self.device}...")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"EfficientNet-B0 checkpoint not found at: {self.checkpoint_path}")

        # 1. Instantiate EfficientNet-B0 architecture
        self.model = models.efficientnet_b0(weights=None)
        
        # 2. Reconstruct the 2-class linear classifier
        in_features = self.model.classifier[1].in_features  # 1280
        self.model.classifier[1] = nn.Linear(in_features, 2)

        # 3. Load checkpoint state dict
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.metadata = {
                "architecture": checkpoint.get("architecture", "efficientnet_b0"),
                "class_to_idx": checkpoint.get("class_to_idx", {"Fake": 0, "Real": 1}),
                "test_accuracy": checkpoint.get("test_accuracy", 0.7413),
                "test_macro_f1": checkpoint.get("test_macro_f1", 0.7387),
                "test_roc_auc": checkpoint.get("test_roc_auc", 0.8292)
            }
        else:
            state_dict = checkpoint
            self.metadata = {
                "architecture": "efficientnet_b0",
                "class_to_idx": {"Fake": 0, "Real": 1},
                "test_accuracy": 0.7413,
                "test_macro_f1": 0.7387,
                "test_roc_auc": 0.8292
            }

        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch loading EfficientNet-B0: missing={missing}, unexpected={unexpected}")

        self.model.to(self.device)
        self.model.eval()
        self._initialized = True

        elapsed = time.perf_counter() - t0
        print(f"[EFFICIENTNET-B0] Model successfully loaded and verified in {elapsed:.3f}s.")

    def is_loaded(self) -> bool:
        return self._initialized

    @property
    def architecture(self) -> str:
        return self.metadata.get("architecture", "efficientnet_b0")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return self.metadata.get("class_to_idx", {"Fake": 0, "Real": 1})

    @property
    def metrics(self) -> Dict[str, float]:
        return {
            "test_accuracy": self.metadata.get("test_accuracy", 0.7413),
            "test_macro_f1": self.metadata.get("test_macro_f1", 0.7387),
            "test_roc_auc": self.metadata.get("test_roc_auc", 0.8292),
        }

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Runs inference on an uploaded image file using the fine-tuned EfficientNet-B0 model.
        Returns AI/Fake-like score, Human/Real-like score, and model classification.
        Percentages strictly sum to 100%.
        """
        self._lazy_init()
        t0 = time.perf_counter()

        # Load and convert image to RGB
        pil_img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)

        # Class mapping: index 0 = Fake, index 1 = Real
        fake_score = probabilities[0][0].item() * 100.0
        real_score = probabilities[0][1].item() * 100.0

        # Normalization guard: ensure exact 100.0% sum
        total = fake_score + real_score
        if total > 0:
            fake_score = (fake_score / total) * 100.0
            real_score = 100.0 - fake_score

        fake_pct = round(fake_score, 2)
        real_pct = round(100.0 - fake_pct, 2)

        classification = "Fake" if fake_pct >= 50.0 else "Real"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "model_name": "EfficientNet-B0 Deepfake Detector",
            "architecture": "efficientnet_b0",
            "checkpoint": self.checkpoint_path.name,
            "human_real_percentage": real_pct,
            "ai_fake_percentage": fake_pct,
            "human_real_score": real_pct,
            "ai_fake_score": fake_pct,
            "fake_score": fake_pct,
            "real_score": real_pct,
            "fake_percentage": fake_pct,
            "real_percentage": real_pct,
            "classification": classification,
            "raw_logits": [round(float(l), 4) for l in logits[0].cpu().tolist()],
            "probabilities": {
                "fake": round(fake_pct / 100.0, 4),
                "real": round(real_pct / 100.0, 4)
            },
            "metrics": self.metadata,
            "execution_time_ms": round(elapsed_ms, 2),
            "disclaimer": (
                "Scores represent softmax confidence from EfficientNet-B0 trained on 10,905 images "
                "(Test Accuracy: 74.13%, ROC-AUC: 0.8292). They reflect dataset pattern alignment, "
                "not definitive proof of authenticity."
            )
        }


efficientnet_detector = EfficientNetDeepfakeDetector()
