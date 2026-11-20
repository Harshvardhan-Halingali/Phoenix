import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import settings


class TextDetector:
    """
    Fine-Tuned RoBERTa AI-Generated vs Human-Written Text Classifier.
    
    Model configuration:
      - Architecture: RoBERTa-base (12 layers, 768 hidden, 12 heads)
      - Task: Binary text classification
      - Class 0: Human-written
      - Class 1: AI-generated
      - Maximum sequence length: 256 tokens
      - Training samples: 5,770 | Val: 722 | Test: 722
      - Best validation Macro F1: 0.9695 (Epoch 2)
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        if model_dir is None:
            self.model_dir = settings.ROBERTA_MODEL_DIR
            if not self.model_dir.exists():
                alt_dir = settings.BASE_DIR / "data" / "models" / "text" / "roberta_text_detector"
                if alt_dir.exists():
                    self.model_dir = alt_dir
        else:
            self.model_dir = Path(model_dir)

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = None
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        t0 = time.perf_counter()
        print(f"[ROBERTA TEXT DETECTOR] Initializing fine-tuned model from: {self.model_dir} on {self.device}...")

        if not self.model_dir.exists():
            raise FileNotFoundError(f"RoBERTa model directory does not exist at: {self.model_dir}")

        config_file = self.model_dir / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Missing config.json in {self.model_dir}")

        # Check weights file (safetensors or bin)
        weights_safetensors = self.model_dir / "model.safetensors"
        weights_bin = self.model_dir / "pytorch_model.bin"
        if not weights_safetensors.exists() and not weights_bin.exists():
            raise FileNotFoundError(f"No model weights found (model.safetensors or pytorch_model.bin) in {self.model_dir}")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)

        # Load Sequence Classification Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            local_files_only=True
        )

        # Verify 2-class head
        if self.model.num_labels != 2:
            raise ValueError(f"Expected 2-class classifier, but model has {self.model.num_labels} labels.")

        # Load training metadata if present
        meta_file = self.model_dir / "training_metadata.pt"
        if meta_file.exists():
            try:
                self.metadata = torch.load(str(meta_file), map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[ROBERTA TEXT DETECTOR] Warning reading metadata: {e}")
                self.metadata = {
                    "model_name": "roberta-base",
                    "best_validation_macro_f1": 0.9695,
                    "max_length": 256
                }
        else:
            self.metadata = {
                "model_name": "roberta-base",
                "best_validation_macro_f1": 0.9695,
                "max_length": 256
            }

        self.model.to(self.device)
        self.model.eval()
        self._initialized = True

        elapsed = time.perf_counter() - t0
        print(f"[ROBERTA TEXT DETECTOR] Successfully loaded and verified in {elapsed:.3f}s.")

    def is_loaded(self) -> bool:
        return self._initialized

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs inference on input text and returns Human-written vs AI-generated scores.
        Scores strictly sum to 100.0%.
        """
        if not self._initialized:
            self._lazy_init()

        if text is None or not text.strip():
            raise ValueError("Text input cannot be empty or whitespace only.")

        cleaned_text = text.strip()
        t0 = time.perf_counter()

        # Check raw token count to detect truncation accurately
        raw_tokens = self.tokenizer.encode(cleaned_text, add_special_tokens=True)
        raw_token_count = len(raw_tokens)
        max_len = int(self.metadata.get("max_length", 256))
        is_truncated = raw_token_count > max_len

        # Tokenize with exact training parameters
        inputs = self.tokenizer(
            cleaned_text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)[0]

        # Class mapping: 0 = Human, 1 = AI
        raw_human = probabilities[0].item() * 100.0
        raw_ai = probabilities[1].item() * 100.0

        # Normalization guard: strictly sum to 100.0%
        total = raw_human + raw_ai
        if total > 0:
            human_score = (raw_human / total) * 100.0
            ai_score = 100.0 - human_score
        else:
            human_score = 50.0
            ai_score = 50.0

        human_score = round(human_score, 2)
        ai_score = round(100.0 - human_score, 2)

        # Classification decision
        if ai_score >= 50.0:
            prediction = "AI-generated"
            classification_label = "Likely AI-generated"
        else:
            prediction = "Human-written"
            classification_label = "Likely Human-written"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "task": "ai_authorship_detection",
            "prediction": prediction,
            "classification_label": classification_label,
            "human_score": human_score,
            "ai_score": ai_score,
            "human_percentage": human_score,
            "ai_percentage": ai_score,
            "label": "AI" if ai_score >= 50.0 else "Human",
            "confidence": round(max(human_score, ai_score) / 100.0, 4),
            "probs": {
                "Human": round(human_score / 100.0, 4),
                "AI": round(ai_score / 100.0, 4)
            },
            "raw_logits": [round(float(l), 4) for l in logits[0].cpu().tolist()],
            "model": "Fine-tuned RoBERTa-base",
            "truncated": is_truncated,
            "raw_token_count": raw_token_count,
            "max_sequence_length": max_len,
            "status": "success",
            "execution_time_ms": round(elapsed_ms, 2),
            "disclaimer": (
                "Scores represent softmax confidence from a fine-tuned RoBERTa-base model trained on "
                "Human-written and ChatGPT-generated texts. They are not proof of authorship or factual truth."
            )
        }

    def predict_chunks(self, text: str, max_length: int = 256, stride: int = 64) -> Dict[str, Any]:
        """
        Splits text longer than 256 tokens into overlapping chunks with stride,
        averages the probabilities, and returns the aggregate score.
        """
        if not self._initialized:
            self._lazy_init()

        if text is None or not text.strip():
            raise ValueError("Text input cannot be empty or whitespace only.")

        cleaned_text = text.strip()
        tokens = self.tokenizer.encode(cleaned_text, add_special_tokens=False)

        if len(tokens) <= max_length:
            return self.predict(cleaned_text)

        # Build overlapping chunks
        chunk_probs = []
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + max_length]
            if len(chunk_tokens) < 10:  # ignore tiny tail
                break
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            res = self.predict(chunk_text)
            chunk_probs.append((res["human_score"], res["ai_score"]))

        if not chunk_probs:
            return self.predict(cleaned_text)

        avg_human = sum(p[0] for p in chunk_probs) / len(chunk_probs)
        avg_ai = 100.0 - avg_human

        avg_human = round(avg_human, 2)
        avg_ai = round(100.0 - avg_human, 2)

        prediction = "AI-generated" if avg_ai >= 50.0 else "Human-written"
        return {
            "task": "ai_authorship_detection",
            "prediction": prediction,
            "classification_label": "Likely AI-generated" if avg_ai >= 50.0 else "Likely Human-written",
            "human_score": avg_human,
            "ai_score": avg_ai,
            "human_percentage": avg_human,
            "ai_percentage": avg_ai,
            "label": "AI" if avg_ai >= 50.0 else "Human",
            "confidence": round(max(avg_human, avg_ai) / 100.0, 4),
            "probs": {"Human": round(avg_human / 100.0, 4), "AI": round(avg_ai / 100.0, 4)},
            "model": "Fine-tuned RoBERTa-base (Chunked)",
            "truncated": True,
            "chunks_evaluated": len(chunk_probs),
            "status": "success"
        }

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Performs batch inference on a list of texts.
        """
        results = []
        for t in texts:
            if not t or not t.strip():
                results.append({"status": "error", "error": "Empty text"})
            else:
                results.append(self.predict(t))
        return results


# Global singleton instance
text_detector = TextDetector()
