# Phoenix — Digital Authenticity & Multi-Modal Forensic Investigation Platform

Phoenix (Image Trace) is a modular, high-performance digital forensic workbench built with **FastAPI**, **PyTorch**, **OpenCV**, **RoBERTa**, **EfficientNet**, **Wav2Vec2 XLS-R**, **DeMamba**, and **GenD**. It empowers researchers, investigative journalists, OSINT practitioners, and forensic analysts to investigate digital authenticity across **Images**, **Audio**, **Video**, and **Text**.

---

## Multi-Modal Forensic Capabilities

1. **Image Forensics (Dual-Engine)**
   - **Process A (Programmatic Forensics)**: Error Level Analysis (ELA), Laplacian spatial noise variance, copy-move clone clustering (ORB + RANSAC), EXIF/C2PA provenance, and 2D FFT frequency falloff.
   - **Process B (EfficientNet-B0 AI)**: Fine-tuned deepfake classifier (test ROC-AUC: 0.8292).
   - **Process C (Combined Overall Authenticity)**: 50/50 orthogonal physical vs neural score synthesis with automated divergence advisories.

2. **Audio Deepfake Detection**
   - Fine-tuned **DF Arena 1B** (Wav2Vec2 XLS-R 1B + 48-layer Conformer).
   - Mono 16 kHz resampled sliding windows with exact Spoof % vs Bonafide % score breakdown.

3. **Video Deepfake & AI Detection**
   - **DeMamba (X-CLIP + Mamba)**: Trained on GenVideo-100K to detect AI-generated synthetic video (Sora, Kling, Runway Gen-2, Pika, Crafter).
   - **GenD (CLIP-L/14 ViT)**: Cross-benchmark generalized facial deepfake detector (WACV 2026).
   - Integrated **YuNet** real-time facial tracking.

4. **Text AI-Authorship Forensics**
   - Fine-tuned **RoBERTa-base** binary sequence classifier (Class 0: Human-written, Class 1: AI-generated).
   - Evaluates linguistic entropy, perplexity distributions, and statistical token variance (Best Validation Macro F1: 0.9695).

---

## Quickstart

### 1. Launch the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Launch application
python run.py
```
Access the dashboard at:
```
http://127.0.0.1:8080
```

### 2. Run Test Suites
```bash
./venv/bin/python tests/test_text_detector.py
./venv/bin/python tests/test_authenticity_integration.py
./venv/bin/python tests/test_api.py
```

---

## Project Structure

```
.
├── app/
│   ├── config.py                # System settings and upload boundaries
│   ├── main.py                  # FastAPI server & route handlers
│   ├── core/
│   │   ├── base_module.py       # Abstract Base Class (ForensicModule)
│   │   ├── orchestrator.py      # Multithreaded pipeline & SSE publisher
│   │   └── security.py          # Magic byte inspection & file sanitization
│   ├── modules/
│   │   ├── text_detector.py     # Fine-tuned RoBERTa text detector
│   │   ├── efficientnet_detector.py # Fine-tuned EfficientNet-B0 image detector
│   │   ├── audio_detector.py    # DF Arena 1B audio deepfake detector
│   │   ├── video_detector.py    # GenD + DeMamba video deepfake detector
│   │   ├── authenticity_combiner.py # Dual-engine 50/50 score synthesis
│   │   ├── hashing_engine.py    # 7-algorithm image fingerprinting
│   │   ├── metadata_engine.py   # EXIF, XMP, and C2PA manifest checking
│   │   ├── manipulation_ela.py  # ELA, noise variance & copy-move detection
│   │   ├── sift_engine.py       # OpenCV SIFT/ORB keypoints & homography
│   │   ├── reverse_engine.py    # OpenCLIP embedding & vector search
│   │   └── ai_detector.py       # Hugging Face ViT & 2D FFT spectral analysis
│   ├── connectors/              # Instagram, Facebook, and Telegram connectors
│   ├── database/                # SQLite connection pool and repository
│   └── static/
│       ├── index.html           # Tactical cyber-forensic UI
│       ├── css/style.css        # Cyber-glassmorphism design system
│       └── js/app.js            # Frontend controller & real-time telemetry
├── corpus/                      # Verified reference archive
├── detector.py                  # Standalone text detector entry point
├── download_models.py           # Model retrieval helper
├── requirements.txt
├── run.py                       # Server entrypoint
└── tests/                       # Complete automated multi-modal test suite
```

---

## License
MIT License.
