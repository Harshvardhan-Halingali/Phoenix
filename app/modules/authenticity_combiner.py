import time
import math
from typing import Dict, Any, List


def compute_combined_authenticity_score(
    programmatic_result: Dict[str, Any],
    ai_model_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes a technically justified Combined Image Authenticity Assessment
    from two independent processes:
      1. Process A: Programmatic Detection (Physical ELA, noise variance, copy-move, EXIF, FFT)
      2. Process B: AI Model Detection (Trained EfficientNet-B0 Deepfake Classifier)

    Weights:
      w_programmatic = 0.50 (50%) - Evaluates signal-level integrity, compression physics, and provenance.
      w_efficientnet = 0.50 (50%) - Evaluates learned hierarchical convolutional facial & image features (ROC-AUC 0.8292).

    Formula:
      combined_fake_score = w_prog * prog_fake + w_ai * ai_fake
      combined_real_score = 100.0 - combined_fake_score
      Both percentages strictly sum to 100.0%.
    """
    # 1. Extract Programmatic Percentages
    prog_fake = float(programmatic_result.get("ai_fake_percentage", programmatic_result.get("fake_percentage", 50.0)))
    prog_real = float(programmatic_result.get("human_real_percentage", programmatic_result.get("real_percentage", 100.0 - prog_fake)))
    
    # Ensure exact 100% sum
    prog_total = prog_fake + prog_real
    if prog_total > 0:
        prog_fake = (prog_fake / prog_total) * 100.0
        prog_real = 100.0 - prog_fake

    # 2. Extract AI Model (EfficientNet-B0) Percentages
    ai_fake = float(ai_model_result.get("ai_fake_percentage", ai_model_result.get("fake_percentage", 50.0)))
    ai_real = float(ai_model_result.get("human_real_percentage", ai_model_result.get("real_percentage", 100.0 - ai_fake)))
    
    ai_total = ai_fake + ai_real
    if ai_total > 0:
        ai_fake = (ai_fake / ai_total) * 100.0
        ai_real = 100.0 - ai_fake

    # 3. Technically Justified Combination Weights
    w_prog = 0.50
    w_ai = 0.50

    combined_fake = round((w_prog * prog_fake) + (w_ai * ai_fake), 2)
    combined_real = round(100.0 - combined_fake, 2)

    # 4. Measure Discrepancy & Detector Agreement
    discrepancy_delta = round(abs(prog_fake - ai_fake), 2)
    detectors_disagree = discrepancy_delta >= 30.0

    # 5. Determine Overall Classification Bracket
    if combined_real >= 65.0:
        overall_classification = "AUTHENTIC / HUMAN-LIKE CAPTURE"
        verdict_badge = "HUMAN_AUTHENTIC"
        assessment_summary = (
            f"MUTUAL CORROBORATION: Image exhibits natural physical capture characteristics "
            f"(Programmatic Real: {prog_real:.1f}%) and organic human facial features (EfficientNet-B0 Real: {ai_real:.1f}%). "
            f"Combined Authenticity Score: {combined_real:.1f}% Real."
        )
    elif combined_fake >= 65.0:
        overall_classification = "AI-GENERATED / DEEPFAKE-LIKE MANIPULATION"
        verdict_badge = "AI_DEEPFAKE"
        assessment_summary = (
            f"MANIPULATION DETECTED: Image exhibits synthetic generation or localized tampering patterns "
            f"(Programmatic Fake: {prog_fake:.1f}%, EfficientNet-B0 Fake: {ai_fake:.1f}%). "
            f"Combined Synthetic Likelihood: {combined_fake:.1f}% Fake."
        )
    else:
        overall_classification = "AMBIGUOUS / INCONCLUSIVE EVIDENCE"
        verdict_badge = "INCONCLUSIVE"
        assessment_summary = (
            f"EVIDENCE DIVERGENCE: The two forensic processes produced differing indications "
            f"(Programmatic Fake: {prog_fake:.1f}%, EfficientNet-B0 Fake: {ai_fake:.1f}%). "
            f"Combined Assessment is ambiguous ({combined_real:.1f}% Real vs {combined_fake:.1f}% Fake). Manual inspection recommended."
        )

    # 6. Uncertainty Advisory if detectors diverge
    if detectors_disagree:
        prog_view = "Authentic Camera" if prog_real >= 50 else "Tampered / Anomalous"
        ai_view = "Real" if ai_real >= 50 else "Synthetic / Fake"
        uncertainty_advisory = (
            f"DETECTOR DIVERGENCE NOTICE (Discrepancy: {discrepancy_delta}%): Programmatic physical forensics assessed image as "
            f"'{prog_view}' ({prog_real:.1f}% Real), while the EfficientNet-B0 classifier assessed it as '{ai_view}' ({ai_fake:.1f}% Fake). "
            f"Such divergence commonly occurs when re-saving/compression alters high-frequency pixel noise or when synthetic content is blended into real backgrounds. "
            f"Refer to individual detector signals below."
        )
    else:
        uncertainty_advisory = (
            f"DETECTOR CONVERGENCE: Both detection processes mutually corroborate each other within {discrepancy_delta}% margin."
        )

    return {
        "overall_human_real_percentage": combined_real,
        "overall_ai_fake_percentage": combined_fake,
        "overall_classification": overall_classification,
        "verdict_badge": verdict_badge,
        "assessment_summary": assessment_summary,
        "detectors_disagree": detectors_disagree,
        "discrepancy_delta": discrepancy_delta,
        "uncertainty_advisory": uncertainty_advisory,
        "combination_methodology": {
            "formula": "Overall_Fake = (0.50 * Programmatic_Fake) + (0.50 * EfficientNet_Fake)",
            "weights": {"programmatic_physical": w_prog, "efficientnet_ai": w_ai},
            "weight_justification": (
                "50% assigned to programmatic physical forensics (ELA, noise variance, copy-move, EXIF, FFT) "
                "and 50% assigned to EfficientNet-B0 learned deepfake classifier (test accuracy: 74.13%, ROC-AUC: 0.8292). "
                "Outputs represent complementary physical and neural dimensions."
            ),
            "limitations": (
                "The combined assessment is a structured weighted synthesis of empirical signals. "
                "It represents model confidence and forensic indicators, not definitive legal proof of image authenticity."
            )
        }
    }
