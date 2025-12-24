# Budget-Aware Diabetes Risk Triage: Student Model + Escalation (LLM Review)
```markdown
# Diabetes Risk Triage System (3-Class) — FT-Transformer + (Optional) LLM Review (WIP)

This repository implements a **3-class diabetes risk stratification and triage system** based on tabular health indicators (BRFSS 2015 `Diabetes_012`: *No diabetes / Prediabetes / Diabetes*).  
The core idea is to build a **reliable, auditable pipeline**: a student model (FT-Transformer) produces **calibrated probabilities**, **uncertainty signals**, and **evidence**; borderline/low-confidence cases can be **escalated** to an optional LLM reviewer that returns **structured JSON** (WIP). The system is designed to support calibration, fairness evaluation, and future extensions (rules, report generation, distillation).

> Status: **Work in progress (WIP)** — core 3-class pipeline and uncertainty outputs are implemented; LLM review and full triage/clinical reporting are under development.

---

## Key Features

- **3-class student model**: FT-Transformer for `Diabetes_012` (0/1/2)
- **Probability calibration**: Temperature Scaling
- **Uncertainty estimation**: max-prob / entropy / margin (for escalation & rejection analysis)
- **Fairness evaluation**: group metrics (e.g., by sex / age groups)
- **(Optional, WIP) LLM second opinion**: structured JSON outputs for escalated cases
- **Artifact-friendly outputs**: model checkpoints, calibrator, metrics, and uncertain-case exports

---

## Project Structure

```

configs/
ftt_3class.yaml

scripts/
llm_review.py

src/diabetes_ml/
calibration/              # calibration + uncertainty
data/                     # preprocessing + dataset
fairness/                 # group fairness metrics
llm/                      # LLM reviewer (WIP)
metrics/                  # evaluation metrics
models/                   # ft_transformer.py
pipelines/                # training/evaluation pipelines
train/                    # trainer.py
utils/                    # utilities

main.py
requirements.txt

````

---

## Installation

Recommended: Python 3.9+ (3.10/3.11 preferred).

```bash
pip install -r requirements.txt
````

---

## Data (BRFSS 2015 / Diabetes_012)

This project expects a BRFSS 2015 diabetes dataset with the target column `Diabetes_012` (3 classes: 0/1/2).

**The dataset is not included in this repository.**
Configure the CSV path in `configs/ftt_3class.yaml`.

> Note: The current config may still use an absolute path during development.
> For open-source reproducibility, it is recommended to use a relative path such as:
> `data/diabetes_012_health_indicators_BRFSS2015.csv`

---

## Run: Train + Evaluate (FT-Transformer 3-Class)

```bash
python main.py --pipeline ftt_3class --config configs/ftt_3class.yaml
```

Typical outputs (depending on your pipeline implementation) will be saved under `outputs/`, such as:

* `model.pt` (student model weights)
* `scaler.pkl` (preprocessing/scaler if enabled)
* `temperature.pkl` (calibrator for temperature scaling)
* `metrics.json` / `metrics.md` (Macro-F1, AUROC(OVR), Brier, ECE, confusion matrix, etc.)
* `uncertain_samples_for_llm.csv` (exported uncertain cases for optional review)

---

## LLM Review

The repository contains an **experimental** LLM review component for escalated cases.
The intended behavior is: for low-confidence or conflicting samples, the LLM returns a **strict JSON** output (schema-checked), e.g.:

```json
{
  "class_probs": {"no": 0.12, "pre": 0.28, "dia": 0.60},
  "decision": "dia",
  "rationale_bullets": ["...", "..."],
  "cited_features": ["BMI", "PhysActivity", "HighBP", "HighChol"],
  "rule_conflict": false
}
```

Script entry (WIP):

```bash
python scripts/llm_review.py
```

> Recommendation (to be implemented): use environment variables for API keys (e.g., `OPENAI_API_KEY`) and enforce a JSON schema validator for auditability.

---

## Roadmap (System Design)

Planned end-to-end system (design document aligns with the accompanying PDF):

1. **Student 3-class prediction** (FT-Transformer)
2. **Calibration + uncertainty** (temperature scaling; entropy/max-prob/margin)
3. **Budget-aware triage** (risk–coverage trade-off for escalation)
4. **Structured second opinion** (LLM JSON review + validation)
5. **Fusion decision** (agreement/weighted fusion/gating; stacking later)
6. **(Optional) Distillation** (LLM soft labels to improve the student model)
7. **Report generation** (structured JSON → clinical/nursing-style report)
8. **Reliability & fairness** (ECE/Brier, rejection curves/AURC, conformal prediction, ΔTPR/ΔECE)

---

## Current Implementation Status

* [x] FT-Transformer 3-class training/evaluation pipeline
* [x] Temperature scaling calibration
* [x] Uncertainty outputs (entropy/max-prob/margin)
* [x] Basic group fairness metrics
* [ ] Budgeted triage (Top-K / quantile thresholding)
* [ ] LLM review: strict JSON schema + validator + cost/latency logging
* [ ] Fusion strategy ablations
* [ ] Distillation loop
* [ ] Report generation (HTML/PDF)
* [ ] Conformal prediction + DCA clinical utility

---

## Notes for Open Sourcing

* Add a `.gitignore` to exclude:

  * `outputs/`, `*.pt`, `*.pkl`, caches, IDE files, notebook checkpoints
* Avoid committing datasets and API keys
* Prefer relative paths in configs for portability

---

## Disclaimer

This repository is for research and educational purposes only and does **not** constitute medical advice. Any clinical use requires rigorous validation and regulatory compliance.

```
