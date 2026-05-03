# JEPA News Recommendation System

A self-supervised news recommendation system using **Joint-Embedding Predictive Architecture (JEPA)** on the [MIND dataset](https://msnews.github.io/), benchmarked against an XGBoost baseline.

## Architecture

```
User Click History → [Temporal Mask] → Context Encoder (4L Transformer)
                                              ↓
                                    Predictor (2L Transformer)
                                              ↓
                                    Predicted Target Embeddings
                                              ↕  smooth_L1 + VICReg
                                    Target Encoder (EMA copy)
                                              ↑
                        Masked Items → [Stop Gradient]
```

**Key design choices:**
- Transformer predictor (not MLP) with positional queries for per-target prediction
- EMA schedule: τ = 0.996 → 0.9999 (cosine anneal, BYOL convention)
- VICReg variance/covariance regularisation for proactive collapse prevention  
- Differential LR: item embeddings at 0.1× the encoder LR
- Temporal masking: mask last 20-40% of history (matches inference scenario)

## Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `data/download.py` | Download MIND-small dataset |
| 1 | `data/preprocess.py` | Parse TSVs, compute MiniLM embeddings, build sequences |
| 2 | `scripts/train_simcse.py` | SimCSE co-click pretraining for item embeddings |
| 3 | `scripts/train_jepa.py` | JEPA self-supervised pretraining |
| 4 | `scripts/finetune.py` | Fine-tune ranking head on labeled impressions |
| — | `baselines/xgboost_ranker.py` | XGBoost LambdaRank baseline (parallel track) |
| 5 | `scripts/evaluate.py` | Evaluate both models (AUC, MRR, nDCG@5, nDCG@10) |
| 6 | `scripts/error_analysis.py` | Collapse check, per-category breakdown, failure cases |

## Quick Start (Colab)

```python
# Clone and install
!git clone <your-repo-url> && cd jepa-news-rec
!pip install -r requirements.txt

# Run full pipeline
!python data/download.py --dataset mind-small
!python data/preprocess.py --dataset mind-small
!python scripts/train_simcse.py --epochs 5
!python scripts/train_jepa.py --epochs 10
!python scripts/finetune.py --epochs 3
!python baselines/xgboost_ranker.py
!python scripts/evaluate.py --model both
!python scripts/error_analysis.py
```

## Configuration

All hyperparameters in `configs/default.yaml`. Key settings:

- **Model**: 128d embeddings, 4-layer context encoder, 2-layer predictor
- **EMA**: τ = 0.996 → 0.9999 (cosine anneal)
- **Masking**: temporal, 20-40% from sequence end
- **Training**: AdamW, LR=1e-4, 10 epochs pretraining, 3 epochs fine-tuning

## Expected Results (MIND-small)

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|-------|-----|-----|--------|---------|
| XGBoost | ~0.63-0.65 | ~0.28-0.30 | ~0.30-0.32 | ~0.36-0.38 |
| JEPA | ~0.65-0.68 | ~0.30-0.33 | ~0.32-0.35 | ~0.38-0.41 |

*Note: MIND-small has 50k users — numbers will be lower than MIND-large leaderboard.*

## Project Structure

```
jepa-news-rec/
├── configs/default.yaml        # All hyperparameters
├── data/                       # Download + preprocessing
├── models/                     # Item encoder, context encoder, predictor, JEPA, SimCSE, ranking head
├── baselines/                  # XGBoost LambdaRank
├── evaluation/                 # AUC, MRR, nDCG metrics
├── scripts/                    # Training + evaluation entry points
├── utils/                      # Masking, EMA, VICReg, collapse monitor
└── requirements.txt
```
