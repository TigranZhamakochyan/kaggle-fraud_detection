# Predicting Credit Card Fraud: Kaggle Playground Series S6E11

My solution notebooks for [Playground Series Season 6, Episode 11](https://www.kaggle.com/competitions/playground-series-s6e11). The task is to predict `is_fraud` for card transactions; submissions are scored by ROC AUC.

- 699,960 training rows, 299,984 test rows, 24 features, 1.67% fraud
- All models are validated on the same split: `StratifiedKFold(5, shuffle=True, random_state=42)`

## Results

| Step | Model | CV AUC | Public LB |
|---|---|---|---|
| 1 | Blend of default LightGBM, XGBoost and CatBoost | 0.94701 | 0.94718 |
| 2 | Tuned LightGBM (Optuna) | 0.94754 | 0.94798 |
| 3 | Digit features as categories, 4-leaf LightGBM | 0.94855 | 0.94914 |
| 4 | + in-fold target encoding of exact values (blend) | 0.94956 | 0.95027 |
| 5 | + target encoding of digit keys (blend) | 0.95031 | 0.95118 |
| 6 | Trees boosted from a logistic regression (blend) | 0.95067 | 0.95130 |
| 6b | Step 6 refit on all training rows | | 0.95156 |
| 7 | Stronger logistic model (pair keys, raw inputs), boosted trees (blend) | 0.95097 | 0.95114 |
| 8 | Public wide stack from goodpjw2008's published predictions | 0.95256 (nested) | 0.95277 |

Step 8 is mostly other people's work: it blends goodpjw2008's published 200-member stacks. My own best models (step 7) scored about 0.9510 CV.

## What worked

1. **Tiny trees.** The labels behave almost like a logistic formula, so 4-leaf LightGBM and depth-2 XGBoost beat deeper trees.
2. **Digit features.** The synthetic data generator leaves patterns in individual digits (for example, `velocity_score` values ending in `.0` have a lower fraud rate). Each digit of 8 numeric columns becomes a categorical feature.
3. **Per-value target encoding**, built strictly inside each fold (nested out-of-fold), of every column and of "digit keys": cents, `int(x)`, `int(x/10)`, `int(10x)` and leading digits.
4. **Boosting from a logistic regression.** A logistic regression on the encoding logits gives the starting score (`init_score`) for 4-leaf LightGBM, which then fits only the corrections.
5. **A stronger logistic model**: 13 pair keys, one-hot categories, log-scaled numbers, smoothing 50 and the `newton-cholesky` solver.
6. **Refitting on all rows** for the test predictions (once +0.00026 on the public LB; not repeatable for step 7).

## What didn't work

- Hand-made features (flag counts, interactions, ratios): +0.00003
- Adding the [original dataset](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) as extra rows: -0.00026
- Fraud rates and counts looked up from the original dataset: worse on both tested folds
- Seed averaging of models: about +0.0001 at most
- Re-tuning with Optuna after adding digit features: no gain
- A nested logistic stacker instead of hand-picked blend weights: no gain
- Adding my own models to the public wide stack: no gain (rank correlation about 0.98)

## Notebooks

- `detect.ipynb`: exploration, all models, blends and submissions, in the order they were built
- `mix.ipynb`: blending with the published out-of-fold predictions, checked with nested CV

The Optuna cells take 20+ minutes each, so avoid **Run All**. `run_cv` saves out-of-fold and test predictions to `oof/` and reloads them on later runs.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Join the competition on Kaggle and log in to `kagglehub`. The notebooks download the data with:

```python
import kagglehub
path = kagglehub.competition_download("playground-series-s6e11")
```

The data, saved predictions and submission files are not included in this repository.

## Credits

Several key ideas came from the Kaggle community, all used on the same folds:

- **goodpjw2008**: digit keys, tiny trees and the published wide stack and prediction artifacts (`s6e11-fraud-series-artifacts`)
- **Yash Teotia**: "S6E11 Beginner to Expert, Every Step Measured", the stronger logistic-model recipe (pair keys, raw inputs, smoothing 50)
- **megayak**: boosting trees from a logistic model's logit
- The discussion post that first pointed out digit features
