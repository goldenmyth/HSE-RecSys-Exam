# sweet.tv TV Programme Recommender

A recommendation-system project for the [sweet.tv Kaggle competition](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) and the accompanying [exam assignment](exam_quiz_FKN.md). The goal is to recommend five TV programme IDs to each of 1,260 users using viewing sessions and broadcast schedules.

## Final result

**Kaggle Private MAP@5: 0.42216**

The designated final prediction file is **[`submission_v4.csv`](submission_v4.csv)**. It contains the 1,260 users in the competition template's original order, with five distinct programme IDs per user.

The final strategy is a **Histogram-based Gradient Boosting (HGB) reranker** combining evidence from Personalized PageRank (PPR), implicit matrix factorization (ALS), content-based filtering, personal repeats, popularity, and viewing-time/channel preferences.

## Methods and results

All offline scores below use the same strict main holdout, the same 1,260 users, and the same MAP@5 implementation. A programme is relevant if one continuous viewing fragment covers **more than 80%** of its airing; this is a proxy for the competition's hidden labels, not verified Kaggle evaluator code.

| Method | Offline MAP@5 |
| --- | ---: |
| Popularity (distinct training users) | 0.432098 |
| Personal Repeat (>50% training views) | 0.502008 |
| Implicit ALS, 16 factors | 0.337698 |
| Content-based filtering | 0.237528 |
| Personalized PageRank | 0.517428 |
| PPR + future-airing availability | 0.524651 |
| **HGB reranker (final configuration)** | **0.552324** |

The HGB feature-group experiments were conducted cumulatively, keeping the candidate generation and evaluation protocol fixed:

| HGB configuration | Features | Offline MAP@5 |
| --- | ---: | ---: |
| A: model scores, ranks, popularity, availability | 15 | 0.535282 |
| B: A + viewing history | 22 | 0.547429 |
| C: B + time preferences | 28 | 0.548394 |
| **D: C + channel preferences** | **33** | **0.552324** |

The final Kaggle score (**0.42216**) is a separate hidden-test result, not directly interchangeable with the offline score. Earlier local estimates used a looser 60-second viewing definition and were substantially more optimistic.

## Recommendation pipeline

1. **Match viewing sessions to programmes.** Join by channel and overlapping half-open time intervals. Exclude unlabelled (`tv_show_id = 0`) and ambiguous airtime.
2. **Build training interactions.** For personalized methods, require a single continuous fragment covering more than 50% of the airing. Keep a distinct-user popularity fallback based on the original 60-second rule.
3. **Fit base recommenders.** ALS uses implicit count confidence; content-based filtering uses training-only metadata and TF-IDF profiles; PPR diffuses preferences through a co-view programme graph.
4. **Generate candidates.** Retain the availability-adjusted PPR shortlist and supplement it with Repeat, ALS, content-based, and popularity candidates to form 200 distinct eligible programmes per user.
5. **Rerank and return Top-5.** HGB uses 33 numeric features: 15 recommendation/availability features, seven history features, six time-affinity features, and five channel-affinity features. Unknown-history users receive the shared popularity fallback.

The HGB classifier uses 100 iterations, learning rate 0.08, at most 15 leaves, depth 4, minimum leaf size 100, L2 regularization 10, balanced class weights, and seed 42. Its scores are used for ranking; they are **not calibrated watch probabilities**.

## Temporal evaluation

| Stage | History available through | Target or forecast window |
| --- | --- | --- |
| HGB training snapshot | 2020-04-13 | 2020-04-13 to 2020-05-18 |
| Main validation | 2020-05-18 | 2020-05-18 to 2020-07-27 |
| Final test predictions | 2020-07-27 | 2020-07-27 to 2020-10-19 |

All prediction intervals are half-open. Base models and features for each historical snapshot use **only observations preceding that snapshot's cutoff**. Validation viewing supplies labels for evaluation, never features for those evaluated models. Following model selection, the final HGB is fitted using historical snapshots from two non-overlapping target windows, preserving their original feature cutoffs.

MAP@5 averages AP@5 over all 1,260 evaluation users, including users without a positive validation label (AP@5 = 0). Its AP denominator is `min(5, number of distinct relevant programmes)`. Recommending previously watched programmes is permitted, but duplicate IDs within a Top-5 list are not.

## Project structure

```text
scripts/              Data preparation, model experiments, final prediction
src/                  Metrics, base recommenders, PPR, HGB features and ranking
tests/                Automated checks for matching, metrics and model logic
results/              Recorded experiment metrics and small diagnostics
data/                 Kaggle data, processed interactions and model caches (ignored)
.venv/                Local Python environment (ignored)
requirements.txt      Pinned dependencies
exam_quiz_FKN.md       Exam specification
submission_v4.csv     Final submitted prediction file
```

The project was run with **Python 3.12.14 on macOS**. Core dependencies include `kagglehub==1.0.3`, `numpy==2.5.4`, `scipy==1.18.1`, `implicit==0.7.3`, and `scikit-learn==1.9.1`. See [`requirements.txt`](requirements.txt) for the complete environment. Exact numerical reproducibility across other platforms and BLAS builds is not guaranteed.

## Setup and data

From the repository root (macOS/Linux):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Accept the competition rules in your own Kaggle account and configure Kaggle credentials **outside** the repository. Then download the original files:

```bash
.venv/bin/python -B scripts/download_data.py
```

The script saves the viewing log, three broadcast schedules, and the original Kaggle submission **template** under `data/`. The template is an input, not the final prediction. The `.gitignore` excludes source data, `.venv`, processed interactions, fitted model caches, and secrets.

## Reproducing the results

Run the stages in order from the repository root, preferably in a **separate checkout**: some scripts regenerate local outputs. The final HGB fit requires the historical feature snapshots generated at their original cutoffs; do not delete those prerequisites before the last step.

```bash
# Data preparation and reference methods
.venv/bin/python -B scripts/prepare_interactions.py
.venv/bin/python -B scripts/evaluate_popularity.py
.venv/bin/python -B scripts/compare_models.py
.venv/bin/python -B scripts/build_submission.py

# Strict viewing labels and personalized-model experiments
.venv/bin/python -B scripts/strict_experiments.py
.venv/bin/python -B scripts/build_submission_v2.py

# History/schedule context and availability-aware PPR
.venv/bin/python -B scripts/prepare_history_context.py --mode development
.venv/bin/python -B scripts/refine_ppr.py
.venv/bin/python -B scripts/refine_ppr.py --final

# HGB feature ablations and final model
.venv/bin/python -B scripts/experiment_hgb.py
.venv/bin/python -B scripts/experiment_hgb.py --final
```

The last command creates **`submission_v4.csv`** after its prerequisite checks pass. None of these scripts submits predictions to Kaggle automatically.

Run automated tests separately:

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

Detailed configurations and recorded measurements are available in [`results/strict_experiments.json`](results/strict_experiments.json), [`results/ppr_refinement.json`](results/ppr_refinement.json), and [`results/hgb_experiments.json`](results/hgb_experiments.json).

## Limitations and acknowledgements

Only about **10.8%** of rows in the first broadcast schedule contain a nonzero programme ID. The records do not fully establish device-level channel switching or an explicit timezone. The >80% validation target approximates, but does not independently reproduce, the hidden competition labels. The validation period was reused for model selection, and the HGB training, validation, and final forecast horizons differ. The final reranker recommends a relatively small share of the eligible catalogue and uses popularity for cold-start users; no online A/B test or revenue effect was measured.

Development and evaluation were AI-assisted using **OpenAI Codex**. The strict viewing-proxy hypothesis was informed by a [published participant methodology](https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156); it is not treated as official judge code. The implementation uses the `implicit` package for ALS and scikit-learn for HGB. See the project report and recorded experiment results for methods, verification, and further discussion.
