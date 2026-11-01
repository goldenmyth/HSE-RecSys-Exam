# Final Exam Quiz (FKN): TV Program Recommendation on sweet.tv

**Deadline:** 1 November 2026, 19:00 MSK. **Individual submission.**

## 1. Purpose

Apply recommendation methods to an unfamiliar dataset, compare candidate methods fairly on one documented hold-out, and justify a final strategy for the platform. You will implement at least two methods, submit a ranked five-programme list per test user to Kaggle, and argue for a strategy a real service could adopt.

## 2. The competition

- **Competition page:** [the competition page on Kaggle](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) — title **sweet.tv — TV program recommender**, slug `sweettv-tv-program-recommender`, id 24201, host `ogurbych`, a community prediction competition.
- **Accept the rules first.** On that page open the *Rules* tab and click *I Understand and Accept* with your own Kaggle account. Until you do, the data download and the submit button both return 403. Acceptance is per account: your instructor's acceptance does not apply to yours.
- Metric: **MAP@{K}, higher is better**.
- **K = 5.** The page states the metric only as MAP@{K}. The submission template carries five space-separated programme ids per row, so this quiz fixes K = 5 to match the file you must produce. Use K = 5 everywhere: submission, hold-out, reported numbers.
- **The competition accepts late submissions.** The competition closed on 2020-11-21, but late submissions are still accepted and scored. Verified 2026-10-21: the unmodified template was submitted (ref 57484768) and returned status COMPLETE with **public 0.00000** and **private 0.00000**.
- **The leaderboard is frozen**, so a submission cannot change any standing. What matters here is the **returned private score**: the cohort ranking in §10 uses it, never your leaderboard position.

## 3. Data

| File | Size | Rows | Role |
| --- | ---: | ---: | --- |
| `dataset11-30.csv` | 396 MB | 5,821,987 | view log |
| `export_arh_11-20-final.csv` | 153 MB | 1,074,235 | broadcast schedule, training weeks |
| `export_arh_21-30-final.csv` | 154 MB | 1,074,394 | broadcast schedule, training weeks |
| `export_arh_31-42-final.csv` | 186 MB | 1,295,378 | broadcast schedule, **test** weeks |
| `submission.csv` | — | 1,260 rows x 5 ids | template to fill |

Archive 131 MB, 890 MB extracted, 5 files, 9,265,994 rows across the four data files.

**View log columns:** `user_id, vsetv_id, start_time, stop_time, duraton`. `duraton` (misspelled in the file) is in seconds and equals `stop_time` minus `start_time` — the length of the viewing interval, **not** the length of the programme. Median 396 s, max 342,145 s.

**Schedule columns:** `channel_id, channel_title, start_time, duration, tv_show_title, tv_show_id, tv_show_category, tv_show_genre_1, tv_show_genre_2, tv_show_genre_3, year_of_production, director, actors`.

**Scale and coverage:** 5,848 / 6,364 / 6,637 distinct programmes per schedule file; 400 / 403 / 404 channels; 9 categories.

**The labelling caveat that shapes this task: only 10.8% of schedule rows carry a programme id.** In weeks 11-20, 115,613 of 1,074,235 rows have `tv_show_id != 0`; the other 958,622 are unlabelled airtime (`tv_show_id = 0`). Most of the grid has a channel, a time slot and a title but **no programme id**, so only labelled rows can become an item.

**Reading cost:** all four files read with the standard-library `csv` module in about 4.6 s at 12.6 MB peak memory. Repeated passes are cheap, so prefer a streaming design and document your environment.

**Your task is a join, then a prediction.** Join the view log to the schedule — same channel, and the user's view interval overlapping a programme's airing interval — to obtain user-programme interactions. Then predict, for each of the 1,260 test users, the five programmes they will watch in weeks 31-42.

Two consequences to plan for. The test weeks are **later than all training weeks**, so every method must extrapolate forward. And any user absent from the training window, or whose history touches only unlabelled airtime, is a **cold-start** case with no usable interactions.

## 4. Implementation task

Implement **at least two** of:

1. **Content-based filtering.** Now allowed, because the schedule supplies `tv_show_category`, three genre fields, `year_of_production`, `director` and `actors`. Build item vectors from these attributes and a user profile from that user's watched items.
2. **Association rules** over each user's history or time windows.
3. **Matrix factorization** for implicit feedback over user-programme counts, plain or duration-weighted, since no rating scale exists.
4. **Personalized PageRank** on a programme co-view graph, personalised from each user's history.

A popularity or mean baseline does **not** count as one of your two methods, but a **popularity baseline is required** as the reference every method is measured against.

## 5. Common requirements

- **Define how a viewing is matched to a programme.** State the join rule (channel plus interval overlap) and the threshold you use to call an overlap "watched". The data has **no programme-length-normalised watch percentage built in**, so you must construct any normalisation yourself and justify it against the observed duration distribution — an absolute-seconds cut and a fraction-of-`duration` cut are both defensible, but you must say which you used and why.
- **State how `tv_show_id = 0` rows are handled.** They are unlabelled airtime, not items. Say whether you drop them, use them as watching context, or treat them as misses.
- **Define your own validation split and justify it.** The test window is weeks 31-42, so validation must come from weeks 21-30 or earlier. Preserve temporal order.
- **Build every model from training data only.** Validation is for selection, never fitting.
- **Define the candidate pool.** It is bounded by the labelled programmes in the schedule; state whether recommending an already-watched programme is allowed. The competition does not forbid repeats; your choice must be explicit and identical across methods.
- **Handle users with no history.** State exactly what your system returns for them, and why — and give the same fallback everywhere.
- **Explain how each method produces exactly five programme ids.** Scores from different methods are not interchangeable probabilities — say how each output becomes a ranked list of five.
- **Document** hyperparameters, random seeds, runtime and environment.
- **Never use future viewing behaviour.** A user's views in the test window (weeks 31-42) may not be used in training or in any feature. The test-period schedule may be used only to know which programme airs on which channel at which time; it must never be used to look up who watched what.

## 6. Final strategy

Submit one implemented method, or a hybrid combining methods by score fusion, rank fusion, candidate generation with re-ranking, or a documented switching rule. For a hybrid, state its components, how weights or thresholds were chosen, and how it compares with each individual method on the hold-out. A hybrid is optional; combining two methods without implementing and evaluating each one individually does not satisfy the comparison requirement.

## 7. Recommendation examples

Choose **two users yourself** and justify the choice — for example one heavy and one light viewer, or one user with no usable history. Use the **same two users across all methods**. For each user, show **each method's** Top-5 in order, with the programme id, the title and the category looked up from the schedule, so the methods can be read side by side. Reproduce the ids in your report. If a method cannot fill five slots, say so and explain the fallback.

## 8. Business and algorithmic analysis — one required question

**Which recommendation strategy would you select for this platform, and why?**

Using the same two users, compare each method's five ids against what those users actually watched, and discuss:

- whether the lists reflect familiar preferences or offer relevant new programmes;
- how and why the methods differ — what association rules capture that matrix factorization misses, what content attributes contribute where co-view signal is thin, how PageRank's graph diffusion reshapes the ranking, and where the popularity baseline already does well;
- why your final strategy is preferable, considering validation performance, computational cost at this scale, and what the lists imply for the user experience;
- limitations and failure cases your results actually support — beginning with the fact that only 10.8% of airtime carries an id.

Two cautions. An unfamiliar programme is **not** automatically a useful discovery — it may simply be a poor recommendation, and nothing here reveals whether a user enjoyed it. And absence from the log does not establish that a user has never seen a programme: the log records this platform's views only. Qualify "never", "always" and "better" to the evidence and scope you examined.

**No live A/B test, online experiment or revenue claim is required — and none may be claimed.** Every assertion must follow from the supplied data and your implementation.

## 9. Submission

Submit exactly two files:
- your report as a **PDF**;
- the **original, unmodified AI-session export**.

Nothing else is submitted through the course channel. The Kaggle submission file belongs in your repository, not in the submission.

**The PDF must contain:**
- a link to your **GitHub repository**;
- a **screenshot** showing the returned Kaggle private score of your designated final submission (a written score alone is not sufficient);
- your method comparison, validation results, and the business and algorithmic analysis.

**The GitHub repository must contain:** your implementation code; instructions for reproducing the results (optional); and the final submission file used to obtain the reported Kaggle score.

The homework format applies otherwise unchanged: course report, and the records the existing homework guidelines require. This quiz adds no new language requirement, page limit or replacement rubric. Disclose external code, models, data and assistance as the homework instructions and competition rules require. Do not fabricate experiments, references or results.

**Gate 0 is an automatic fail of the exam.** Re-read the homework guidelines before submitting. Fabricated experiments or references, a claim of verification you did not perform, and an AI-session export that does not correspond to the work all trigger it.

## 10. Grading — 100 points

### A. Kaggle private-score ranking within the exam cohort — 50 points

Students are ranked **within this exam cohort**, never against the original Kaggle participants, by the **returned private score** of their designated submission. The leaderboard is frozen, so position on it carries no information; only the returned score does. The metric is MAP@{K} with **higher is better**, so 0.00000 is the worst possible score rather than a neutral one — the template's own 0.00000 makes that concrete.

| Cohort band (by returned private score, best first) | Points |
| --- | ---: |
| Top 20% | 50 |
| Next 20% | 40 |
| Next 20% | 30 |
| Next 20% | 20 |
| Bottom 20% | 10 |

### B. Homework-style evaluation — 50 points

The existing homework criteria apply, scaled to 50 points, covering implementation, report, verification, business analysis and the required records. This quiz does not invent a new subscore breakdown. A high private score does not replace the requirement to implement and compare two methods, justify the strategy, and provide evidence.
