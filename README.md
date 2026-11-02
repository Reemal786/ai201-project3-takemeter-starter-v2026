
# TakeMeter

> **New to this repository?** Start with [RUNNING.md](RUNNING.md) for environment setup, notebook instructions, baseline evaluation, and troubleshooting.
>
> After `python test.py` passes, open `takemeter.ipynb` in this repository with the project's `.venv` kernel. Training and evaluation run locally and write output files alongside the notebook.

---

<!-- UNIT 5 — THE BUILD -->

## What This Does

This project develops a text classification model that categorizes posts from the r/gaming subreddit based on their primary purpose. The classifier distinguishes between posts seeking information (`question`), expressing personal judgments or preferences (`opinion`), and sharing gaming-related news or factual information (`information`). By analyzing the language used in each post, the model aims to identify the author's main communicative intent. The project evaluates how reliably these categories can be distinguished using a labeled dataset of gaming discussions.

## Label Taxonomy

### `question`

**Definition:** Primarily requests advice, recommendations, explanations, or information from other users.

**Example 1:** [Looking for games with ancient civilisations, lost history, mysterious ruins and secrets to uncover](https://www.reddit.com/r/gaming/comments/1x0l6kw/) — The author describes their gaming preferences and asks for recommendations matching a specific atmosphere.

**Example 2:** [Best Website for Gaming News?](https://www.reddit.com/r/gaming/comments/1dq0fzn/) — The author seeks recommendations for reliable gaming news websites.

### `opinion`

**Definition:** Primarily expresses a personal judgment, preference, criticism, or argument about a gaming-related topic.

**Example 1:** [Have gaming discussions gotten more negative in recent years?](https://www.reddit.com/r/gaming/comments/1gifhc5/) — The author argues that gaming discussions have become overly negative.

**Example 2:** [First Light is a really great game. I also can't believe it was released in the state it's in.](https://www.reddit.com/r/gaming/comments/1x0i46u/) — The author praises the game's presentation while criticizing its technical problems.

### `information`

**Definition:** Primarily shares gaming news, updates, announcements, or factual information without making personal judgment the main focus.

**Example 1:** [Skate has been de-live serviced by ReSkate](https://www.reddit.com/r/gaming/comments/1x0iz03/) — The post shares news about a modding platform and its developers' response to potential legal action.

**Example 2:** Cyberpunk 2077 expansions are still in development, CD Projekt confirms — The post reports an official update from CD Projekt about planned expansions and downloadable content for Cyberpunk 2077, making its primary purpose informational.

### The hardest boundary

**Which two labels:** `opinion` and `information`.

**Decision rule:** If a post primarily reports gaming news, announcements, or verifiable information, classify it as `information`, even when it includes minor commentary. If the post primarily evaluates, praises, criticizes, or argues about a gaming-related topic, classify it as `opinion`, even when factual details are used as supporting evidence.

## The Dataset

**Where the posts came from:** I collected 200 gaming-related Reddit posts from the Hugging Face dataset [`sentence-transformers/reddit-title-body`](https://huggingface.co/datasets/sentence-transformers/reddit-title-body), using posts from r/gaming. The source corpus contains Reddit titles and bodies; I combined the relevant text into one `text` field for classification. My collection script is `collect_posts.py`.

**How I labeled them:** I used rule-based pre-labeling to create initial labels and AI-assisted suggestions to identify cases needing review. I examined the suggested labels against my taxonomy, corrected classifications based on each post's primary purpose, and finalized the labels in `labels.csv`. For the 20 posts I specifically reviewed with AI suggestions, I evaluated the proposed labels and made the final decisions rather than treating the suggestions as automatically correct. I also checked the finished dataset for missing text, missing labels and notes, duplicate posts, invalid labels, and empty fields; all checks passed. AI assistance is disclosed in more detail below.

**Counts per label:**

| Label | Count | Share |
|---|---:|---:|
| `question` | 128 | 64% |
| `opinion` | 60 | 30% |
| `information` | 12 | 6% |
| **Total** | **200** | **100%** |

The largest category is below the assignment's 70% limit. The small `information` category is an important limitation for model training and evaluation.

### Three hard cases

**1. “What superhero game would you love to see”**

- **Could have been:** `question` or `opinion`.
- **I chose:** `question`. The body includes the author's preferences about superhero games, but the central purpose is to invite other users to name the games they want.

**2. “I want to know your thoughts on games with Engagement/Skill based matchmaking”**

- **Could have been:** `question` or `opinion`.
- **I chose:** `opinion`. Despite the title asking for thoughts, the author develops a clear argument against skill-based matchmaking; the argument is the main point of the post.

**3. “Childhood story! (first post here)”**

- **Could have been:** `opinion` or `information`.
- **I chose:** `opinion`. The author recounts a personal gaming experience and emphasizes its emotional significance rather than primarily reporting gaming news or a general factual update. This is an imperfect fit within a three-label taxonomy, but it is closer to personal expression than to the `information` definition.

## The Training Run

**Base model:** `distilbert-base-uncased`, fine-tuned for three-class sequence classification.

**Settings:**

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate | `2e-5` |
| Batch size | 16 |
| Maximum sequence length | 128 tokens |
| Random seed | 42 |
| Device | CPU |
| PyTorch version | `2.14.1+cpu` |

**Anything I changed from the defaults, and why:** I replaced the notebook's placeholder label names with `question`, `opinion`, and `information` to match my dataset. I retained the supplied base model and default training hyperparameters for this first run.

**Split sizes:** Stratified train/validation/test split (approximately 70%/15%/15%).

| Label | Train | Validation | Test |
|---|---:|---:|---:|
| `question` | 89 | 20 | 19 |
| `opinion` | 42 | 9 | 9 |
| `information` | 8 | 2 | 2 |
| **Total** | **139** | **31** | **30** |

Only two `information` posts were in the test split, so performance metrics for that category are especially sensitive to individual predictions.

**Initial held-out test results (seed 42):**

| Measure | Result |
|---|---:|
| Accuracy | 63.3% |
| Macro F1 | 0.259 |
| `question` F1 | 0.776 |
| `opinion` F1 | 0.000 |
| `information` F1 | 0.000 |

The model appears to have predicted `question` for all 30 test posts: the accuracy equals the fraction of test posts that are questions (19/30), and the other two categories have F1 scores of zero. This is a limitation of the initial model, not evidence that it can reliably distinguish all three intents. The evaluation outputs are saved in `results.json` and `test_split.csv`.

## How I Used AI

**Moment 1 — Reviewing ambiguous labels**

- **What I asked for:** Help assessing posts whose initial labels might not match the definitions of `question`, `opinion`, and `information`.
- **What came back:** Suggested labels and explanations based on whether a post primarily requested a response, expressed a judgment, or reported facts.
- **What I changed:** I compared suggestions against the actual post text and taxonomy and corrected misclassified entries in `labels.csv`. I did not automatically accept every suggestion; for example, I kept the matchmaking post as `opinion` despite an AI suggestion of `question`.

**Moment 2 — Checking dataset quality and preparing evaluation**

- **What I asked for:** Help checking dataset completeness, label distribution, and the notebook's training and evaluation workflow.
- **What came back:** Commands to check for missing values, duplicates, invalid labels, and empty fields, along with guidance on the notebook's setup, split, training, and results sections.
- **What I changed:** I ran the validation checks locally, configured the notebook to use my actual three labels, and executed training and evaluation. The checks found no structural problems, and the model's weak performance on `opinion` and `information` remained visible in the reported results.

**Pre-labeling disclosure:** The initial dataset labels were produced with rule-based automation, and AI-generated label suggestions were used during review. I reviewed and corrected labels using my taxonomy, including 20 posts for which I assessed AI suggestions and made the final labeling decisions. The dataset was not independently hand-labeled from scratch, and I am not presenting AI-suggested labels as unaided human annotations.

---

<!-- UNIT 6 — THE TEST. Leave these sections for Unit 6. -->

# Unit 6 — The Test

## Baseline vs. Trained

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy | 0.600 | 0.633 | +0.033 |
| Macro F1 | 0.414 | 0.259 | -0.155 |
| F1 — `question` | 0.643 | 0.776 | +0.133 |
| F1 — `opinion` | 0.600 | 0.000 | -0.600 |
| F1 — `information` | 0.000 | 0.000 | 0.000 |

**What I predicted before I looked:**

I expected my fine-tuned model to outperform the general baseline in overall accuracy because it was trained specifically on r/gaming posts and my three label definitions, although I anticipated difficulty identifying the underrepresented information category.

**What the gap actually means:**

Fine-tuning increased overall accuracy by 3.3 percentage points, but reduced macro F1 from 0.414 to 0.259. The trained model performed better on the majority `question` category but failed to correctly identify `opinion` and `information` posts.

This suggests that the imbalanced training dataset may have caused the model to favor the majority class rather than learning to distinguish all three categories effectively. The baseline achieved better balanced performance despite having slightly lower overall accuracy.

## Run Log — Before

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.65 | 0.633 | 0.633 | 0.633 | NOT MET |
| 2. Macro F1 | ≥ 0.60 | 0.259 | 0.259 | 0.259 | NOT MET |
| 3. Per-label F1 | ≥ 0.50 for every label | Not met | Not met | Not met | NOT MET |
| 4. Dataset balance | No label exceeds 50% | 64% `question` | 64% `question` | 64% `question` | NOT MET |
| 5. Accuracy stability | Spread ≤ 0.10 | 0.633 | 0.633 | 0.633 | MET |

The accuracy spread across all three seeds was 0.000, indicating consistent results across the tested random seeds.

### Confusion matrix

Rows represent actual labels, and columns represent predicted labels.

| Actual / Predicted | `question` | `opinion` | `information` |
|---|---|---|---|
| `question` | 19 | 0 | 0 |
| `opinion` | 9 | 0 | 0 |
| `information` | 2 | 0 | 0 |

**My biggest off-diagonal number, and what it means:**

The largest off-diagonal value was 9, representing `opinion` posts incorrectly classified as `question`. The model predicted `question` for every test example, including all 9 `opinion` posts and both `information` posts.

This indicates that the model learned to favor the majority class instead of reliably distinguishing between the three categories.

## Verdicts and Diagnoses

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 | Overall accuracy | ≥ 0.65 for every seed | NOT MET | Accuracy was 0.633 for all three seeds, below the required 0.65. |
| 2 | Macro F1 | ≥ 0.60 for every seed | NOT MET | Macro F1 was 0.259 for every seed, indicating poor performance across the three classes. |
| 3 | Per-label F1 | ≥ 0.50 for every label | NOT MET | `question` achieved 0.776, but `opinion` and `information` both scored 0.000. |
| 4 | Dataset balance | No label exceeds 50% | NOT MET | `question` accounted for 128 of 200 posts (64%), exceeding the 50% limit. |
| 5 | Accuracy stability | Spread ≤ 0.10 | MET | All three seeds produced 0.633 accuracy, giving a spread of 0.000. |

**Diagnoses:**

The most significant issue was class imbalance. Of the 200 labeled posts, 128 were `question`, 60 were `opinion`, and only 12 were `information`. This uneven distribution likely encouraged the model to prioritize the majority category.

The confusion matrix supports this diagnosis because the trained model classified every test example as `question`. Although this produced 63.3% accuracy, the model failed to recognize the other two categories.

The difference between accuracy and macro F1 also demonstrates why overall accuracy alone is insufficient for evaluating an imbalanced classification problem. The model achieved reasonable accuracy while performing poorly on two of the three labels.

Finally, the identical results across all three seeds suggest that the majority-class prediction pattern was consistent under the tested training conditions rather than an isolated result from one random seed.

## Agreement Report

**Agreement rate:** 30 / 30 = 100%

**Disagreements:**

There were no disagreements between my labels and the staff labels across the 30 reviewed posts.

**What the pattern in my disagreements tells me:**

The 100% agreement rate suggests that I consistently applied the staff's labeling definitions to the reviewed examples.

However, this agreement does not necessarily mean the full training dataset was perfectly labeled or balanced. The model's poor minority-class performance indicates that consistent labeling alone was not enough to produce effective classification across all three categories.

## The Improvement

**What I changed:**

For Milestone 5, I increased the model's training duration from 3 epochs to 5 epochs.

I kept the same DistilBERT model, dataset, label definitions, learning rate, batch size, and three random seeds. This allowed me to isolate the effect of additional training on model performance.

**Which diagnosis pointed at it:**

The initial evaluation showed that the model failed to distinguish between the three labels and predicted `question` for every test example.

I wanted to test whether additional training would give the model more opportunities to learn the differences between `question`, `opinion`, and `information` posts.

However, because the dataset remained imbalanced, additional epochs alone might not address the underlying majority-class bias.

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.65 | 0.633 | 0.633 | 0.633 | NOT MET |
| 2. Macro F1 | ≥ 0.60 | 0.259 | 0.259 | 0.259 | NOT MET |
| 3. Per-label F1 | ≥ 0.50 for every label | Not met | Not met | Not met | NOT MET |
| 4. Dataset balance | No label exceeds 50% | 64% `question` | 64% `question` | 64% `question` | NOT MET |
| 5. Accuracy stability | Spread ≤ 0.10 | 0.633 | 0.633 | 0.633 | MET |

**Did it help, and how do I know:**

Increasing training from 3 to 5 epochs did not improve the model's performance.

All three seeds produced the same accuracy of 0.633 and macro F1 of 0.259 as the original experiment. The model continued to achieve an F1 score of 0.776 for `question` and 0.000 for both `opinion` and `information`.

These results suggest that additional training duration was insufficient to overcome the model's tendency to predict the majority class.

The experiment helped narrow down the problem by showing that simply training longer was not an effective solution under the current conditions.

A more promising future improvement would be to address class imbalance directly through additional minority-class examples, balanced sampling, or class-weighted training.

## What's Still Broken

**The gap between what I meant my labels to capture and what the model learned:**

My original goal was to classify r/gaming posts into three meaningful categories: `question`, `opinion`, and `information`.

The `question` category was intended to capture posts seeking answers or assistance, `opinion` was intended to capture personal viewpoints and discussions, and `information` was intended to capture posts primarily sharing facts, news, or updates.

However, the trained model did not learn to reliably distinguish these categories. Instead, it predicted `question` for every example in the test set.

The dataset's uneven distribution likely contributed to this behavior, particularly because `information` represented only 6% of the labeled examples.

The experiment also showed that stable results across random seeds do not necessarily indicate a successful model. Although the model consistently achieved the same accuracy, its performance remained heavily biased toward one category.

Increasing training duration did not resolve this issue. Future improvements should focus on better class representation and training strategies that encourage the model to learn all three categories rather than primarily predicting the majority label.

---

<!-- SUBMISSION CHECKLIST — UNIT 5
[ ] criteria.md has five numbered measurable criteria, each with a reason
[ ] labels.csv has at least 150 rows and text/label/note columns
[ ] No label exceeds 70% of the dataset
[ ] All five Unit 5 README sections are complete
[ ] Label Taxonomy includes the hardest-boundary decision rule
[ ] The Dataset includes three hard cases
[ ] results.json and test_split.csv are committed
[ ] At least four new commits
[ ] Submit the repository URL

SUBMISSION CHECKLIST — UNIT 6
[ ] Baseline vs. Trained table and prior prediction
[ ] Three-seed Run Log — Before
[ ] Typed Markdown confusion matrix
[ ] Criterion verdicts and diagnoses
[ ] Agreement Report with adjudicated disagreements
[ ] One improvement and Run Log — After
[ ] What's Still Broken
[ ] results_three_seeds_before.json, results_three_seeds_after.json,
    baseline_results.json, and agreement_results.json committed
[ ] At least four new commits
[ ] Submit the same repository URL
-->

**How to run this project:** [RUNNING.md](RUNNING.md)
