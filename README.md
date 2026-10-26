
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

## Baseline vs. Trained

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy | | | |
| Macro F1 | | | |
| F1 — `question` | | | |
| F1 — `opinion` | | | |
| F1 — `information` | | | |

**What I predicted before I looked:**

**What the gap actually means:**

## Run Log — Before

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. | | | | | |
| 2. | | | | | |
| 3. | | | | | |
| 4. | | | | | |
| 5. | | | | | |

### Confusion matrix

*Paste the notebook's Markdown confusion matrix table here during Unit 6. Do not use a screenshot.*

**My biggest off-diagonal number, and what it means:**

## Verdicts and Diagnoses

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

**Diagnoses:**

## Agreement Report

**Agreement rate:** ___ / 30 = ___%

**Disagreements:** For each disagreement, document the post, the staff label, your label, your reasoning under the staff taxonomy, and whether the issue was inconsistent application, a gap in the definitions, or a defensible ambiguous interpretation.

**What the pattern in my disagreements tells me:**

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. | | | | | |
| 2. | | | | | |
| 3. | | | | | |
| 4. | | | | | |
| 5. | | | | | |

**Did it help, and how do I know:**

## What's Still Broken

**The gap between what I meant my labels to capture and what the model learned:**

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
