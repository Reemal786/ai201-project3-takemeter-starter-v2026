# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---

## 1. Overall Classification Accuracy

The trained model must achieve an overall accuracy of at least **65%** on the held-out test set for each of the three training seeds.

**Why this target:**

With three categories (`question`, `opinion`, and `information`), random guessing on a balanced dataset would achieve approximately 33% accuracy. A target of 65% would demonstrate that the model can identify meaningful differences in how users communicate within r/gaming.

---

## 2. Macro F1 Score

The trained model must achieve a macro F1 score of at least **0.60** for each of the three training seeds.

**Why this target:**

The three categories may not occur equally often in r/gaming. Macro F1 evaluates each label equally, helping ensure the model performs reasonably across questions, opinions, and informational posts rather than favoring the most common category.

---

## 3. Per-Label Performance

Each label (`question`, `opinion`, and `information`) must achieve an F1 score of at least **0.50** on the held-out test set for each training seed.

**Why this target:**

Some gaming posts may contain both factual information and personal opinions, making these categories difficult to distinguish. With approximately 200 posts divided among three labels, each category will also have relatively few test examples. A minimum F1 of 0.50 accounts for these challenges while requiring the model to recognize all three categories.

---

## 4. Dataset Balance

No individual label may represent more than **50%** of the complete labeled dataset.

**Why this target:**

r/gaming includes different types of discussions, but questions and opinions may be more common than informational posts. Limiting any single category to 50% helps ensure the model receives sufficient examples of all three labels during training.

---

## 5. Training Consistency

The difference between the highest and lowest overall test accuracy across training seeds 42, 7, and 2024 must be no greater than **10 percentage points**.

**Why this target:**

Training on a relatively small dataset can produce different results depending on the random seed. A maximum variation of 10 percentage points provides a measurable standard for determining whether the classifier performs consistently across repeated training runs.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 6 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath, like this:

         ## 2. Every label performs acceptably

         The model performs well on all labels.

         **Why this target:** ...

         > **Revised in unit 6:** Every label has an F1 of at least 0.60 on
         > the held-out set.
         >
         > **Why revised:** "performs well" gave me nothing to check. I
         > couldn't produce a verdict from it at all.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "Overall accuracy of at least 0.65" → "at least 0.55", because
           0.65 turned out to be optimistic for 200 examples.

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.
     ───────────────────────────────────────────────────────────────────────── -->
