# Anomalies and distillation

How the system produces cardiovascular anomaly labels without ever asking a user to
provide one. This document describes the whole chain end to end: how synthetic anomalies
are made, how the autoencoder becomes a detector, how its one free parameter is
calibrated, how it is evaluated, and how it hands labels to the classifier that actually
runs on the embedded device.

---

## 1. Why any of this exists

The thesis builds a federated learning architecture. Everything in this document is
downstream of that: **two case studies that put the architecture to use**, chosen because
each one exercises a part of the system that would otherwise only be asserted to work.

- **FPR calibration** — how a reconstruction-error autoencoder becomes an actual detector.
  The model produces a score per window; turning that score into a decision needs a
  threshold, and the case study is how that threshold gets chosen without any client
  needing labels or anyone else's data. The server calibrates one number — the **expected
  false-positive rate** — and each client turns it into its own threshold locally.
- **Label distillation** — how a model with comparable behaviour can be made to run on a
  light wearable. The autoencoder is accurate but far too heavy for an ESP32-class device,
  and it is unsupervised, which is what lets it exist without labels in the first place.
  So it acts as a **teacher**: it emits soft labels, and those labels train a **student** —
  a small supervised multilayer perceptron (`FeatureMLP`), much smaller and cheaper than
  the teacher, and the model that actually ships to the device.

The two chain together. Calibration is what makes the teacher's labels meaningful, and
distillation is what turns them into a deployable model. Read end to end, they demonstrate
the claim that matters: **an unsupervised teacher can train the on-device model with no
ground truth anywhere on the client.**

### Why there are no real labels

The system trains on photoplethysmography (PPG) recordings from people going about their
day. Nobody annotates those recordings: there is no cardiologist marking the seconds where
something went wrong, and the privacy premise of the architecture means the raw signal
never leaves the user's devices anyway, so nobody *could* annotate it centrally. That rules
out ordinary supervised training for the on-device model, and is the reason the teacher has
to be unsupervised.

The teacher needs labels for exactly one thing: to check whether it works. Those labels
are **synthetic** — anomalies injected deliberately into otherwise-clean recordings, so
the ground truth is known by construction. The synthetic labels are used for evaluation
and for picking one global constant; they are never needed on a real client.

> **Scope.** The point of the thesis is the federated architecture, not cardiology. The
> synthetic anomalies are waveform-level corruptions chosen to be visible and
> reproducible, not validated clinical arrhythmia models, and detection quality is a
> secondary result. What matters is that the label pipeline runs end to end without
> ground truth on the client.

---

## 2. The three datasets

Everything starts from PPG-DaLiA: 15 subjects (S1–S15), wrist BVP at 64 Hz and
accelerometer at 32 Hz. The signal is cut into non-overlapping **8-second windows** (512
BVP samples), which is the unit of every label, score and decision in this document.

From each subject's clean recording, three datasets are derived:

| Dataset | What it is | Used for |
|---|---|---|
| **clean-signals** | The original recording, untouched | Training the teacher; each subject's own reference for thresholding |
| **anomalous-signals** | One dataset *per anomaly kind*, that kind applied to every window | Per-kind evaluation in isolation |
| **mixed-signals** | A realistic blend: ~50% of windows corrupted, kinds drawn at random | Calibration, evaluation, distillation, student training |

Anomalies are injected into **BVP only** — the accelerometer channel is never corrupted.
ACC exists in the pipeline for one purpose: it feeds the hand-crafted feature vector the
classifier consumes. No model takes ACC as a signal input.

### The five anomaly kinds

Each is a transformation of a span of clean BVP:

- **blowup** — amplitude scaled 2–4× around the local mean. A gross integrity fault.
- **noise** — a band-limited wavy interference burst added on top.
- **tachy** — the waveform resampled 0.5–0.65× and tiled, i.e. sped up.
- **brady** — the waveform stretched 1.5–1.65×, i.e. slowed down.
- **afib** — an irregularly-irregular rhythm, produced by warping time along a jittered
  but monotonic speed curve (~1 control point per second, speed 0.3–1.7×).

Corruptions are applied over **whole-window-aligned spans** of 8–30 consecutive windows,
never partially overlapping a window. That alignment is what lets a single binary label
describe a window: every window is entirely clean or entirely anomalous, and the labels
line up 1:1 with the feature grid and the score grid.

### A caveat that matters later

The mixed set is **~50% anomalous by construction**. That is a deliberate convenience for
building a balanced evaluation set, and it is nothing like a real deployment, where
anomalies would be rare. Section 4 explains why this does not compromise the calibration —
and where it does still leak into the reported numbers.

---

## 3. The detector

The teacher is a convolutional autoencoder trained on clean BVP windows only. It
compresses each window to a small code and reconstructs it. Trained exclusively on normal
waveforms, it becomes good at normal waveforms and only those.

The **anomaly score of a window is its reconstruction error** — the mean squared
difference between the window and the autoencoder's reconstruction of it. Nothing else.
A window that looks like the training distribution is reproduced closely and scores low;
one that does not is reproduced badly and scores high.

Turning a score into a decision needs a threshold, and this is where the design gets its
one interesting idea.

### Thresholds are per-subject

Reconstruction error is not comparable across people. Skin tone, sensor fit, wrist size
and motion habits all shift the scale of a subject's errors. One global threshold would be
dominated by whoever happens to be noisiest: they would trip it constantly while a
clean-signal subject would never trip it at all.

So each subject gets **their own threshold, derived from their own clean baseline**: the
threshold is the `1 − f` quantile of that subject's clean-window scores, where `f` is a
single global constant — the **expected FPR**, the false-alarm rate on clean signal the
system is calibrated to produce.

This is a shape a real client can actually execute. **The server calibrates the expected
FPR; the client computes the threshold.** A client has its own clean baseline — it does not
need anyone else's data, and it does not need any labels. The server ships one number, `f`;
the client turns it into a threshold locally.

### Why the expected FPR is the parameter

`f` is not a proxy for the false-alarm rate, or a knob that happens to correlate with it.
It *is* the false-alarm rate, stated directly: by definition, exactly a fraction `f` of a
subject's clean scores lie above their `1 − f` quantile. Choosing `f` is choosing how often
the detector fires on clean signal, and the threshold is what a client derives from it —
never the other way around.

It is *expected* rather than guaranteed because the identity is exact only on the subjects
whose own clean windows set the thresholds. On a subject the system has never seen, the
threshold comes from that subject's own baseline too, so the rate lands near `f` but is not
pinned to it. The rate actually measured on a given set is its **empirical FPR**; the whole
point of calibrating `f` is that the two agree.

### The absolute error scale carries no information

Because the threshold is a quantile
of the same distribution the scores come from, applying any monotonically increasing
transformation to every error — multiply by 10, take the square root, add a constant —
moves the threshold identically and flags exactly the same windows. A model whose clean
reconstruction error drops by 50× has not, on that basis alone, become a better detector.

What *does* matter is the **overlap between the clean and anomalous score
distributions**. Detection improves only when anomalous errors move up *relative to* clean
errors. Any headline about the error floor dropping is, by itself, not a detection result.

---

## 4. Calibration: choosing the expected FPR

This is the first case study (§1). The expected FPR is the only thing calibration picks —
one number for the whole system, and the only number the server has to ship for a client to
build a detector.

Because `f` *is* the false-positive rate (§3), the trade-off collapses into a single
dimension. Raise `f` and the threshold drops: more anomalies caught, more clean windows
wrongly flagged. Lower `f` and the reverse. The whole calibration is therefore a 1-D sweep:
try a grid of expected FPRs, measure recall at each, pick the best.

"Best" is defined by **Youden's J**:

```
J(f) = recall(f) − FPR(f) = recall(f) − f
```

and the selected value is the one maximizing it. The sweep is recorded in full — recall,
precision, F1, empirical clean FPR and J at every level — and plotted, so the chosen
operating point is auditable rather than asserted.

### Why Youden's J and not F1

This is the part most worth understanding, because F1 is the more familiar metric and it
is the wrong tool here.

Write `π` for the **prevalence**: the fraction of windows that are truly anomalous. Then:

- **Recall** = (anomalous windows caught) / (anomalous windows). Computed *only over
  anomalous windows*. Clean windows do not appear in the formula.
- **FPR** = (clean windows flagged) / (clean windows). Computed *only over clean windows*.
  Anomalous windows do not appear in the formula.

Each is conditioned on a single class. Change the *mix* — duplicate every clean window,
say, dropping `π` from 0.5 to 0.33 — and neither number moves: the same anomalous windows
are caught at the same rate, and the duplicated clean windows are flagged at the same
rate. `J = recall − FPR` is a difference of two class-conditional rates, so it inherits
that invariance: **J does not depend on prevalence.**

Precision does not have this property, because it mixes the two classes — its numerator
comes from the anomalous pool and its denominator draws from both:

```
precision = TP / (TP + FP) = TPR·π / (TPR·π + FPR·(1−π))
```

`π` appears explicitly. Duplicate the clean windows and false positives double while true
positives do not, so precision falls. **F1 is built from precision and inherits that
dependence on `π`.** (So does accuracy.)

Now the argument for calibration. Calibration picks an operating point that will be used
**at deployment**, on data whose prevalence is unknown, differs from the calibration set's,
varies between subjects, and drifts over time. If the threshold is chosen to maximize F1
on the calibration set, what has been chosen is the argmax of a function of the calibration
set's `π` — move to a different `π` and the argmax moves with it. The choice does not
transfer. J's argmax depends only on the two class-conditional score distributions, so it
is the same operating point whatever `π` turns out to be.

That is the reason, and note that it does not depend on 50% being an unrealistic
prevalence. It would hold at any prevalence, because the problem is that deployment
prevalence is *unknown*, not that this particular number is *wrong*. Threshold selection on
the ROC curve — of which Youden's J is the simplest criterion — is standard practice for
exactly this reason.

### The honest corollary

Because F1 is not the selection criterion, **the reported F1 is deliberately not the
F1-optimal value**. On this dataset F1 keeps climbing as `f` grows, and is maximized where
the detector fires on half of all clean windows — an operating point no deployment would
tolerate. That is the symptom; prevalence-dependence is the disease.

And the bias runs in the *unflattering* direction for the numbers that do get reported.
From the formula above, precision falls as `π` falls. Real prevalence would be far below
50%, so the reported precision is **better than a deployment would see**, not worse. It is
an artifact of a balanced evaluation set, and should be read as such.

The prevalence-independent numbers — per-kind recall, empirical clean FPR, and the J sweep — are the
ones that mean something on their own. Precision and F1 are reported for completeness,
against a stated prevalence, and no comparison rests on them.

---

## 5. Evaluation

Evaluation is the diagnostic step, and unlike distillation it is allowed to look at
anything: ground-truth labels, the per-kind datasets, every subject.

Three things are measured, all at the calibrated expected FPR:

**Detector vs. ground truth on the mixed set** — precision, recall, F1, accuracy, and the
empirical clean false-positive rate.

**Recall per anomaly kind**, scored on the per-kind datasets where every window is an
example of that one kind. This is the informative breakdown: an aggregate recall hides
which faults the detector actually catches. Since the clean FPR equals `f`, the expected
FPR also serves as the chance floor — a kind whose recall sits near `f` is not being
detected at all, and a kind *below* `f` is being flagged less often than clean signal.

### A structural blind spot

Reconstruction error detects anomalies that make a waveform **harder** to reconstruct.
Bradycardia makes it **easier** — a slowed waveform is smoother and more predictable, so
its error moves in the wrong direction and the method is blind to it by construction. The
detector flags bradycardia *less* often than it flags clean signal.

Band-limited noise is nearly as bad: the injected interference is smooth and low-frequency,
so the autoencoder reconstructs it comfortably.

Together these two kinds are 2/5 of the injected anomalies, which caps achievable recall
and is the reason recall sits low while precision sits high. This is a property of
reconstruction scoring, not a tuning failure — it is declared, not disguised.

---

## 6. Distillation: from teacher to student

This is the second case study (§1). The autoencoder detects well, but it is unsupervised
and far too heavy for the embedded device — the two properties are linked, since being
unsupervised is exactly what lets it exist without labels. Distillation resolves that: the
teacher emits soft labels, and those labels train a small **supervised** classifier
(`FeatureMLP`) that fits on the wearable. The student is not a compressed copy of the
teacher — it is a different, cheaper model consuming hand-crafted features, trained to
agree with the teacher's judgement.

This is also the one step deliberately constrained to what a **real client could do**.

The client-facing label step touches only:

- its own clean-signal baseline (to set its threshold),
- the signal it wants labeled,
- the features it computed on-device,
- and the one global expected FPR the server sent.

It never touches ground-truth labels, the per-kind datasets, or any other subject's data.
That restriction is the point: it demonstrates the pipeline works **without ground truth on
the client**, which is what makes the privacy premise viable.

### Soft labels

The label is not a bare 0/1. Each window gets a **soft label in [0, 1]**:
`sigmoid((error − threshold) / s)`, where `s` is the standard deviation of that subject's
own clean-window scores. The signed distance to the threshold says which side of the
decision the window falls on and how far; dividing by `s` puts that distance on the
subject's own error scale — reconstruction-error magnitude varies between people (§4), so
without it the same logit would mean different things for different subjects. The sigmoid
then ramps smoothly: a window right at the threshold gets `0.5`, so `label > 0.5`
reproduces the hard decision, while windows well beyond it saturate toward 1 and windows
well below toward 0. That gradation is information the student can learn from and a hard
label would throw away.

The result is a label set shaped exactly like the real feature dataset, so the student's
training run consumes it by pointing at a different directory and changing nothing else.
The student is then compared against the same student trained on the direct synthetic
labels — that comparison is the end-to-end claim: **unsupervised teacher → soft labels →
student, with no ground truth anywhere on the client.**

---

## 7. The chain at a glance

```
clean BVP ──> autoencoder (teacher)          trained on clean signal only, no labels
    │
    ├──> reconstruction error per window     the anomaly score
    │
    ├──> per-subject threshold               1-f quantile of that subject's clean scores
    │         ▲                              (computed client-side, from its own baseline)
    │         │
    │   expected FPR f ── calibration        1-D sweep, maximize J = recall(f) - f
    │                                        (server-side; J is prevalence-independent, F1 is not)
    │
    ├──> evaluation                          per-kind recall, empirical clean FPR;
    │                                        uses ground truth — diagnostic only
    │
    └──> soft labels ──> FeatureMLP (student) sigmoid((error-threshold)/clean-error std);
                                              client-side data only
```

Implementation lives in `backend/scripts/figures/` (`calibrate_fpr`, `anomaly_detection`,
`knowledge_distillation`) over the shared scoring helpers in
`backend/scripts/common/scoring.py`; synthetic-anomaly generation is in
`backend/ml/preprocessing.py`. See [model-types.md](model-types.md) for the model
architectures and `backend/PLOTS.md` for the commands that produce each result.
