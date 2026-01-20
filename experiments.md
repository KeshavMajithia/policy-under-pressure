Phase 2: Behavioral Comparison Experiments

(All agents already know how to drive)

Important rule:
No experiment is allowed to re-teach driving.
We only stress, distort, or restrict what already exists.

1️⃣ Reward Exploitation (Cheating)

Setup

Add a tiny loophole reward (e.g. +0.1 for lateral oscillation, boundary grazing, lap reset timing)

Question

Who discovers non-driving behaviors faster?

Expected

RL: precise exploit, sharp policy collapse

ES: ignores unless exploit is globally stable

Metric

Reward vs actual progress divergence

2️⃣ Environment Generalization

Setup

Train on Track A

Test on:

unseen curvature

mirrored tracks

randomized widths

Question

Who learned “driving” vs “this track”?

Expected

ES generalizes better due to invariant geometry

RL overfits curvature statistics

3️⃣ Observation Noise

Setup

Inject noise into:

lateral error

heading error

speed

Question

Who maintains control under sensor uncertainty?

Metric

Steering smoothness

Track exit rate

4️⃣ Reward Delay & Corruption

Setup

Delay reward by N steps

Randomly drop reward 30% of the time

Insight
ES doesn’t care when reward arrives — RL does.

5️⃣ Sample Efficiency

Setup

Same environment

Same architecture

Compare reward vs environment steps

Interpretation
Not “who wins” — but how expensive intelligence is.

6️⃣ Training Stability

Setup

10 random seeds each

Track collapses, regressions

Expected

RL: faster rise, occasional collapse

ES: boring, monotonic improvement

7️⃣ Behavioral Diversity

Setup

Log trajectories across seeds

Metric

Entropy of action distribution

Path variance

Interpretation
Exploration style, not performance.

8️⃣ Reward Sensitivity

Setup

Change penalty coefficients by ±10%

Question

Who breaks when the rules slightly change?

9️⃣ Alignment with True Objective

Setup

Intentionally misalign reward

Observe behavior visually

This is your “AI alignment” experiment
Very strong paper angle.

🔟 Graceful Degradation

Setup

Smaller networks

Less compute

Fewer steps

Question

Who still drives like a car instead of a glitch?

1️⃣1️⃣ Interpretability of Failure

Setup

Force failure cases

Analyze why the policy broke

ES failures are often smooth and explainable.

1️⃣2️⃣ Compute Scaling

Setup

Same wall-clock

ES with more CPUs

RL with larger batch sizes

1️⃣3️⃣ Partial Observability

Setup

Remove heading info

Remove speed

Question

Who infers missing state better?

1️⃣4️⃣ Overfitting to Physics Quirks

Setup

Slightly change friction / mass

Expected

RL breaks earlier

ES tolerates drift

1️⃣5️⃣ Natural Motion

Setup

Human evaluation or smoothness metrics

This is your visual + qualitative killer result.

4️⃣ Final verdict (important)

You are not comparing Usain Bolt vs a newborn.

You are comparing:

A sharp optimizer vs a robust optimizer
A gradient thinker vs a population thinker

And Phase 1 did exactly what it needed to:

teach both how to drive

remove survival nonsense

remove reward confusion

enforce discipline

5️⃣ What you should do next (concrete)

Do NOT code yet.

Next steps:

Freeze Phase 1 (no more tuning)

Write Phase 2 Experiment Protocols (like above)

Decide which 5–6 experiments become the paper core

Then build frontend visualizations only for those


1️⃣ Reward Exploitation (Cheating Behavior)
What you test

Introduce a reward that can be exploited:

Example: high reward for speed, weak penalty for corner deviation

What you measure

Reward ↑ vs track deviation ↑

Steering saturation

Corner cutting frequency

Expected reality

RL: exploits aggressively (rides edges, cuts corners)

ES: ignores exploit unless globally safe

Interpretation

RL finds loopholes faster
ES resists reward hacking but sacrifices performance

📌 This is not failure — it’s value alignment vs optimization power.

2️⃣ Generalization Across Environment Changes
What you change

New unseen tracks

Slightly altered curvature

Different straight/turn ratios

What you measure

Speed drop

Off-track events

Recovery time

Expected

RL: sharp performance drop, then partial recovery

ES: slower but stable immediately

📌 RL memorizes how to win
📌 ES learns how not to die

3️⃣ Robustness to Observation Noise
What you change

Gaussian noise on position/heading

Partial sensor dropout

Metrics

Steering oscillation

Lane deviation

Crash probability

Expected

RL: twitchy, oscillatory

ES: damped, smooth

📌 ES wins here — not because it’s smart, but because it’s cautious.

4️⃣ Robustness to Reward Noise & Delay
What you change

Delayed reward

Sparse rewards

Random reward masking

Expected

RL: destabilizes (credit assignment hell)

ES: largely unaffected

📌 ES doesn’t care when reward comes
📌 RL absolutely does

5️⃣ Sample Efficiency
What you measure

Reward vs environment steps

Time to first clean lap

Expected

RL dominates

ES lags badly

📌 This is not controversial.
📌 This is why RL is used in robotics, games, control.

6️⃣ Stability of Training
What you test

Multiple seeds

Long training runs

Metrics

Reward variance

Sudden collapses

Expected

RL: sharp gains, occasional collapse

ES: boring, monotonic

📌 ES’s “boring” behavior is a feature.

7️⃣ Behavioral Diversity
What you measure

Trajectory variance

Speed profiles

Steering entropy

Expected

ES explores policy space globally

RL converges to a single dominant behavior

📌 ES = population thinker
📌 RL = winner-takes-all thinker

8️⃣ Sensitivity to Reward Shaping
What you change

Slight reward coefficient tweaks

Expected

RL: behavior shifts dramatically

ES: relatively unchanged

📌 RL is fragile to reward design
📌 ES is reward-robust but conservative

9️⃣ Alignment with True Objective
True objective

“Drive cleanly, smoothly, and correctly — even if reward is imperfect”

Expected

ES behaves “correctly” even with bad rewards

RL follows reward literally

📌 This is your alignment experiment

🔟 Graceful Degradation
What you change

Smaller networks

Fewer training steps

Reduced compute

Expected

RL collapses suddenly

ES degrades smoothly

📌 ES fails gracefully
📌 RL fails catastrophically

1️⃣1️⃣ Interpretability of Failure
What you analyze

Why did the agent fail?

Expected

ES failures are simple: “too slow”, “too cautious”

RL failures are chaotic: oscillation, oversteering, reward chasing

📌 ES is easier to reason about.

1️⃣2️⃣ Compute Scaling Behavior
What you change

CPUs / parallel rollouts

Expected

ES scales linearly

RL hits diminishing returns

📌 This is ES’s biggest strength historically.

1️⃣3️⃣ Memory & Partial Observability
What you change

Remove heading

Mask future curvature

Expected

ES unaffected (trajectory-level optimization)

RL struggles (Markov violation)

📌 ES doesn’t need perfect state.

1️⃣4️⃣ Overfitting to Environment Quirks
What you test

Remove invisible shortcuts

Slight track randomization

Expected

RL overfits quirks

ES learns invariant behavior

📌 RL learns tricks
📌 ES learns principles

1️⃣5️⃣ Emergence of Natural Motion
What you observe

Smoothness

Human-like driving

Expected

ES looks human

RL looks optimal but robotic

📌 This is huge for real-world systems.