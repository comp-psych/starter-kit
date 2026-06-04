# %% [markdown]
# # Tutorial 01: The Rescorla-Wagner Model
#
# **Computational Psychiatry Starter Kit** — PRAXIS at USC
#
# ---
#
# ## What you'll learn
#
# 1. What *prediction errors* are and why they're central to computational psychiatry
# 2. How to implement the Rescorla-Wagner (RW) learning rule from scratch
# 3. How to simulate a Pavlovian conditioning experiment
# 4. How to fit the model's learning rate to observed data using maximum likelihood estimation
# 5. Why this matters for understanding depression
#
# ## Background
#
# The Rescorla-Wagner model (1972) is the foundation of modern learning theory.
# It formalizes a simple but powerful idea: **we learn from the difference between
# what we expected and what actually happened**. That difference is called a
# *prediction error* (PE).
#
# $$\delta_t = r_t - V_t$$
#
# where $\delta_t$ is the prediction error on trial $t$, $r_t$ is the outcome
# (e.g., reward = 1, no reward = 0), and $V_t$ is the current expected value
# (association strength).
#
# The value is then updated:
#
# $$V_{t+1} = V_t + \alpha \cdot \delta_t$$
#
# where $\alpha \in [0, 1]$ is the **learning rate** — how much we adjust our
# expectations in response to each prediction error.
#
# ### Why this matters for psychiatry
#
# Prediction errors are computed by dopamine neurons in the midbrain (Schultz
# et al., 1997). In depression, these signals may be blunted — patients fail to
# update their expectations when good things happen, maintaining a pessimistic
# world-model. Computational models let us *quantify* this: a depressed patient
# might show a lower learning rate for positive outcomes, or attenuated prediction
# error signals in neuroimaging.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# %% [markdown]
# ## Part 1: Implement the Rescorla-Wagner update rule
#
# Let's start by writing the core learning rule as a Python function.

# %%
def rescorla_wagner(outcomes, alpha, V0=0.0):
    """
    Run the Rescorla-Wagner model on a sequence of outcomes.

    Parameters
    ----------
    outcomes : array-like
        Sequence of outcomes (e.g., 1 = reward, 0 = no reward).
    alpha : float
        Learning rate, must be in [0, 1].
    V0 : float
        Initial value (association strength). Default = 0.

    Returns
    -------
    values : np.ndarray
        Value (association strength) *before* each trial.
    prediction_errors : np.ndarray
        Prediction error on each trial.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    n_trials = len(outcomes)
    values = np.zeros(n_trials)
    prediction_errors = np.zeros(n_trials)

    V = V0
    for t in range(n_trials):
        values[t] = V
        prediction_errors[t] = outcomes[t] - V   # PE = outcome - expectation
        V = V + alpha * prediction_errors[t]      # Update rule

    return values, prediction_errors

# %% [markdown]
# ## Part 2: Simulate a conditioning experiment
#
# We'll simulate a classic Pavlovian conditioning paradigm:
# - A conditioned stimulus (CS) is presented on every trial
# - An unconditioned stimulus (US, e.g., food) is delivered on 80% of trials
# - We track how the association between CS and US develops over time

# %%
# Simulation parameters
n_trials = 100
p_reward = 0.8       # Probability of US given CS
alpha_true = 0.1     # True learning rate we'll try to recover

# Generate outcomes
outcomes = np.random.binomial(1, p_reward, size=n_trials)

# Run the model
values, pes = rescorla_wagner(outcomes, alpha=alpha_true)

print(f"True learning rate: {alpha_true}")
print(f"Final value (association strength): {values[-1]:.3f}")
print(f"Theoretical asymptote (= p_reward): {p_reward}")

# %% [markdown]
# ## Part 3: Visualize the learning curve

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Top panel: Value (association strength) over trials
axes[0].plot(values, color='#2196F3', linewidth=2, label='Learned value $V_t$')
axes[0].axhline(p_reward, color='#E91E63', linestyle='--', linewidth=1.5,
                label=f'True P(reward) = {p_reward}')
axes[0].fill_between(range(n_trials), values, alpha=0.15, color='#2196F3')
axes[0].set_ylabel('Association Strength ($V_t$)', fontsize=12)
axes[0].set_title('Rescorla-Wagner Learning Curve', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_ylim(-0.05, 1.05)

# Bottom panel: Prediction errors
colors = ['#4CAF50' if pe > 0 else '#F44336' for pe in pes]
axes[1].bar(range(n_trials), pes, color=colors, alpha=0.7, width=1.0)
axes[1].axhline(0, color='gray', linewidth=0.8)
axes[1].set_xlabel('Trial', fontsize=12)
axes[1].set_ylabel('Prediction Error ($\\delta_t$)', fontsize=12)
axes[1].set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../data/01_learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **What to notice:**
#
# - The value $V_t$ gradually approaches the true reward probability (0.8)
# - Prediction errors are large early on (lots of surprise) and shrink as
#   learning progresses
# - Green bars = positive PEs (better than expected), red = negative PEs
#   (worse than expected)
#
# In depression, you might imagine the green bars being smaller — the brain
# doesn't fully register when things go *better* than expected.

# %% [markdown]
# ## Part 4: Load pre-generated data and fit the model
#
# Now let's fit the learning rate $\alpha$ to data using maximum likelihood
# estimation. We'll use the pre-generated dataset that ships with this kit.

# %%
# Load the pre-generated data
data = pd.read_csv('../data/01_conditioning_data.csv', comment='#')
print(f"Loaded {len(data)} trials")
data.head(10)

# %% [markdown]
# ### Defining the negative log-likelihood
#
# To fit the model, we need a measure of how well a given $\alpha$ explains
# the observed data. We use the negative log-likelihood (NLL):
#
# For each trial, the model's predicted probability of reward is $V_t$ (clipped
# to avoid log(0)). The likelihood of the observed outcome under a Bernoulli
# distribution is:
#
# $$\mathcal{L}_t = V_t^{r_t} \cdot (1 - V_t)^{1 - r_t}$$
#
# We minimize $-\sum_t \log \mathcal{L}_t$ over $\alpha$.

# %%
def neg_log_likelihood(params, outcomes):
    """
    Compute negative log-likelihood of the RW model given outcomes.

    Parameters
    ----------
    params : array-like
        [alpha] — the learning rate.
    outcomes : array-like
        Observed outcomes (0 or 1).

    Returns
    -------
    nll : float
        Negative log-likelihood (lower = better fit).
    """
    alpha = params[0]

    # Enforce bounds
    if alpha <= 0 or alpha >= 1:
        return 1e10

    values, _ = rescorla_wagner(outcomes, alpha)

    # Clip values to avoid log(0)
    eps = 1e-8
    V_clipped = np.clip(values, eps, 1 - eps)

    # Bernoulli log-likelihood
    log_lik = outcomes * np.log(V_clipped) + (1 - outcomes) * np.log(1 - V_clipped)

    return -np.sum(log_lik)

# %%
# Fit the model using scipy.optimize.minimize
outcomes_data = data['us'].values

result = minimize(
    neg_log_likelihood,
    x0=[0.5],                          # Initial guess
    args=(outcomes_data,),
    method='Nelder-Mead',
    options={'xatol': 1e-6, 'fatol': 1e-6}
)

alpha_fit = result.x[0]
print(f"Fitted learning rate (alpha): {alpha_fit:.4f}")
print(f"Negative log-likelihood: {result.fun:.2f}")
print(f"Optimization converged: {result.success}")

# %% [markdown]
# ## Part 5: Compare fitted model to data

# %%
# Run model with fitted alpha
values_fit, pes_fit = rescorla_wagner(outcomes_data, alpha=alpha_fit)

fig, ax = plt.subplots(figsize=(10, 5))

# Plot the learned values
ax.plot(values_fit, color='#2196F3', linewidth=2.5, label=f'Fitted RW ($\\alpha$={alpha_fit:.3f})')
ax.scatter(range(len(outcomes_data)), outcomes_data, color='#FF9800',
           alpha=0.3, s=15, label='Observed outcomes', zorder=1)

# Running average of outcomes for comparison
window = 10
running_avg = pd.Series(outcomes_data).rolling(window, min_periods=1).mean()
ax.plot(running_avg, color='#E91E63', linewidth=1.5, linestyle='--',
        label=f'Running avg (window={window})', alpha=0.8)

ax.set_xlabel('Trial', fontsize=12)
ax.set_ylabel('Value / Outcome', fontsize=12)
ax.set_title('Model Fit vs. Observed Data', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('../data/01_model_fit.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Part 6: Explore how learning rate affects behavior
#
# Different values of $\alpha$ produce very different learning curves.
# A high $\alpha$ means fast, noisy learning; a low $\alpha$ means slow,
# stable learning.

# %%
fig, ax = plt.subplots(figsize=(10, 5))

alphas = [0.01, 0.05, 0.1, 0.3, 0.7]
colors_alpha = ['#1A237E', '#1565C0', '#2196F3', '#64B5F6', '#BBDEFB']

for alpha, color in zip(alphas, colors_alpha):
    v, _ = rescorla_wagner(outcomes_data, alpha=alpha)
    ax.plot(v, linewidth=2, color=color, label=f'$\\alpha$ = {alpha}')

ax.axhline(0.8, color='#E91E63', linestyle='--', alpha=0.5, label='P(reward)')
ax.set_xlabel('Trial', fontsize=12)
ax.set_ylabel('Learned Value ($V_t$)', fontsize=12)
ax.set_title('Effect of Learning Rate on Value Tracking', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, ncol=2)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('../data/01_alpha_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Clinical Relevance: Prediction Errors and Depression
#
# ### The dopamine hypothesis of depression
#
# Prediction errors are encoded by dopamine neurons in the ventral tegmental
# area (VTA) and substantia nigra. When an outcome is *better* than expected,
# dopamine neurons fire more (positive PE). When it's *worse* than expected,
# they fire less (negative PE).
#
# In major depressive disorder (MDD), there's evidence that:
#
# 1. **Blunted positive PEs**: Patients show reduced neural response to
#    unexpected rewards (Gradin et al., 2011; Kumar et al., 2008). This maps
#    to a lower effective learning rate for positive outcomes.
#
# 2. **Amplified negative PEs**: Some studies show enhanced sensitivity to
#    negative prediction errors, meaning patients over-learn from bad outcomes.
#
# 3. **Anhedonia as a learning deficit**: The inability to experience pleasure
#    may partly reflect a failure to *learn* from rewarding experiences —
#    not just a failure to feel them in the moment.
#
# ### What the model gives us
#
# By fitting the Rescorla-Wagner model (or its extensions, like the dual
# learning rate model in Tutorial 02), we can extract a *quantitative* measure
# of learning rate from behavioral data. This gives us:
#
# - A **biomarker** that can be compared across patient groups
# - A **mechanistic explanation** for symptoms (not just "they feel sad")
# - A way to track **treatment response** (does learning rate improve with SSRIs?)
#
# This is the core value proposition of computational psychiatry: turning
# subjective symptoms into measurable computational quantities.

# %% [markdown]
# ## Exercises
#
# 1. **Extinction**: After trial 100, switch to 0% reward (CS presented but
#    no US). How does the value change? How many trials does it take to
#    reach V < 0.1? How does $\alpha$ affect extinction speed?
#
# 2. **Parameter recovery**: Generate data with a known $\alpha$, then fit
#    the model. Try this 100 times with different $\alpha$ values. Plot
#    true vs. recovered $\alpha$. When does recovery break down?
#
# 3. **Model comparison**: What if the true learning rule is *not* RW?
#    Generate data from a Pearce-Hall model (where $\alpha$ changes over
#    time based on unsigned PEs) and try to fit RW. How does the fit compare?
#
# 4. **Clinical simulation**: Simulate two "patients" — one with $\alpha = 0.1$
#    (healthy) and one with $\alpha = 0.02$ (depressed). After 50 trials of
#    80% reward, switch to 100% reward. How long does each patient take to
#    notice the improvement? What does this imply about therapy response?

# %% [markdown]
# ---
#
# **Next tutorial**: [02 — Dual Learning Rate Model](./02_dual_learning_rate.py)
# introduces separate learning rates for positive and negative prediction
# errors, directly modeling anhedonia.
#
# ---
# *Computational Psychiatry Starter Kit — PRAXIS at USC — Peter Zhou*
