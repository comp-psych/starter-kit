# %% [markdown]
# # Tutorial 02: Dual Learning Rate Model
#
# **Computational Psychiatry Starter Kit** — PRAXIS at USC
#
# ---
#
# ## What you'll learn
#
# 1. Why a single learning rate isn't enough to capture psychiatric symptoms
# 2. How to implement a dual learning rate model ($\alpha^+$ and $\alpha^-$)
# 3. How asymmetric learning rates map to **anhedonia**
# 4. How to compare models using the **Bayesian Information Criterion (BIC)**
# 5. When a more complex model is justified
#
# ## Prerequisites
#
# Complete [Tutorial 01: Rescorla-Wagner](./01_rescorla_wagner.py) first.
#
# ## Background
#
# The standard Rescorla-Wagner model uses one learning rate $\alpha$ for all
# prediction errors. But there's no reason to assume the brain treats positive
# and negative surprise equally.
#
# The **dual learning rate model** splits $\alpha$ into two parameters:
#
# $$V_{t+1} = V_t + \begin{cases} \alpha^+ \cdot \delta_t & \text{if } \delta_t > 0 \text{ (better than expected)} \\ \alpha^- \cdot \delta_t & \text{if } \delta_t \leq 0 \text{ (worse than expected)} \end{cases}$$
#
# This small change has profound clinical implications:
#
# - **Low $\alpha^+$**: Failure to learn from positive outcomes → **anhedonia**
# - **High $\alpha^-$**: Over-learning from negative outcomes → **negativity bias**
# - **Low $\alpha^+$ + high $\alpha^-$**: The computational signature of depression

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# %% [markdown]
# ## Part 1: Implement the dual learning rate model

# %%
def dual_learning_rate(outcomes, alpha_pos, alpha_neg, V0=0.0):
    """
    Rescorla-Wagner model with separate learning rates for positive
    and negative prediction errors.

    Parameters
    ----------
    outcomes : array-like
        Sequence of outcomes (can be +1, -1, or 0).
    alpha_pos : float
        Learning rate for positive prediction errors (delta > 0).
    alpha_neg : float
        Learning rate for negative prediction errors (delta <= 0).
    V0 : float
        Initial expected value.

    Returns
    -------
    values : np.ndarray
        Value before each trial.
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
        delta = outcomes[t] - V
        prediction_errors[t] = delta

        # Use different learning rate depending on PE sign
        if delta > 0:
            V = V + alpha_pos * delta
        else:
            V = V + alpha_neg * delta

    return values, prediction_errors


def standard_rw(outcomes, alpha, V0=0.0):
    """Standard single-alpha Rescorla-Wagner for comparison."""
    outcomes = np.asarray(outcomes, dtype=float)
    n_trials = len(outcomes)
    values = np.zeros(n_trials)
    prediction_errors = np.zeros(n_trials)

    V = V0
    for t in range(n_trials):
        values[t] = V
        delta = outcomes[t] - V
        prediction_errors[t] = delta
        V = V + alpha * delta

    return values, prediction_errors

# %% [markdown]
# ## Part 2: Simulate how asymmetric learning creates anhedonia
#
# Let's compare three "virtual patients":
# 1. **Healthy**: $\alpha^+ = 0.15$, $\alpha^- = 0.15$ (symmetric learning)
# 2. **Anhedonic**: $\alpha^+ = 0.02$, $\alpha^- = 0.15$ (blunted reward learning)
# 3. **Pessimistic**: $\alpha^+ = 0.02$, $\alpha^- = 0.30$ (blunted reward + enhanced punishment learning)

# %%
# Generate a mixed environment: rewards and punishments
n_trials = 150
np.random.seed(123)

# Block 1 (trials 0-49): mostly rewarding (70% reward, 30% nothing)
# Block 2 (trials 50-99): mostly punishing (70% punishment, 30% nothing)
# Block 3 (trials 100-149): mixed (50/50)
block1 = np.random.choice([1, 0], size=50, p=[0.7, 0.3])
block2 = np.random.choice([-1, 0], size=50, p=[0.7, 0.3])
block3 = np.random.choice([1, -1], size=50, p=[0.5, 0.5])
outcomes = np.concatenate([block1, block2, block3])

# Define our virtual patients
patients = {
    'Healthy\n($\\alpha^+=0.15, \\alpha^-=0.15$)': (0.15, 0.15, '#4CAF50'),
    'Anhedonic\n($\\alpha^+=0.02, \\alpha^-=0.15$)': (0.02, 0.15, '#FF9800'),
    'Pessimistic\n($\\alpha^+=0.02, \\alpha^-=0.30$)': (0.02, 0.30, '#F44336'),
}

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for idx, (label, (ap, an, color)) in enumerate(patients.items()):
    values, pes = dual_learning_rate(outcomes, ap, an)

    axes[idx].plot(values, color=color, linewidth=2.5, label=label)
    axes[idx].axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    axes[idx].axvline(50, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
    axes[idx].axvline(100, color='black', linewidth=0.8, linestyle=':', alpha=0.5)

    # Shade blocks
    axes[idx].axvspan(0, 50, alpha=0.06, color='green')
    axes[idx].axvspan(50, 100, alpha=0.06, color='red')
    axes[idx].axvspan(100, 150, alpha=0.06, color='gray')

    axes[idx].set_ylabel('Value ($V_t$)', fontsize=11)
    axes[idx].legend(loc='upper right', fontsize=10)
    axes[idx].set_ylim(-1.1, 1.1)

# Labels
axes[0].set_title('Dual Learning Rate: Three Virtual Patients', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Trial', fontsize=12)

# Block annotations
for ax in axes:
    ax.text(25, -0.95, 'Reward Block', ha='center', fontsize=9, color='green', fontstyle='italic')
    ax.text(75, -0.95, 'Punishment Block', ha='center', fontsize=9, color='red', fontstyle='italic')
    ax.text(125, -0.95, 'Mixed Block', ha='center', fontsize=9, color='gray', fontstyle='italic')

plt.tight_layout()
plt.savefig('../data/02_patient_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **What to notice:**
#
# - The **healthy** learner tracks both reward and punishment blocks effectively
# - The **anhedonic** learner barely increases value during the reward block —
#   they fail to learn from positive outcomes. But they learn normally from
#   punishment.
# - The **pessimistic** learner shows the worst pattern: blunted reward learning
#   AND amplified punishment learning. Their value stays chronically negative.
#
# This is the computational model of depression: not that the world is bad,
# but that the brain *processes* good and bad information asymmetrically.

# %% [markdown]
# ## Part 3: Load pre-generated data and fit both models

# %%
# Load the pre-generated data
data = pd.read_csv('../data/02_dual_learning_data.csv', comment='#')
print(f"Loaded {len(data)} trials from subject: {data['subject_id'].iloc[0]}")
print(f"Blocks: {data['block'].unique()}")
data.head()

# %%
outcomes_data = data['outcome'].values

# %% [markdown]
# ### Define negative log-likelihoods for both models
#
# For outcomes in {-1, 0, +1}, we use a Gaussian likelihood centered on $V_t$
# rather than Bernoulli (which only works for {0, 1}).

# %%
def nll_standard(params, outcomes):
    """NLL for standard RW model (1 free parameter: alpha)."""
    alpha = params[0]
    if alpha <= 0 or alpha >= 1:
        return 1e10

    values, _ = standard_rw(outcomes, alpha)

    # Gaussian likelihood: outcome ~ N(V_t, sigma^2)
    sigma = 0.5  # Fixed noise parameter
    residuals = outcomes - values
    nll = 0.5 * np.sum(residuals**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
    return nll


def nll_dual(params, outcomes):
    """NLL for dual learning rate model (2 free parameters: alpha_pos, alpha_neg)."""
    alpha_pos, alpha_neg = params
    if alpha_pos <= 0 or alpha_pos >= 1 or alpha_neg <= 0 or alpha_neg >= 1:
        return 1e10

    values, _ = dual_learning_rate(outcomes, alpha_pos, alpha_neg)

    sigma = 0.5
    residuals = outcomes - values
    nll = 0.5 * np.sum(residuals**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
    return nll

# %% [markdown]
# ### Fit both models

# %%
# Fit standard RW
res_standard = minimize(nll_standard, x0=[0.3], args=(outcomes_data,),
                        method='Nelder-Mead', options={'xatol': 1e-6})
alpha_fit = res_standard.x[0]
nll_std = res_standard.fun

print("=== Standard RW Model ===")
print(f"  alpha     = {alpha_fit:.4f}")
print(f"  NLL       = {nll_std:.2f}")

# %%
# Fit dual learning rate model
res_dual = minimize(nll_dual, x0=[0.3, 0.3], args=(outcomes_data,),
                    method='Nelder-Mead', options={'xatol': 1e-6})
alpha_pos_fit, alpha_neg_fit = res_dual.x
nll_dual_val = res_dual.fun

print("\n=== Dual Learning Rate Model ===")
print(f"  alpha_pos = {alpha_pos_fit:.4f}")
print(f"  alpha_neg = {alpha_neg_fit:.4f}")
print(f"  NLL       = {nll_dual_val:.2f}")

# %% [markdown]
# ## Part 4: Model comparison with BIC
#
# The dual model has more parameters (2 vs. 1), so of course it will fit
# better (lower NLL). But is the improvement *worth* the added complexity?
#
# The **Bayesian Information Criterion** (BIC) penalizes model complexity:
#
# $$\text{BIC} = k \cdot \ln(n) + 2 \cdot \text{NLL}$$
#
# where $k$ is the number of free parameters and $n$ is the number of data
# points. **Lower BIC = better model** (balancing fit and parsimony).

# %%
n = len(outcomes_data)

# Standard RW: k=1
bic_standard = 1 * np.log(n) + 2 * nll_std

# Dual LR: k=2
bic_dual = 2 * np.log(n) + 2 * nll_dual_val

print("=== Model Comparison (BIC) ===")
print(f"  Standard RW:  BIC = {bic_standard:.2f}")
print(f"  Dual LR:      BIC = {bic_dual:.2f}")
print(f"  Δ BIC = {bic_standard - bic_dual:.2f} (positive = dual model wins)")
print()

if bic_dual < bic_standard:
    print("  → Dual learning rate model is preferred.")
    print("    The additional parameter is justified by improved fit.")
else:
    print("  → Standard RW model is preferred.")
    print("    The additional parameter is not justified.")

# %% [markdown]
# ## Part 5: Visualize fitted model comparison

# %%
# Run both fitted models
v_std, _ = standard_rw(outcomes_data, alpha_fit)
v_dual, _ = dual_learning_rate(outcomes_data, alpha_pos_fit, alpha_neg_fit)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: Both model fits
axes[0].plot(v_std, color='#2196F3', linewidth=2,
             label=f'Standard RW ($\\alpha$={alpha_fit:.3f})')
axes[0].plot(v_dual, color='#E91E63', linewidth=2, linestyle='--',
             label=f'Dual LR ($\\alpha^+$={alpha_pos_fit:.3f}, $\\alpha^-$={alpha_neg_fit:.3f})')
axes[0].scatter(range(n), outcomes_data, color='gray', alpha=0.2, s=10, zorder=1)
axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.3)
axes[0].axvspan(0, 50, alpha=0.06, color='green')
axes[0].axvspan(50, 100, alpha=0.06, color='red')
axes[0].axvspan(100, 150, alpha=0.06, color='gray')
axes[0].set_ylabel('Value ($V_t$)', fontsize=12)
axes[0].set_title('Model Comparison: Standard vs. Dual Learning Rate', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)

# Bottom: Residuals (model error)
residuals_std = outcomes_data - v_std
residuals_dual = outcomes_data - v_dual
axes[1].plot(np.abs(residuals_std), color='#2196F3', alpha=0.6, linewidth=1, label='|Residual| Standard')
axes[1].plot(np.abs(residuals_dual), color='#E91E63', alpha=0.6, linewidth=1, label='|Residual| Dual')
axes[1].set_xlabel('Trial', fontsize=12)
axes[1].set_ylabel('|Residual|', fontsize=12)
axes[1].set_title('Absolute Prediction Residuals', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('../data/02_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Part 6: BIC landscape — when does the dual model win?
#
# Let's systematically explore: if a person truly has asymmetric learning
# rates, how different do $\alpha^+$ and $\alpha^-$ need to be before BIC
# reliably detects the asymmetry?

# %%
def simulate_and_compare(alpha_pos_true, alpha_neg_true, n_trials=150, n_sims=20):
    """Simulate data and compare models n_sims times. Returns fraction of sims where dual wins."""
    dual_wins = 0
    for _ in range(n_sims):
        # Generate outcomes (same structure as our data)
        block1 = np.random.choice([1, 0], size=50, p=[0.7, 0.3])
        block2 = np.random.choice([-1, 0], size=50, p=[0.7, 0.3])
        block3 = np.random.choice([1, -1], size=50, p=[0.5, 0.5])
        sim_outcomes = np.concatenate([block1, block2, block3])

        # Generate "choices" based on true model (add noise)
        v_true, _ = dual_learning_rate(sim_outcomes, alpha_pos_true, alpha_neg_true)

        # Fit both models
        r1 = minimize(nll_standard, x0=[0.3], args=(sim_outcomes,), method='Nelder-Mead')
        r2 = minimize(nll_dual, x0=[0.3, 0.3], args=(sim_outcomes,), method='Nelder-Mead')

        bic1 = 1 * np.log(n_trials) + 2 * r1.fun
        bic2 = 2 * np.log(n_trials) + 2 * r2.fun

        if bic2 < bic1:
            dual_wins += 1

    return dual_wins / n_sims

# Test a few cases
cases = [
    (0.1, 0.1, "Symmetric (no asymmetry)"),
    (0.1, 0.2, "Mild asymmetry"),
    (0.05, 0.3, "Moderate asymmetry"),
    (0.02, 0.5, "Strong asymmetry (depression-like)"),
]

print("=== When does BIC detect asymmetric learning? ===\n")
for ap, an, label in cases:
    frac = simulate_and_compare(ap, an, n_sims=20)
    bar = '█' * int(frac * 20) + '░' * (20 - int(frac * 20))
    print(f"  {label:40s}  {bar}  {frac*100:.0f}%")

# %% [markdown]
# ## Clinical Relevance: Computational Phenotyping of Anhedonia
#
# ### Why this model matters
#
# Anhedonia — the inability to experience pleasure — is a core symptom of
# depression, but it's poorly measured by questionnaires. A patient might say
# "I don't enjoy things anymore," but we can't tell if that's because:
#
# 1. They don't *experience* pleasure in the moment (consummatory anhedonia)
# 2. They don't *learn* from positive experiences (motivational anhedonia)
# 3. They don't *seek out* pleasurable activities (anticipatory anhedonia)
#
# The dual learning rate model directly measures option (2): the rate at which
# a person updates their expectations after positive vs. negative outcomes.
# This is a **computational phenotype** — a quantitative, mechanistic
# description of a symptom.
#
# ### Clinical findings
#
# - **Huys et al. (2013)**: Depressed patients showed selectively reduced
#   learning from positive prediction errors in a reinforcement learning task.
# - **Eshel & Roiser (2010)**: Computational models of reward learning can
#   distinguish between subtypes of depression that look identical on standard
#   clinical assessments.
# - **Treatment prediction**: Changes in $\alpha^+$ may predict who will
#   respond to antidepressant treatment before clinical symptoms change.

# %% [markdown]
# ## Exercises
#
# 1. **Simulate a treatment effect**: A depressed patient starts with
#    $\alpha^+ = 0.02, \alpha^- = 0.25$. After treatment (trial 75), their
#    $\alpha^+$ gradually increases to 0.15 over 25 trials. Simulate and plot
#    the learning curve. Does the model capture the "lag" between biological
#    change and behavioral change?
#
# 2. **Individual differences**: Generate data for 50 "subjects" with
#    $\alpha^+$ and $\alpha^-$ drawn from a Beta distribution. Fit the dual
#    model to each. Plot the distribution of fitted parameters. Do you see
#    the expected cluster structure?
#
# 3. **AIC vs. BIC**: Repeat the model comparison using AIC instead of BIC.
#    When do they disagree? Which is more conservative?
#
# 4. **Three learning rates**: Extend the model to have three learning rates:
#    one for positive PEs, one for negative PEs, and one for zero PEs (no
#    surprise). Does this additional parameter ever improve BIC? Under what
#    conditions?

# %% [markdown]
# ---
#
# **Next tutorial**: [03 — Softmax & Exploration](./03_softmax_exploration.py)
# introduces decision-making via Q-learning and explores how exploration
# deficits relate to psychiatric symptoms.
#
# ---
# *Computational Psychiatry Starter Kit — PRAXIS at USC — Peter Zhou*
