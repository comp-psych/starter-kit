# %% [markdown]
# # Tutorial 03: Softmax Action Selection & Exploration
#
# **Computational Psychiatry Starter Kit** — PRAXIS at USC
#
# ---
#
# ## What you'll learn
#
# 1. How to implement **Q-learning** — extending value learning to choices
# 2. How **softmax action selection** converts values into choice probabilities
# 3. What the **inverse temperature** parameter ($\beta$) controls and why
#    it matters clinically
# 4. How to simulate and fit a **2-armed bandit** task
# 5. The clinical relevance of exploration deficits in depression and
#    excessive exploration in mania
#
# ## Prerequisites
#
# Complete Tutorials [01](./01_rescorla_wagner.py) and
# [02](./02_dual_learning_rate.py) first.
#
# ## Background
#
# So far, our models have learned *values* but haven't made *choices*. In
# real experiments (and real life), people must decide between options, and
# their choices reveal their internal value representations.
#
# **Q-learning** (Watkins, 1989) extends the Rescorla-Wagner update to
# multiple actions:
#
# $$Q_{a,t+1} = Q_{a,t} + \alpha \cdot (r_t - Q_{a,t})$$
#
# where $Q_{a,t}$ is the estimated value of action $a$ at time $t$.
#
# But how does an agent choose between actions? The **softmax** rule converts
# Q-values into choice probabilities:
#
# $$P(\text{choose } a) = \frac{\exp(\beta \cdot Q_a)}{\sum_j \exp(\beta \cdot Q_j)}$$
#
# The key parameter is $\beta$ — the **inverse temperature**:
#
# | $\beta$ | Behavior | Clinical analogy |
# |---------|----------|-----------------|
# | $\beta \to 0$ | Random choices (pure exploration) | Disorganized behavior, mania |
# | $\beta \approx 1\text{-}5$ | Balanced exploration/exploitation | Healthy decision-making |
# | $\beta \to \infty$ | Always choose highest Q (pure exploitation) | Rigid, stuck behavior, depression |

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# %% [markdown]
# ## Part 1: Implement Q-learning with softmax

# %%
def softmax(Q, beta):
    """
    Softmax action selection: convert Q-values to choice probabilities.

    Parameters
    ----------
    Q : np.ndarray
        Q-values for each action.
    beta : float
        Inverse temperature (higher = more exploitative).

    Returns
    -------
    probs : np.ndarray
        Choice probabilities for each action.
    """
    # Subtract max for numerical stability
    Q_scaled = beta * (Q - np.max(Q))
    exp_Q = np.exp(Q_scaled)
    return exp_Q / np.sum(exp_Q)


def q_learning_softmax(rewards, choices, alpha, beta, n_actions=2, Q0=0.5):
    """
    Q-learning with softmax action selection.

    Parameters
    ----------
    rewards : array-like
        Observed rewards on each trial.
    choices : array-like
        Observed choices (action indices) on each trial.
    alpha : float
        Learning rate.
    beta : float
        Inverse temperature.
    n_actions : int
        Number of available actions.
    Q0 : float
        Initial Q-value for all actions.

    Returns
    -------
    Q_history : np.ndarray
        Q-values over time, shape (n_trials, n_actions).
    choice_probs : np.ndarray
        Probability assigned to the *chosen* action on each trial.
    pe_history : np.ndarray
        Prediction errors on each trial.
    """
    rewards = np.asarray(rewards, dtype=float)
    choices = np.asarray(choices, dtype=int)
    n_trials = len(rewards)

    Q = np.ones(n_actions) * Q0
    Q_history = np.zeros((n_trials, n_actions))
    choice_probs = np.zeros(n_trials)
    pe_history = np.zeros(n_trials)

    for t in range(n_trials):
        Q_history[t] = Q.copy()

        # Compute choice probabilities
        probs = softmax(Q, beta)
        choice_probs[t] = probs[choices[t]]

        # Prediction error (only for the chosen action)
        pe = rewards[t] - Q[choices[t]]
        pe_history[t] = pe

        # Update Q-value for chosen action only
        Q[choices[t]] += alpha * pe

    return Q_history, choice_probs, pe_history

# %% [markdown]
# ## Part 2: Visualize the effect of inverse temperature
#
# Let's see how different $\beta$ values affect behavior in a simple task.

# %%
# Create a scenario: Q-values that differ
Q_demo = np.array([0.3, 0.7])  # Action 1 is clearly better

betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Choice probabilities as a function of beta
probs_action1 = [softmax(Q_demo, b)[1] for b in betas]
axes[0].plot(betas, probs_action1, 'o-', color='#2196F3', linewidth=2.5, markersize=8)
axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
axes[0].set_xlabel('Inverse Temperature ($\\beta$)', fontsize=12)
axes[0].set_ylabel('P(choose better option)', fontsize=12)
axes[0].set_title('Softmax: How $\\beta$ Shapes Choices', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_ylim(0.45, 1.02)

# Annotate clinical regions
axes[0].axvspan(0, 1, alpha=0.08, color='red')
axes[0].axvspan(1, 7, alpha=0.08, color='green')
axes[0].axvspan(7, 22, alpha=0.08, color='orange')
axes[0].text(0.5, 0.97, 'Erratic\n(mania?)', ha='center', fontsize=8, color='red')
axes[0].text(4, 0.97, 'Balanced\n(healthy)', ha='center', fontsize=8, color='green')
axes[0].text(14, 0.97, 'Rigid\n(depression?)', ha='center', fontsize=8, color='#CC6600')

# Right: Probability distribution over actions for select betas
selected_betas = [0.5, 2.0, 10.0]
colors = ['#F44336', '#4CAF50', '#FF9800']
bar_width = 0.25
x = np.array([0, 1])

for i, (b, c) in enumerate(zip(selected_betas, colors)):
    probs = softmax(Q_demo, b)
    offset = (i - 1) * bar_width
    axes[1].bar(x + offset, probs, bar_width, color=c, alpha=0.8,
                label=f'$\\beta$ = {b}')

axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Action 0\n(Q=0.3)', 'Action 1\n(Q=0.7)'], fontsize=11)
axes[1].set_ylabel('Choice Probability', fontsize=12)
axes[1].set_title('Choice Distribution by $\\beta$', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('../data/03_beta_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Key insight**: $\beta$ doesn't change what the agent *knows* (the Q-values
# are the same). It changes how *decisively* the agent acts on that knowledge.
#
# - Low $\beta$: "I know option B is better, but I'll still try A sometimes"
# - High $\beta$: "I know option B is better, and I'm sticking with it"

# %% [markdown]
# ## Part 3: Simulate a 2-armed bandit task

# %%
def simulate_bandit(alpha, beta, p_reward, n_trials=200, reversal_trial=None):
    """
    Simulate a 2-armed bandit task with Q-learning + softmax.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta : float
        Inverse temperature.
    p_reward : list
        Reward probabilities for each arm [p0, p1].
    n_trials : int
        Number of trials.
    reversal_trial : int or None
        If set, swap reward probabilities at this trial.

    Returns
    -------
    choices, rewards, Q_history : arrays
    """
    n_actions = len(p_reward)
    Q = np.ones(n_actions) * 0.5  # Start uncertain
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    Q_history = np.zeros((n_trials, n_actions))

    p = np.array(p_reward, dtype=float)

    for t in range(n_trials):
        # Reversal?
        if reversal_trial and t == reversal_trial:
            p = p[::-1]  # Swap reward probabilities

        Q_history[t] = Q.copy()

        # Choose action
        probs = softmax(Q, beta)
        choices[t] = np.random.choice(n_actions, p=probs)

        # Generate reward
        rewards[t] = np.random.binomial(1, p[choices[t]])

        # Update Q-value
        pe = rewards[t] - Q[choices[t]]
        Q[choices[t]] += alpha * pe

    return choices, rewards, Q_history

# %%
# Simulate three agents with different beta values
n_trials = 200
reversal = 100
p_reward = [0.3, 0.7]

agents = {
    'Low $\\beta$ = 0.5 (explorer)': (0.1, 0.5, '#F44336'),
    'Medium $\\beta$ = 3.0 (balanced)': (0.1, 3.0, '#4CAF50'),
    'High $\\beta$ = 15.0 (exploiter)': (0.1, 15.0, '#2196F3'),
}

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for idx, (label, (alpha, beta, color)) in enumerate(agents.items()):
    np.random.seed(42)  # Same seed for fair comparison
    choices, rewards, Q_hist = simulate_bandit(alpha, beta, p_reward,
                                                n_trials, reversal)

    # Plot Q-values
    axes[idx].plot(Q_hist[:, 0], color='#FF9800', linewidth=1.5, alpha=0.7,
                   label='$Q_0$')
    axes[idx].plot(Q_hist[:, 1], color='#9C27B0', linewidth=1.5, alpha=0.7,
                   label='$Q_1$')

    # Overlay choices as rug plot
    chose_0 = np.where(choices == 0)[0]
    chose_1 = np.where(choices == 1)[0]
    axes[idx].scatter(chose_0, np.ones_like(chose_0) * -0.05, color='#FF9800',
                      s=3, alpha=0.5, marker='|')
    axes[idx].scatter(chose_1, np.ones_like(chose_1) * 1.05, color='#9C27B0',
                      s=3, alpha=0.5, marker='|')

    axes[idx].axvline(reversal, color='black', linewidth=1.5, linestyle='--',
                      alpha=0.7, label='Reversal')
    axes[idx].set_ylabel('Q-value', fontsize=11)
    axes[idx].set_title(label, fontsize=12, fontweight='bold', color=color)
    axes[idx].legend(loc='center right', fontsize=9)
    axes[idx].set_ylim(-0.15, 1.15)

axes[0].text(50, 1.1, 'Arm 1 better', ha='center', fontsize=9, fontstyle='italic')
axes[0].text(150, 1.1, 'Arm 0 better', ha='center', fontsize=9, fontstyle='italic')
axes[2].set_xlabel('Trial', fontsize=12)
fig.suptitle('2-Armed Bandit with Reversal: Effect of Inverse Temperature',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('../data/03_bandit_simulation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **What to notice:**
#
# - **Low $\beta$ (explorer)**: Q-values track the environment, but choices
#   are noisy — the agent keeps trying the bad arm even when it "knows" better.
# - **Medium $\beta$ (balanced)**: Learns and exploits effectively, adapts to
#   reversal within ~20 trials.
# - **High $\beta$ (exploiter)**: Locks onto the best option quickly, but is
#   *slow to reverse* — once committed, it keeps choosing the now-bad option
#   for many trials after reversal.

# %% [markdown]
# ## Part 4: Load data and fit the Q-learning model

# %%
# Load the pre-generated bandit data
data = pd.read_csv('../data/03_bandit_data.csv', comment='#')
print(f"Loaded {len(data)} trials")
print(f"Phases: {data['phase'].unique()}")
print(f"Choice proportions: arm0={np.mean(data['choice']==0):.2f}, arm1={np.mean(data['choice']==1):.2f}")
data.head()

# %%
choices_data = data['choice'].values
rewards_data = data['reward'].values

# %% [markdown]
# ### Negative log-likelihood for Q-learning + softmax

# %%
def nll_qlearning(params, choices, rewards, n_actions=2):
    """
    Negative log-likelihood for Q-learning with softmax.

    Parameters
    ----------
    params : array-like
        [alpha, beta] — learning rate and inverse temperature.
    choices : array-like
        Observed choices (0 or 1).
    rewards : array-like
        Observed rewards (0 or 1).

    Returns
    -------
    nll : float
    """
    alpha, beta = params

    if alpha <= 0 or alpha >= 1 or beta <= 0 or beta > 50:
        return 1e10

    _, choice_probs, _ = q_learning_softmax(rewards, choices, alpha, beta,
                                             n_actions=n_actions)

    # Avoid log(0)
    eps = 1e-8
    log_lik = np.sum(np.log(np.clip(choice_probs, eps, 1.0)))

    return -log_lik

# %%
# Fit the model
result = minimize(
    nll_qlearning,
    x0=[0.2, 3.0],
    args=(choices_data, rewards_data),
    method='Nelder-Mead',
    options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000}
)

alpha_fit, beta_fit = result.x
print(f"Fitted parameters:")
print(f"  alpha (learning rate):       {alpha_fit:.4f}")
print(f"  beta (inverse temperature):  {beta_fit:.4f}")
print(f"  NLL:                         {result.fun:.2f}")
print(f"  Converged:                   {result.success}")

# %% [markdown]
# ## Part 5: Model-predicted vs. observed choices

# %%
# Run fitted model
Q_fitted, probs_fitted, pe_fitted = q_learning_softmax(
    rewards_data, choices_data, alpha_fit, beta_fit)

# Compute model's predicted P(choose arm 1)
p_arm1_model = np.array([softmax(Q_fitted[t], beta_fit)[1] for t in range(len(Q_fitted))])

# Observed choice proportions (smoothed)
window = 15
observed_arm1 = pd.Series((choices_data == 1).astype(float)).rolling(window, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: Q-values and choice probabilities
axes[0].plot(Q_fitted[:, 0], color='#FF9800', linewidth=2, label='$Q_0$ (fitted)')
axes[0].plot(Q_fitted[:, 1], color='#9C27B0', linewidth=2, label='$Q_1$ (fitted)')
axes[0].axvline(100, color='black', linewidth=1.5, linestyle='--', alpha=0.6)
axes[0].set_ylabel('Q-value', fontsize=12)
axes[0].set_title(f'Fitted Q-Learning ($\\alpha$={alpha_fit:.3f}, $\\beta$={beta_fit:.2f})',
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)

# Bottom: Model P(arm 1) vs. observed proportion
axes[1].plot(p_arm1_model, color='#2196F3', linewidth=2,
             label='Model P(arm 1)', alpha=0.8)
axes[1].plot(observed_arm1, color='#E91E63', linewidth=2,
             label=f'Observed (rolling avg, w={window})', alpha=0.8)
axes[1].axhline(0.5, color='gray', linewidth=0.5, linestyle='--')
axes[1].axvline(100, color='black', linewidth=1.5, linestyle='--', alpha=0.6)
axes[1].set_xlabel('Trial', fontsize=12)
axes[1].set_ylabel('P(choose arm 1)', fontsize=12)
axes[1].set_title('Model vs. Observed Choice Behavior', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('../data/03_model_fit.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Part 6: Parameter recovery — can we trust our estimates?
#
# Before interpreting fitted parameters clinically, we need to verify that
# our fitting procedure actually works. This is called **parameter recovery**.

# %%
# Parameter recovery: simulate data with known params, then try to recover them
n_recovery = 30
true_alphas = np.random.uniform(0.05, 0.5, n_recovery)
true_betas = np.random.uniform(0.5, 15.0, n_recovery)
recovered_alphas = np.zeros(n_recovery)
recovered_betas = np.zeros(n_recovery)

for i in range(n_recovery):
    # Simulate
    sim_choices, sim_rewards, _ = simulate_bandit(
        true_alphas[i], true_betas[i], [0.3, 0.7], n_trials=200, reversal_trial=100)

    # Fit
    res = minimize(nll_qlearning, x0=[0.2, 3.0],
                   args=(sim_choices, sim_rewards),
                   method='Nelder-Mead',
                   options={'xatol': 1e-6, 'maxiter': 5000})
    recovered_alphas[i] = res.x[0]
    recovered_betas[i] = res.x[1]

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Alpha recovery
axes[0].scatter(true_alphas, recovered_alphas, color='#2196F3', alpha=0.7, s=50, edgecolors='white')
axes[0].plot([0, 0.5], [0, 0.5], 'k--', linewidth=1, alpha=0.5, label='Perfect recovery')
corr_alpha = np.corrcoef(true_alphas, recovered_alphas)[0, 1]
axes[0].set_xlabel('True $\\alpha$', fontsize=12)
axes[0].set_ylabel('Recovered $\\alpha$', fontsize=12)
axes[0].set_title(f'Learning Rate Recovery (r={corr_alpha:.3f})', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)

# Beta recovery
axes[1].scatter(true_betas, recovered_betas, color='#E91E63', alpha=0.7, s=50, edgecolors='white')
axes[1].plot([0, 15], [0, 15], 'k--', linewidth=1, alpha=0.5, label='Perfect recovery')
corr_beta = np.corrcoef(true_betas, recovered_betas)[0, 1]
axes[1].set_xlabel('True $\\beta$', fontsize=12)
axes[1].set_ylabel('Recovered $\\beta$', fontsize=12)
axes[1].set_title(f'Inv. Temperature Recovery (r={corr_beta:.3f})', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('../data/03_parameter_recovery.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Interpreting recovery plots**: Points on the diagonal mean perfect
# recovery. Scatter around the diagonal indicates estimation noise. If
# recovery is poor (r < 0.7), parameters should not be interpreted at the
# individual level — only group-level comparisons may be reliable.

# %% [markdown]
# ## Clinical Relevance: Exploration, Exploitation, and Mental Health
#
# ### Depression: Exploitation trap
#
# Depressed individuals tend to show **high $\beta$** — they exploit known
# options rather than exploring new ones. This creates a vicious cycle:
#
# 1. Negative mood → low expected value for novel options
# 2. High $\beta$ → stick with current (unsatisfying) routines
# 3. Never discover rewarding alternatives → reinforces negative mood
#
# Behavioral activation therapy, one of the most effective treatments for
# depression, essentially forces exploration: scheduling novel activities
# to break the exploitation trap.
#
# ### Mania: Excessive exploration
#
# In manic episodes, patients show the opposite pattern — **low $\beta$** with
# erratic, novelty-seeking behavior. They try new things impulsively,
# regardless of past outcomes. This maps to the disorganized, risk-taking
# behavior characteristic of mania.
#
# ### Reversal learning deficits
#
# The reversal point in our bandit task is particularly informative. Patients
# with:
# - **OCD**: Slow to reverse (perseveration on previously correct option)
# - **Borderline personality disorder**: Fast to reverse but then unstable
# - **Schizophrenia**: Impaired across both phases (learning + reversal)
#
# ### The computational insight
#
# What makes this approach powerful is that $\alpha$ and $\beta$ capture
# *different* computational processes:
#
# - $\alpha$ = how fast you update beliefs (learning)
# - $\beta$ = how decisively you act on beliefs (exploration/exploitation)
#
# A patient could have normal learning ($\alpha$) but abnormal decision-making
# ($\beta$), or vice versa. Without computational modeling, these distinct
# deficits are invisible — the patient just "makes bad choices."

# %% [markdown]
# ## Exercises
#
# 1. **Depression simulation**: Simulate 100 trials for a "depressed" agent
#    ($\alpha = 0.1$, $\beta = 20$) and a "healthy" agent ($\alpha = 0.1$,
#    $\beta = 3$) in a 4-armed bandit where reward probabilities are
#    [0.2, 0.4, 0.6, 0.8]. Plot cumulative reward. How much reward does the
#    depressed agent miss by failing to explore?
#
# 2. **Epsilon-greedy vs. softmax**: Implement an epsilon-greedy agent
#    (choose randomly with probability $\epsilon$, otherwise choose best).
#    Compare to softmax on the same bandit task. Which produces more
#    human-like behavior? Which is easier to fit?
#
# 3. **Directed exploration**: Extend the softmax model to include an
#    exploration bonus: $P(a) \propto \exp(\beta \cdot Q_a + \phi \cdot U_a)$
#    where $U_a$ is the uncertainty about action $a$ (e.g., inverse of
#    number of times chosen). Fit this 3-parameter model and compare BIC.
#
# 4. **Mood-congruent learning**: Implement a model where $\beta$ fluctuates
#    trial-by-trial as a function of recent reward history (a crude "mood"
#    signal). Does this create realistic-looking behavior?

# %% [markdown]
# ---
#
# ## Where to go from here
#
# You now have the foundational toolkit for computational psychiatry:
#
# 1. **Rescorla-Wagner**: Simple learning from prediction errors
# 2. **Dual learning rate**: Asymmetric processing of positive/negative outcomes
# 3. **Q-learning + softmax**: Decision-making under uncertainty
#
# Natural next steps:
#
# - **Bayesian models**: Replace point estimates with posterior distributions
#   (see `pymc` and `arviz`, both installed in this Docker image)
# - **Hierarchical models**: Estimate group-level and individual-level
#   parameters simultaneously (see `hBayesDM` — optional local install)
# - **Model-based fMRI**: Correlate trial-by-trial prediction errors with
#   BOLD signal (see `nilearn`, installed in this image)
# - **Real data**: Apply these models to published datasets from the
#   [Computational Psychiatry Course](https://www.computationalpsychiatry.org/)
#
# ---
# *Computational Psychiatry Starter Kit — PRAXIS at USC — Peter Zhou*
