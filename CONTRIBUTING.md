# Contributing to the Computational Psychiatry Starter Kit

Thanks for your interest in contributing! This project is part of the
[comp-psych GitHub organization](https://github.com/comp-psych) and follows
its conventions.

## Adding a New Tutorial

1. **Create a `.py` file** in `notebooks/` using the Jupytext percent format:

   ```python
   # %% [markdown]
   # # Tutorial Title
   # Description of what this tutorial covers.

   # %%
   import numpy as np
   # ... your code ...
   ```

2. **Follow the naming convention**: `NN_short_name.py` where `NN` is the
   next available number (e.g., `04_hierarchical_bayes.py`).

3. **Include synthetic data** in `data/` as a CSV with a header comment
   explaining the columns and generation process. Name it to match:
   `NN_descriptive_name.csv`.

4. **Structure every tutorial with**:
   - A markdown introduction explaining the clinical/theoretical motivation
   - Step-by-step model implementation (no black-box imports)
   - Parameter fitting or inference
   - Visualization of results
   - A "Clinical Relevance" section linking the model to psychiatry
   - Exercises at the end for self-study

5. **Keep dependencies within the Dockerfile stack**. If your tutorial needs
   a package not in the image, open an issue first so we can discuss adding it.

6. **Test your tutorial** by running all cells top-to-bottom in a fresh
   kernel. The notebook must execute without errors.

## Style Guidelines

- Use `matplotlib` and `seaborn` for plots (they're in the Docker image).
- Set a consistent style at the top: `plt.style.use('seaborn-v0_8-darkgrid')`.
- Use descriptive variable names (`learning_rate`, not `lr`).
- Add inline comments for non-obvious math.
- Write markdown cells as if explaining to a smart undergrad who has never
  seen computational modeling before.

## Code Quality

- Format code with `black` (not enforced in CI yet, but appreciated).
- Keep functions small and single-purpose.
- No hardcoded file paths — use `os.path.join` or `pathlib`.

## Submitting Changes

1. Fork the repository.
2. Create a feature branch: `git checkout -b add-tutorial-04`.
3. Commit with clear messages: `Add tutorial 04: hierarchical Bayesian model`.
4. Open a pull request against `main`.
5. Ensure the Docker image still builds: `docker build -t comp-psych-starter .`

## Reporting Issues

- Use GitHub Issues for bugs, typos, or tutorial requests.
- Label tutorial requests with `tutorial-idea`.

## Code of Conduct

Be kind, be constructive, be inclusive. We're all here to learn.

---

*Maintained by [PRAXIS at USC](https://github.com/comp-psych) and Peter Zhou.*
