**This uses Logistic Regression to predict the survival rate of the passengers in Titanic**

## Why Logistic Regression?

- **Natural Fit for Binary Outcomes**  
  Predicting “survived vs. didn’t survive” is a two-class problem, and logistic regression directly models the probability of each class using the logistic (sigmoid) function.

- **Probabilistic Predictions**  
  Rather than just outputting a hard label, logistic regression gives a probability score (0–1) for survival, which lets you tune your decision threshold or rank passengers by risk.

- **Interpretability**  
  Coefficients correspond to log-odds changes per feature unit. You can easily see which factors (e.g. class, age, fare) increase or decrease survival odds, making it ideal for post-hoc analysis.

- **Fast to Train and Inference-Efficient**  
  With a closed-form or efficient iterative solvers, it scales to large datasets quickly and has O(n) prediction time, so you can iterate on features or retrain frequently.

- **Baseline and Regularization**  
  As a well-understood, parametric model, it makes a great baseline. You can add L1/L2 penalties (via the `C` hyperparameter) to prevent overfitting when you add more features.

- **Robust Under Many Conditions**  
  Handles both numeric and categorical inputs (with encoding), is less sensitive to outliers than some methods when properly regularized, and often performs competitively when features are approximately linearly separable in log-odds.

Together, these strengths make logistic regression a logical first choice for modeling Titanic survival, balancing predictive power, speed, and clarity of insight.  


### The Sigmoid Function in Logistic Regression

In logistic regression, we model the **log-odds** of the positive class as a linear function of the inputs:
\[
z = \mathbf{w}^\top \mathbf{x} + b
\]
However, we need to convert this real-valued score \(z\) into a probability \(p\in(0,1)\). That’s where the **sigmoid** (or logistic) function comes in:

\[
\sigma(z) \;=\; \frac{1}{1 + e^{-z}}
\]

**Key properties:**
- **Range:** \(\sigma(z)\) always lies between 0 and 1, making it a valid probability.
- **S-shaped curve:**  For large positive \(z\), \(\sigma(z) \to 1\); for large negative \(z\), \(\sigma(z) \to 0\). At \(z=0\), \(\sigma(0)=0.5\).
- **Smooth gradient:**  
  \[
  \frac{d\sigma}{dz} \;=\; \sigma(z)\,\bigl(1 - \sigma(z)\bigr)
  \]  
  This simple derivative makes gradient-based optimization (e.g. gradient descent) efficient.

**How it’s used:**
1. **Compute log-odds:** \(z = \mathbf{w}^\top \mathbf{x} + b\).  
2. **Convert to probability:** \(p = \sigma(z)\).  
3. **Decision rule:** classify as “positive” (survived) if \(p \ge 0.5\) (or another threshold), else “negative.”

By applying the sigmoid, logistic regression turns a linear score into a meaningful probability, enabling both **classification** and **probabilistic interpretation** of the output.  
