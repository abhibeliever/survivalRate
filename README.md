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
