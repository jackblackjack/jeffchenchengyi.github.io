---
interact_link: content/machine-learning/miscellaneous-topics/empirical-risk-minimization.ipynb
kernel_name: python3
has_widgets: false
title: 'Empirical Risk Minimization'
prev_page:
  url: /machine-learning/miscellaneous-topics/statistics
  title: 'Statistics'
next_page:
  url: /machine-learning/miscellaneous-topics/etl-pipelines
  title: 'ETL Pipelines'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Empirical Risk Minimization

Losses can be derived from MLE / MAP principles if algorithm is parametric such as cross-entropy loss. However, Hinge Loss is derived from SVM, a non-parametric ML algo. Squared Loss / MSE Loss can be derived through both MLE when we assume the data follows a Gaussian Distribution and when we use the Ordinary Least Squares method to minimize the squared residuals directly.



---
# Classification Losses



## Cross-Entropy Loss

$$H(p, q) = -\sum_{i=1}^n {\mathrm{p}(x_i) \log_b \mathrm{q}(x_i)}\,\{\text{for discrete }x\} \\ = -\int_{x} {\mathrm{p}(x) \log_b \mathrm{q}(x)}{dx}\,\{\text{for continuous }x\}$$

If we build an animal image classifier to predict a red panda:

<img src="https://live.staticflickr.com/8146/29545409156_e6c3547efc_b.jpg" width="500px"/>

| Animal Classes | Predicted Distribution $q$ | True Distribution $p$ |
| :------------: | :------------------------: | :-------------------: |
| Cat            | 0.02                       | 0.00                  |
| Dog            | 0.30                       | 0.00                  |
| Fox            | 0.45                       | 0.00                  |
| Cow            | 0.00                       | 0.00                  |
| Red Panda      | 0.25                       | 1.00                  |
| Bear           | 0.05                       | 0.00                  |
| Dolphin        | 0.00                       | 0.00                  |

$$H(p, q) = -\sum_{i=1}^n {\mathrm{p}(x_i) \log_b \mathrm{q}(x_i)} = -{log}_2{0.25} = 1.386$$



---
# Regression Losses



---
## Resources

- [Empirical Risk Minimization Lecture by Kilian Weinberger](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote10.html)
- [5 Regression Loss Functions All Machine Learners Should Know](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)

