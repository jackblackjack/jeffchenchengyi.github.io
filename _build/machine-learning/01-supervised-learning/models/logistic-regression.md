---
interact_link: content/machine-learning/01-supervised-learning/models/logistic-regression.ipynb
kernel_name: python3
has_widgets: false
title: 'Logistic Regression'
prev_page:
  url: /machine-learning/01-supervised-learning/models/naivebayes.html
  title: 'Naive Bayes Classifier'
next_page:
  url: /machine-learning/01-supervised-learning/models/statistical-regression-analysis.html
  title: 'Statistical Regression Analysis'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Logistic Regression

Here we meet the discriminative counterpart to the generative Gaussian Naive Bayes. Logistic regression is normally a better choice when we have alot of data because if we have very little data, logistic regression is prone to overfitting those few data points, whereas in Gaussian Naive Bayes, we explicitly state that the MLE estimates must fit a Gaussian distribution. In contrast to Gaussian Naive Bayes, Logistic regression is slow because you can only find the MLE estimates through iterative optimization techniques such as gradient descent, whereas gaussian naive bayes is extremely fast as it just performs counting to calculate probabilities and fit the gaussians. Instead of modelling $P(\mathbf{x}_i, y) = P(\mathbf{x}_i \mid y) * P(y)$ in order to use bayes theorem to model:

$$
\begin{aligned}
P(y=+1\mid x) &= \frac{P(y=+1)*P(x\mid y=+1)}{P(x,y=+1) + P(x,y=-1)} \\
P(y=-1\mid x) &= \frac{P(y=-1)*P(x\mid y=-1)}{P(x,y=+1) + P(x,y=-1)}
\end{aligned}
$$

, let's model this directly:

$$
P(y\mid\mathbf{x}_i;\theta=\mathbf{w}, b)=\frac{1}{1+e^{-y(\mathbf{w}^T \mathbf{x}_i+b)} }
$$



In Gaussian Naive Bayes, our MLE parameters $\hat{\theta}_{MLE}$ were 

$$
\begin{cases}
P(y=spam) \\
P(y=not spam) \\
\mu_{\alpha c=spam} \\
\mu_{\alpha c=not spam} \\
\Sigma_{\alpha} \\
\end{cases}
\text{We find these by counting the words and deriving the probabilities}
$$

$\Sigma_{\alpha}$ if we assumed the features to share the same covariance matrix, AKA all unique words from **all classes** in the vocabulary are sampled from Gaussian distributions of different means (1 for each class=spam/not spam), but same covariance matrix made from the words in our vocabulary instead of different means (1 for each class=spam/not spam) and different covariance matrices $\Sigma_{\alpha c=spam}, \Sigma_{\alpha c=notspam}$ (1 for each class=spam/not spam -> unique spam words make up one covariance matrix, unique non-spam words make up another)).

In Logistic Regression, our MLE parameters $\hat{\theta}_{MLE}$ are 
$$
\begin{cases}
\mathbf{w} \text{ or } \mathbf{w} \text{ and } b \text{ if you didn't absorb it.}
\end{cases}
\text{We find these by using iterative optimnization methods}
$$



<h3 id="maximum-likelihood-estimate-mle">Maximum likelihood estimate (MLE)</h3>


<p>In MLE we choose parameters that <b>maximize the conditional likelihood</b>. The conditional data likelihood $P(\mathbf y \mid X, \mathbf{w})$  is the probability of the observed values $\mathbf y \in \mathbb R^n$ in the training data conditioned on the feature values <span class="math inline">\(\mathbf{x}_i\)</span>. Note that $X=\left[\mathbf{x}_1, \dots,\mathbf{x}_i, \dots, \mathbf{x}_n\right] \in \mathbb R^{d \times n}$. We choose the paramters that maximize this function and we assume that the $y_i$'s are independent given the input features $\mathbf{x}_i$ and $\mathbf{w}$. So,
    
$$
\begin{aligned}
P(Data; \theta) &= P(\mathbf y \mid X; \mathbf{w}) \\
&= \prod_{i=1}^n P(y_i \mid \mathbf{x}_i; \mathbf{w}).
\end{aligned}$$
Now if we take the log,  e obtain
$$\begin{aligned}
\log \bigg(\prod_{i=1}^n P(y_i|\mathbf{x}_i;\mathbf{w})\bigg) &= -\sum_{i=1}^n \log(1+e^{-y_i \mathbf{w}^T \mathbf{x}_i})\\
\end{aligned}$$
</p>

$$\begin{aligned}
\hat{\mathbf{w} }_{MLE} &= \operatorname*{argmax}_{\mathbf{w} } -\sum_{i=1}^n \log(1+e^{-y_i \mathbf{w}^T \mathbf{x}_i})\\
&=\operatorname*{argmin}_{\mathbf{w} }\sum_{i=1}^n \log(1+e^{-y_i \mathbf{w}^T \mathbf{x}_i})
\end{aligned}$$
<p>We need to estimate the parameters <span class="math inline">\(\mathbf{w}\)</span>. To find the values of the parameters at minimum, we can try to find solutions for <span class="math inline">\(\nabla_{\mathbf{w} } \sum_{i=1}^n \log(1+e^{-y_i \mathbf{w}^T \mathbf{x}_i}) =0\)</span>. This equation has no closed form solution, so we will use <a href="http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote07.html">Gradient Descent</a> on the  <i>negative log likelihood</i> $\ell(\mathbf{w})=\sum_{i=1}^n \log(1+e^{-y_i \mathbf{w}^T \mathbf{x}_i})$.</br> </p>


<h3 id="map-estimate">Maximum a Posteriori (MAP) Estimate</h3>
<p>
In the MAP estimate we treat $\mathbf{w}$ as a random variable and can specify a prior belief distribution over it. We may use: <span class="math inline">\(\mathbf{w} \sim \mathbf{\mathcal{N} }(\mathbf 0,\sigma^2 I)\)</span>. This is the Gaussian approximation for LR.</p>
<p>Our goal in MAP is to find the <i>most likely</i> model parameters  <i>given the data</i>, i.e., the parameters that <b>maximaize the posterior</b>.  
<span class="math display">\[\begin{aligned}
P(\mathbf{w} \mid D) = P(\mathbf{w} \mid X, \mathbf y) & \propto P(\mathbf y \mid X, \mathbf{w}) \; P(\mathbf{w})\\
\hat{\mathbf{w} }_{MAP} = \operatorname*{argmax}_{\mathbf{w} } \log \, \left(P(\mathbf y \mid X, \mathbf{w}) P(\mathbf{w})\right) &= \operatorname*{argmin}_{\mathbf{w} } \sum_{i=1}^n \log(1+e^{-y_i\mathbf{w}^T \mathbf{x}_i})+\lambda\mathbf{w}^\top\mathbf{w},
\end{aligned},
\]</span></p>
<p> where $\lambda = \frac{1}{2\sigma^2}$. 
Once again, this function has no closed form solution, but we can use <a href="http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote07.html">Gradient Descent</a> on the <i>negative log posterior</i> $\ell(\mathbf{w})=\sum_{i=1}^n \log(1+e^{-y_i\mathbf{w}^T \mathbf{x}_i})+\lambda\mathbf{w}^\top\mathbf{w}$ to find the optimal parameters $\mathbf{w}$. </p>

<p>For a better understanding for the connection of Naive Bayes and Logistic Regression, you may take a peek at <a href="https://alliance.seas.upenn.edu/~cis520/wiki/index.php?n=Lectures.Logistic">these excellent notes</a>.</p>



---
## Resources
- [Kilian Weinberger's Logistic Regression Lecture](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote06.html)

