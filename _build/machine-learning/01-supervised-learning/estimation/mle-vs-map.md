---
interact_link: content/machine-learning/01-supervised-learning/estimation/mle-vs-map.ipynb
kernel_name: python3
has_widgets: false
title: 'Maximum Likelihood Estimation Vs Maximum A Priori'
prev_page:
  url: /machine-learning/01-supervised-learning/estimation/multivariate-gaussian
  title: 'Multivariate Gaussian Distribution'
next_page:
  url: /machine-learning/01-supervised-learning/estimation/convex-optimization
  title: 'Convex Optimization'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# MLE VS MAP

We will go through the differences in MLE and MAP here which will be very helpful in understanding the connection between naive bayes and logistic regression, as well as how MAP creates a "regularized" version of MLE. We will use a simple coin tossing scenario to show the differences and similarities between MLE and MAP.

<h3>Machine Learning and estimation</h3>

In supervised Machine learning you are provided with training data $D$. You use this data to train a model, represented by its parameters $\theta$. With this model you want to make predictions on a test point $x_t$. Recall that 

$${p(Y|X) = \frac{p(X|Y) * p(Y)}{p(X)} }, p(X) = \sum^{\text{Set of distinct labels} Y}_{\text{label }y\,\in\,\text{Set of distinct labels} Y} p(X|y) * p(y)$$

<li>
    When we estimate 
    $$P(X,Y)[\text{Joint probability of features and labels}] \\
    = \mathbf{P(X|Y)P(Y)}[\text{Conditional probability of features given labels}\times\text{Marginal probability of labels}] \\
    = \mathbf{P(Y|X)P(X)}[\text{Conditional probability of labels given features}\times\text{Marginal probability of feature}]
    $$ , then we call it 
    <i>generative learning.</i>
</li>
<li>
    When we only estimate $P(X,Y) = P(Y|X) \times P(X) \propto \mathbf{P(Y|X)}=\text{conditional probability of label given feature}$ directly, then we call it
    <i>discriminative learning.</i>
</li>

<h4>Probabilistic Frameworks for setting up Machine learning Problems</h4>
<ul>
<li>
    <b>MLE [Frequentist $\because \theta$ is a parameter] => Frequentist version of MAP with Uniform Prior</b> 
    <ul>
        <li>
            Prediction: $P(y|x_t;\theta)=\text{posterior of label given feature, parameterized by }\theta$ 
        </li>
        <li>
            Learning: $\theta=\operatorname*{argmax}_\theta P(D;\theta)=\text{likelihood of data, parameterized by }\theta$. Here $\theta$ is purely a model parameter.
        </li>
        <li>
            <i>In MLE, whenever we say likelihood, it's always likelihood of data. Never confuse likelihood in MLE with conditional probability. Conditional probability applies to everything like $P(Y = y_k \vert X = \mathbf{x_i})$ is conditional probability of seeing label $y_k$ given observing feature vector $\mathbf{x_i}$</i>
        </li>
    </ul>
</li>
<li>
    <b>MAP [Bayesian $\because \theta$ is a random variable] => Bayesian version of MLE with an arbitrary Prior</b> 
    <ul>
        <li>
            Prediction: $P(y|x_t,\theta)$ 
        </li>
        <li>
            Learning: $\theta=\operatorname*{argmax}_\theta P(\theta|D)\propto P(D \mid \theta) P(\theta)$. Here $\theta$ is a random variable.
        </li>
        <li>
            <i>Don't confuse MAP with generative learning and MLE with discriminative learning! Both MLE and MAP can be applied to both Generative and Discriminative learning. Generative VS Discriminative learning refers to how we'll be modelling the data to be used in our machine learning model while MLE VS MAP refers to how we get the parameters for our machine learning model. <b>E.g. we could say that logistic regression is the discriminative version of naive bayes, while we can use both MLE and MAP to find the optimal parameters for both naive bayes and logistic regression!</b></i>
        </li>
    </ul>
</li>
<li>
    <b>"True Bayesian"</b> 
    <ul>
        <li>
            Prediction: $P(y|x_t,D)=\int_{\theta}P(y|\theta)P(\theta|D)d\theta$. Here $\theta$ is integrated out - our prediction takes all possible models into account.
        </li>
    </ul>
</ul>

<p> As always the differences are subtle. In MLE we maximize $\log\left[P(D;\theta)\right]$ in MAP we maximize $\log\left[P(D|\theta)\right]+\log\left[P(\theta)\right]$. So essentially in MAP we only add the term $\log\left[P(\theta)\right]$ to our optimization. This term is independent of the data and penalizes if the parameters, $\theta$ deviate too much from what we believe is reasonable. We will later revisit this as a form of <a href="http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote10.html">regularization</a>, where $\log\left[P(\theta)\right]$ will be interpreted as a measure of classifier complexity. 
</p>



---

# MLE (Maximum Likelihood Estimation) - Frequentist Statistics

Goal:
- Find $P(Data;\theta) : \theta$ is just a symbol for any parameter. 
- In the simple case when we're merely finding the MLE estimate for a single probability distribution, e.g. inn the case of a binomial distribution, $\theta = p$, where the probability of getting exactly k successes in n trials is given by the pmf: 

$$f(k,n,p) = \Pr(k;n,p) = \Pr(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$$

- In the case of a normal distribution, $\theta = \mu, \sigma$, where pdf:

$$f(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2} } e^{ -\frac{(x-\mu)^2}{2\sigma^2} }$$

Steps:
1. Make an explicit modelling assumption about what type of distribution your data is sampled from
2. Set the parameters of the distribution so that data observed is the most likely

Coin Toss:
- **Step 1**: We assume that the coin toss event follows the Binomial distribution of $n$ independent Bernoulli trials and we define the probability of success to be the probability of getting heads, $P(H)=\theta$:
$$
\begin{align}
P(D; \theta) &= \begin{pmatrix} n_H + n_T \\  n_H  \end{pmatrix} \theta^{n_H} (1 - \theta)^{n_T}, 
\end{align}
$$
- **Step 2**: Find $\hat{\theta}$ to maxmize likelihood of the data, $P(D;\theta)$:

$$
\begin{align}
 \hat{\theta}_{MLE} &= \operatorname*{argmax}_{\theta} \,P(D; \theta) \\
  &= \operatorname*{argmax}_{\theta} \begin{pmatrix} n_H + n_T \\ n_H \end{pmatrix} \theta^{n_H} (1 - \theta)^{n_T} \\
&= \operatorname*{argmax}_{\theta} \,\log\begin{pmatrix} n_H + n_T \\ n_H \end{pmatrix} + n_H \cdot \log(\theta) + n_T \cdot \log(1 - \theta) \\
&= \operatorname*{argmax}_{\theta} \, n_H \cdot \log(\theta) + n_T \cdot \log(1 - \theta)
\end{align}
$$

- Take derivative w.r.t. $\theta$ and set the derivative to 0
    
$$
\begin{align}
\frac{n_H}{\theta} = \frac{n_T}{1 - \theta} \Longrightarrow n_H - n_H\theta = n_T\theta \Longrightarrow \theta = \frac{n_H}{n_H + n_T}
\end{align}
$$

- In this coin toss scenario, our problem is one dimensional, because we only care about the probability of getting heads or tails, meaning that we're finding:
    - $$
        \operatorname*{argmax}_{\theta}\,P(D = X;\theta) = \frac{\sum^{n}_{i=1} I[x_i = x]}{n} = \frac{n_H}{n_H + n_T}
      $$, $I[x_i = x]$ is the Iverson bracket, 1 if condition inside bracket is true, 0 otherwise

<ul>
    <li>MLE gives the explanation of the data you observed.</li>
	<li>If $n$ is large and your model/distribution is correct (that is $\mathcal{H}$ includes the true model), then MLE finds the <b>true</b> parameters.</li>
    <li>But the MLE can overfit the data if $n$ is small. It works well when $n$ is large.</li>
	<li>If you do not have the correct model (and $n$ is small) then MLE can be terribly wrong! </li>
</ul>



---

# How to account for situations with little data?

In our coin toss example, what if we had never seen any heads before? This would get us a probability of 0 for seeing heads, which we might believe is ridiculous. To correct for this, we have two approaches.

### Laplace / Additive Smoothing (Frequentist Approach)
<p>Assume you have a hunch that $\theta$ is close to $0.5$. But your sample size is small, so you don't trust your estimate.
</br> </br>
<u>Simple fix:</u> Add $m$ imaginery throws that would result in $\theta'$ (e.g. $\theta = 0.5$). Add $m$ Heads and $m$ Tails to your data.
$$
\hat{\theta} =  \frac{n_H + m}{n_H + n_T + 2m}
$$
For large $n$, this is an insignificant change.
For small $n$, it incorporates your "prior belief" about what $\theta$ should be.
</p>

### Prior Beliefs (Bayesian Approach)
<p>
Model $\theta$ as a <b>random variable</b>, drawn from a distribution $P(\theta)$.
<u>Note</u> that $\theta$ is <b>not</b> a random variable associated with an event in a sample space.
    In frequentist statistics, this is forbidden. In Bayesian statistics, this is allowed and you can specify a prior belief $P(\theta)$ defining what values you believe  $\theta$ is likely to take on.
</p>

<p>
Now, we can look at $P(\theta \mid D) = \frac{P(D\mid \theta) P(\theta)}{P(D)}$ (recall Bayes Rule!), where 
<ul>
  <li> $P(\theta)$ is the <b>prior</b> distribution over the parameter(s) $\theta$, before we see any data.</li>
  <li> $P(D \mid \theta)$ is the <b>likelihood</b> of the data given the parameter(s) $\theta$. </li>
  <li> $P(\theta \mid D)$ is the <b>posterior</b> distribution over the parameter(s) $\theta$ <b>after</b> we have observed the data.</li>
</ul>  
</p>

<p>
A natural choice for the prior $P(\theta$) is the <a href="https://en.wikipedia.org/wiki/Beta_distribution">Beta distribution</a>:
\begin{align}
P(\theta) = \frac{\theta^{\alpha - 1}(1 - \theta)^{\beta - 1} }{B(\alpha, \beta)}
\end{align}
where $B(\alpha, \beta) = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the normalization constant (if this looks scary don't worry about it, it is just there to make sure everything sums to $1$ and to scare children at Halloween). Note that here we only need a distribution over a singly binary random variable $\theta$. (The multivariate generalization of the Beta distribution is the <a href="https://en.wikipedia.org/wiki/Dirichlet_distribution">Dirichlet distribution</a>.)
</p>

<p>
Why is the Beta distribution a good fit?
<ul>
  <li>it models probabilities ($\theta$ lives on $\left[0,1\right]$) </li>
  <li>it is of the same distributional family as the binomial distribution (<b>conjugate prior</b>) $\rightarrow$ the math will turn out nicely: </li>
</ul>
\begin{align}
P(\theta \mid D) \propto P(D \mid \theta) P(\theta) \propto \theta^{n_H + \alpha -1} (1 - \theta)^{n_T + \beta -1}
\end{align}
</p>

<p>
So far, we have a distribution over $\theta$. How can we get an estimate for $\theta$?
<center>
<img src="http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/MLEMAP/BayesianCoinFlipping.png" width="500px" />
</center>



---

# MAP (Maximum A Posteriori) - Bayesian Statistics

For example, we can choose $\hat{\theta}$ to be the <u>most likely $\theta$ given the data</u>.

<b><u>MAP Principle:</u></b>
Find $\hat{\theta}$ that maximizes the posterior distribution $P(\theta \mid D)$:
\begin{align}
 \hat{\theta}_{MAP} &= \operatorname*{argmax}_{\theta} \,P(\theta \mid D) \\
					&= \operatorname*{argmax}_{\theta} \, \log P(D \mid \theta) + \log P(\theta)
\end{align}
For out coin flipping scenario, we get: 
\begin{align}
 \hat{\theta}_{MAP} &= \operatorname*{argmax}_{\theta} \;P(\theta | Data) \\
&= \operatorname*{argmax}_{\theta} \; \frac{P(Data | \theta)P(\theta)}{P(Data)} && \text{(By Bayes rule)} \\
&= \operatorname*{argmax}_{\theta} \;\log(P(Data | \theta)) + \log(P(\theta)) \\
&= \operatorname*{argmax}_{\theta} \;n_H \cdot \log(\theta) + n_T \cdot \log(1 - \theta) + (\alpha - 1)\cdot \log(\theta) + (\beta - 1) \cdot \log(1 - \theta) \\
&= \operatorname*{argmax}_{\theta} \;(n_H + \alpha - 1) \cdot \log(\theta) + (n_T + \beta - 1) \cdot \log(1 - \theta) \\
&\Longrightarrow  \hat{\theta}_{MAP} = \frac{n_H + \alpha - 1}{n_H + n_T + \beta + \alpha - 2}
\end{align}

<p>
A few comments:
<ul>
    <li>The MAP estimate is identical to MLE with $\alpha-1$ hallucinated <i>heads</i> and $\beta-1$ hallucinated <i>tails</i></li>
    <li>As $n \rightarrow \infty$, $\hat\theta_{MAP} \rightarrow \hat\theta_{MLE}$ as $\alpha-1$ and $\beta-1$ become irrelevant compared to very large $n_H,n_T$.</li>
    <li>MAP is a great estimator if an accurate prior belief is available (and mathematically tractable).</li>
    <li>If $n$ is small, MAP can be very wrong if prior belief is wrong!</li>
</ul>
</p>



---

# "True" Bayesian approach

<p>Note that MAP is only one way to get an estimator. There is much more information in $P(\theta \mid D)$, and it seems like a shame to simply compute the mode and throw away all other information. A true Bayesian approach is to use the posterior predictive distribution directly to make prediction about the label $Y$ of a test sample with features $X$:
$$
P(Y\mid D,X) = \int_{\theta}P(Y,\theta \mid D,X) d\theta = \int_{\theta} P(Y \mid \theta, D,X) P(\theta | D) d\theta
$$
Unfortunately, the above is generally <i>intractable</i> in closed form and 
sampling techniques, such as Monte Carlo approximations, are used to approximate the distribution. A pleasant exception are <a href="http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote15.html">Gaussian Processes</a>, which we will cover later in this course. 
 </p>

<p>
Another exception is actually our  coin toss example.  
<!--\begin{align}
 \hat{\theta}_{post\_mean} = E\left[\theta | D\right] = \int_{\theta} \theta P(\theta \mid D) d\theta
\end{align}
For coin flipping, this can be computed as $\hat{\theta}_{post\_mean} = \frac{n_H + \alpha}{n_H + \alpha + n_T + \beta}$.
-->
To make <i>predictions</i> using $\theta$ in our coin tossing example, we can use
\begin{align}
P(heads \mid D) =& \int_{\theta} P(heads, \theta \mid D) d\theta\\
 =& \int_{\theta} P(heads \mid \theta, D) P(\theta \mid D) d\theta \ \ \ \ \ \  \textrm{(Chain rule: $P(A,B|C)=P(A|B,C)P(B|C)$.)}\\ 
  =& \int_{\theta} \theta P(\theta \mid D) d\theta\\ 
  =&E\left[\theta|D\right]\\
 =&\frac{n_H + \alpha}{n_H + \alpha + n_T + \beta}
\end{align}
Here, we used the fact that we defined $P(heads \mid D, \theta)= P(heads \mid \theta)=\theta $ (this is only the case because we assumed that our data is drawn from a binomial distribution - in general this would not hold). 
</p>



---

# Fisher Information

The likelihood / log-likelihood function is a curve.

0th moment - Area / Volume under the likelihood function curve

1st moment - Mean (Expected value of the partial derivative / gradient of the curve with respect to $\theta$ of the natural logarithm of the likelihood function is called the "score")

$$
\begin{align}
\operatorname{E} \left[\left. \frac{\partial}{\partial\theta} \log P(Data;\theta)\right|\theta \right]
&= \int \frac{\frac{\partial}{\partial\theta} P(Data;\theta)}{P(Data; \theta)} P(Data;\theta)\,d(Data) \\
&= \frac{\partial}{\partial\theta} \int P(Data; \theta)\,d(Data) \\
&= \frac{\partial}{\partial\theta} 1 = 0.
\end{align}
$$

2nd moment - Variance (Expected value of the second partial derivative of the curve with respect to $\theta$ of the natural logarithm of the likelihood function is called the Fisher Information)

$$
\mathcal{I}(\theta)=\operatorname{E} \left[\left. \left(\frac{\partial}{\partial\theta} \log f(X;\theta)\right)^2\right|\theta \right] = \int \left(\frac{\partial}{\partial\theta} \log f(x;\theta)\right)^2 f(x; \theta)\,dx,
$$

Near the maximum likelihood estimate, low Fisher information therefore indicates that the maximum appears "blunt", that is, the maximum is shallow and there are many nearby values with a similar log-likelihood. Conversely, high Fisher information indicates that the maximum is sharp.

## Cramer-Rao Lower Bound

$$
Var(\hat{\theta}) \geq {\mathcal{I} }^{-1},\,\hat{\theta}=\text{MLE estimate of }\theta
$$

## How do we quantify degree of uncertainty of our MLE estimate?

Using the Cramer-Rao Lower Bound and the Central Limit Theorem, we can say that:
$$
\hat{\theta} \sim \mathcal{N}(\theta_0, {\mathcal{I} }^{-1}), \theta_0 = \text{True parameter}
$$



---

## Resources:
- [Kilian Weinberger's Estimating Probabilities from data lecture](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html)
- [Mihaela van der Schaar's Generative VS Discriminative Notes](http://www.stats.ox.ac.uk/~flaxman/HT17_lecture5.pdf)
- [Laplace smoothing wiki](https://en.wikipedia.org/wiki/Additive_smoothing)

