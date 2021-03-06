---
interact_link: content/machine-learning/01-supervised-learning/classification/naivebayes.ipynb
kernel_name: python3
has_widgets: false
title: 'Naive Bayes Classifier'
prev_page:
  url: /machine-learning/01-supervised-learning/classification/svm-convex-optimization-derivation
  title: 'SVM (Convex Optimization Derivation)'
next_page:
  url: /machine-learning/01-supervised-learning/classification/logistic-regression
  title: 'Logistic Regression'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Naive Bayes

In this algorithm cookbook, we'll go over the naive bayes algorithm, using MLE to find the optimal parameters for the model.

In some scenarios unlike the Coin Toss example which is "1-dimensional" in that the features / independent var(s) (Heads or Tails) are essentially the labels / dependent var(s) too, we can't straight away assume a probability distribution to the data. 

- In this coin toss scenario, our problem is one dimensional, because we only care about the probability of getting heads or tails, meaning that we're finding:
    - $$
        \hat{\theta}_{MLE} = \operatorname*{argmax}_{\theta}\,P(D = X;\theta) = \frac{\sum^{n}_{i=1} I(x_i = x)}{n} = \frac{n_H}{n_H + n_T}
      $$
      
- In spam classification scenario, our problem is multi dimensional, because we care about how the features of an email predict the label, meaning that we're finding:
    - $$
        \hat{\theta}_{MLE} = P(x_\alpha | y) \text{ and } P(y) = \operatorname*{argmax}_{P(x_\alpha | y) \text{ and } P(y)}\,P(D = y|X;\theta) = \sum_{\alpha = 1}^{d} \log(P(x_\alpha | y)) + \log(P(y))
      $$

For example, in a binary classification scenario like spam classification where we have multiple features (word tokens / n-grams...) and a single label (spam / not spam) or any multi-class classification task like movie genre classification where you have multiple features (movie plot word tokens / n-grams..., movie poster2vecs...) trying to predict multiple labels (comedy / romance / horror), we can't automatically say that the distribution of whether an email is spam / not spam or whether a movie is a comedy / romance / horror follows a binomial / poisson / normal distribution. Hence, we have to decompose $P(D;\theta)$ **(in the case of MLE estimation)** or $P(\theta\vert D) \propto P(D\vert\theta) \times P(\theta)$ **(in the case of MAP estimation)** further and in the context of Naive Bayes, our parameters become 



---
# Brief Overview

What is it used for?
- Classification

### Assumptions

1. Just like the majority of machine learning algorithms that do not work with sequential data, we assume that observations are i.i.d. (independent and identically distributed)
    - i.e. none of your observations affect another's
    
2. The key **Naive Bayes assumption** is that the Naive Bayes classifier assume that the effect of the value of a predictor (x) on a given class (c) is independent of the values of other predictors. This assumption is called class conditional independence.
    - i.e. none of the distinct features of each of your observations affect one another to get the observation's label\



---
# Derivation

We start off by noting that we want to come up with a model that models the **joint probability distribution** of $X \text{ and } Y = P(X, Y)$. Hence, we start off by trying to build a **generative** model.

Looking at $P(X, Y)$, recall that by law of conditional probability, it can be decomposed into 2 ways:
$$
1.\,P(X, Y) = P(Y \vert X) * P(X) = \text{Conditional probability of labels given features} \times \text{Marginal probability of features} \\
2.\,P(X, Y) = P(X \vert Y) * P(Y) = \text{Conditional probability of features given labels} \times \text{Marginal probability of labels}
$$

Let's go ahead and find out which is more feasible:

### 1. <u>$P(X=\mathbf{x}, Y=y) = P(Y=y|X=\mathbf{x}) * P(X=\mathbf{x})$:</u>

Firstly, by Assumption 1 that observations are i.i.d., we can break $P(\mathbf{x}, y) = P(y|\mathbf{x}) * P(\mathbf{x})$ further into $P((\mathbf{x}_1,y_1),\dots,(\mathbf{x}_n,y_n))=\Pi_{i=1}^n P(\mathbf{x}_i,y_i).$

Secondly, $\Pi_{i=1}^n P(\mathbf{x}_i,y_i)$ can be further broken down into $\Pi_{i=1}^n P(y_i \vert \mathbf{x}_i) * P(\mathbf{x}_i)$ and because we already need to calculate $\Pi_{i=1}^n P(y_i \vert \mathbf{x}_i)$, let's forego building a generative model and just focus on building a discriminative version by predicting ${P}(y|\mathbf{x})$ directly:
- To compute this estimate,
$$
\hat{P}(y|\mathbf{x}) = 
\frac{\hat{P}(y,\mathbf{x})}{P(\mathbf{x})} = 
\frac{\sum_{i=1}^{n} I(\mathbf{x}_i = \mathbf{x} \wedge {y}_i = y)}{\sum_{i=1}^{n} I(\mathbf{x}_i = \mathbf{x})}
$$
- This would mean that we have to calculate the probability that the exact set of features $\mathbf{x}_i$ are observed along with the label $y_i$, hence counting all the times when the exact set of features $\mathbf{x}_i$ are observed along with the label $y_i$. This is, however, highly impractical as we go into higher dimensional spaces like faces as the probability that we see the exact same face vectors is extremely low.

<center>
<img src="http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/naive_bayes/naive_bayes_img1.png" width="500px" />
</center>
The Venn diagram illustrates that the MLE method estimates $\hat{P}(y|\mathbf{x})$ as 
$$
\hat{P}(y|\mathbf{x}) = \frac{|C|}{|B|}
$$

- **Hence, the MLE estimate is only good if there are many training vectors with the same identical features as x! In high dimensional spaces (or with continuous x), this never happens!**

### 2. <u>$P(X=\mathbf{x}, Y=y) = P(X=\mathbf{x}|Y=y) * P(Y=\mathbf{y})$:</u>

Looks like we're back to building a **generative** model! Estimating $P(y)$ is easy. For example, if $Y$ takes on discrete binary values estimating $P(Y)$ reduces to coin tossing. We simply need to count how many times we observe each outcome (in this case each class):
$$P(y = c)  = \frac{\sum_{i=1}^{n} I(y_i = c)}{n} = \hat\pi_c
$$

<p>
Estimating $P(\mathbf{x}|y)$, however, is not easy!
The additional assumption that we make is the <i>Naive Bayes assumption</i>.
</p>

<u>Naive Bayes Assumption:</u>
$$
P(\mathbf{x} | y) = \prod_{\alpha = 1}^{d} P(x_\alpha | y), \text{where } x_\alpha = [\mathbf{x}]_\alpha \text{ is the value for feature } \alpha
$$
i.e., feature values are <b>independent given the label!</b> This is a very <b>bold</b> assumption.
<img src="https://image.slidesharecdn.com/bayes-6-140829123306-phpapp02/95/bayes-6-12-638.jpg?cb=1424509648" width="400px" />
<p>
For example, a setting where the Naive Bayes classifier is often used is spam filtering. Here, the data is emails and the label is <i>spam</i> or <i>not-spam</i>.  The Naive Bayes assumption implies that the words in an email are conditionally independent, given that you know that an email is spam or not. Clearly this is not true. Neither the words of spam or not-spam emails are drawn independently at random. However, the resulting classifiers can work well in practice even if this assumption is violated. 
</p>

<center>
<img src="http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/naive_bayes/NBschematic.png" width="800px" />
<caption><i>Illustration behind the Naive Bayes algorithm. We estimate $P(x_\alpha|y)$ independently in each dimension (middle two images) and then obtain an estimate of the full data distribution by assuming conditional independence $P(\mathbf{x}|y)=\prod_\alpha P(x_\alpha|y)$ (very right image).</i></caption>
</center>
<p>
So, for now, let's pretend the Naive Bayes assumption holds.
Then the Bayes Classifier can be defined as
\begin{align}
h(\mathbf{x}) &= \operatorname*{argmax}_y P(y | \mathbf{x}) \\
&= \operatorname*{argmax}_y \; \frac{P(\mathbf{x} | y)P(y)}{P(\mathbf{x})} \\
&= \operatorname*{argmax}_y \; P(\mathbf{x} | y) P(y) && \text{($P(\mathbf{x})$ does not depend on $y$)} \\
&= \operatorname*{argmax}_y \; \prod_{\alpha=1}^{d} P(x_\alpha | y) P(y) && \text{(by the naive Bayes assumption)}\\
&= \operatorname*{argmax}_y \; \sum_{\alpha = 1}^{d} \log(P(x_\alpha | y)) + \log(P(y)) && \text{(as log is a monotonic function)}
\end{align}
</p>

Estimating $\log(P(x_\alpha | y))$ is easy as we only need to consider one dimension. And estimating $P(y)$
is not affected by the assumption.

- **The assumption leads to an overestimation of the probability in the context of spam. E.g. The probability of an email containing the both words "viagra" and "nigerian prince" given the email is spam is very low since they come from different types of spam emails. However, after decomposing the $P(\mathbf{x} \vert y)$, the individual probabilities $P(x_\alpha \vert y)$ is actually pretty high - probability of seeing "viagra" given spam * probability of seeing "nigerian prince" given spam**

## MLE Estimation

$$
\begin{aligned}
    \hat{\theta_{MLE} } &= \operatorname*{argmax}_\theta P(\mathcal{Data}=X=\mathbf{x}, Y=y;\theta) \\
    &= \operatorname*{argmax}_\theta P(X=\mathbf{x}|Y=y) * P(Y=y) \\
    &= \operatorname*{argmax}_\theta \prod_{\alpha=1}^{d} P(x_\alpha | y) P(y) \\
    &= \operatorname*{argmax}_\theta \sum_{\alpha = 1}^{d} \log(P(x_\alpha | y)) + \log(P(y))
\end{aligned}
$$

$\hat{\theta^{(1)}_{MLE} } =P(y = c) $:

$$
P(y = c)  = \frac{\sum_{i=1}^{n} I(y_i = c)}{n} = \hat\pi_c
$$

**This is basically just how often you see spam / not spam.**

### Case 1: Categorical Features
<u>Features:</u> 
$$[\mathbf{x}]_\alpha \in \{f_1, f_2, \cdots, f_{K_\alpha}\}$$
Each feature $\alpha$ falls into one of $K_\alpha$ categories.
(Note that the case with binary features is just a specific case of this, where $K_\alpha = 2$.) An example of such a setting may be medical data where one feature could be <i>gender</i> (male / female) or <i>marital status</i> (single / married / widowed). 

</br> </br>
<u>Model $P(x_\alpha \mid y)$:</u>
$$
P(x_{\alpha} = j | y=c) = [\theta_{jc}]_{\alpha} \\
\text{ and } \sum_{j=1}^{K_\alpha} [\theta_{jc}]_{\alpha} = 1
$$
where $[\theta_{jc}]_{\alpha} $ is the probability of feature $\alpha$ having the value $j$, given that the label is $c$.
And the constraint indicates that $x_{\alpha}$ must have one of the categories $\{1, \dots, K_\alpha\}$.
</br> </br>

<u>Parameter estimation:</u>

$$\hat{\theta^{(2)}_{MLE} } = P(x_{\alpha} = j | y=c) = [\theta_{jc}]_{\alpha} \text{ and } \sum_{j=1}^{K_\alpha} [\theta_{jc}]_{\alpha} = 1$$ where $[\theta_{jc}]_{\alpha}$ is the probability of feature $\alpha$ (e.g. transportation mode $\alpha$) having the value $j$ (e.g. $j = \text{ bus, train, taxi, car }, K_\alpha = 4 $), given that the label is $c$ (e.g. $c = \text{ person is late, person is early, person is on time}$). And the constraint indicates that $x_\alpha$ must have one of the categories $\{1,\ldots,K_\alpha\}$:

$$
\begin{align}
[\hat\theta_{jc}]_{\alpha} &= \frac{\sum_{i=1}^{n} I(y_i = c) I(x_{i\alpha} = j) + l}{\sum_{i=1}^{n} I(y_i = c) + lK_\alpha},
\end{align}
$$

where $x_{i\alpha} = [\mathbf{x}_i]_\alpha$ and $l$ is a smoothing parameter. By setting $l=0$ we get an MLE estimator, $l>0$ leads to MAP. If we set $l= +1$ we get <i>Laplace smoothing</i>.

<p>In words (without the $l$ hallucinated samples) this means
  $$
\frac{\text{# of samples with label c that have feature } \alpha \text{ with value $j$ } }{\text{# of samples with label $c$} }.
$$

<u>Prediction:</u>
$$
\operatorname*{argmax}_y \; P(y=c \mid \mathbf{x}) \propto \operatorname*{argmax}_y \; \hat\pi_c \prod_{\alpha = 1}^{d} [\hat\theta_{jc}]_\alpha
$$

### Case 2: Multinomial Features
<u>Features:</u> 
\begin{align}
x_\alpha \in \{0, 1, 2, \dots, m\} \text{ and } m = \sum_{\alpha = 1}^d x_\alpha 
\end{align}
Each feature $\alpha$ represents a count and m is the length of the sequence. 
An example of this could be the count of a specific word $\alpha$ in a document of length $m$ and $d$ is the size of the vocabulary.
</br> </br>

<u>Model $P(\mathbf{x} \mid y)$:</u>
Use the multinomial distribution
$$
P(\mathbf{x} \mid m, y=c) = \frac{m!}{x_1! \cdot x_2! \cdot \dots \cdot x_d!} \prod_{\alpha = 1}^d
\left(\theta_{\alpha c}\right)^{x_\alpha}
$$
where $\theta_{\alpha c}$ is the probability of selecting $\text{count of word }\alpha = x_\alpha$ and $\sum_{\alpha = 1}^d \theta_{\alpha c} =1$. 
So, we can use this to generate a spam email, i.e., a document $\mathbf{x}$ of class $y = \text{spam}$ by picking $m$ words independently at random from the vocabulary of $d$ words using $P(\mathbf{x} \mid y = \text{spam})$.
</br> </br>

<u>Parameter estimation:</u>

$$
\begin{align}
\hat{\theta^{(2)}_{MLE} } &= \hat\theta_{\alpha c} \\
&= \frac{\sum_{i = 1}^{n} I(y_i = c) x_{i\alpha} + l}{\sum_{i=1}^{n} I(y_i = c) m_i + l \cdot d }
\end{align}
$$
where $m_i=\sum_{\beta = 1}^{d} x_{i\beta}$ denotes the number of words in document $i$. The numerator sums up all counts for feature $x_\alpha$ and the denominator sums up all counts of all features across all data points. E.g.,
$$
\frac{\text{# of times word } \alpha \text{ appears in all spam emails} }{\text{# of words in all spam emails combined} }.
$$
Again, $l$ is the smoothing parameter. Adding $l$ in numerator says I've seen the word $\alpha$ at least $l$ times. Adding $l * d$ in the denominator accounts for seeing each of the $d$ words at least $l$ times
</br> </br>

<u>Prediction:</u>
$$
\operatorname*{argmax}_c \; P(y = c \mid \mathbf{x}) \propto \operatorname*{argmax}_c \; \hat\pi_c \prod_{\alpha = 1}^d \hat\theta_{\alpha c}^{x_\alpha}
$$

**Notice here that our $\frac{m!}{x_1! \cdot x_2! \cdot \dots \cdot x_d!}$ has disappeared because normally we would just calculate $\frac{P(y=\text{ spam} \mid x)}{P(y=\text{ not spam} \mid x)}$, this way both constants cancel out.**

**We observe here that "training" only needs to happen once - we calculate the numerator and denominator. Whenever we see a new document, we can just update the numerator and denominator, which are the only things we have to store for each class (e.g. count of each word $\alpha$ and count of words in each class combined)**

### Case 3: Continuous Features
<table border=0 align=left width=150 cellpadding=5 hspace=4 vspace=4>
	<tr>
<td><img src="http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/naive_bayes/NBGaussian.png" width="150px" align="left" hspace="50" /></td></tr>
<tr><td><center><i>Illustration of Gaussian NB. Each class conditional feature distribution $P(x_\alpha|y)$ is assumed to originate from an independent Gaussian distribution with its own mean $\mu_{\alpha,y}$ and variance $\sigma_{\alpha,y}^2$.  </i></center></td></tr>
</table>

<p>
<u>Features:</u> 
\begin{align}
x_\alpha \in \mathbb{R} && \text{(each feature takes on a real value)}
\end{align}
</p>

<p>
<u>Model $P(x_\alpha \mid y)$:</u> Use Gaussian distribution 

$$
\begin{align}
P(x_\alpha \mid y=c) = \mathcal{N}\left(\mu_{\alpha c}, \sigma^{2}_{\alpha c}\right) = \frac{1}{\sqrt{2 \pi} \sigma_{\alpha c} } e^{-\frac{1}{2} \left(\frac{x_\alpha - \mu_{\alpha c} }{\sigma_{\alpha c} }\right)^2} 
\end{align}
$$

Note that the model specified above is based on our assumption about the data - that each feature $\alpha$ comes from a class-conditional Gaussian distribution. The full distribution $P(\mathbf{x}|y)\sim \mathcal{N}(\mathbf{\mu}_y,\Sigma_y)$, where $\Sigma_y$ is a diagonal covariance matrix with $[\Sigma_y]_{\alpha,\alpha}=\sigma^2_{\alpha,y}$.
</p>

<p>
<u>Parameter estimation:</u>
As always, we estimate the parameters of the distributions for each dimension and class independently. Gaussian distributions only have two parameters, the mean and variance. The mean $\mu_{\alpha,y}$ is estimated by the average feature value of dimension $\alpha$ from all samples with label $y$. The (squared) standard deviation is simply the variance of this estimate. 
$$
\begin{align}
\hat{\theta^{(2a)}_{MLE} } = \mu_{\alpha c} &\leftarrow \frac{1}{n_c} \sum_{i = 1}^{n} I(y_i = c) x_{i\alpha} && \text{where $n_c = \sum_{i=1}^{n} I(y_i = c)$} \\
\hat{\theta^{(2b)}_{MLE} } = \sigma_{\alpha c}^2 &\leftarrow \frac{1}{n_c} \sum_{i=1}^{n} I(y_i = c)(x_{i\alpha} - \mu_{\alpha c})^2
\end{align}
</p>

**Above are more intuitive ways of finding the MLE estimates for the parameters, if you want a more explicit way, refer to [here](https://www.cs.toronto.edu/~urtasun/courses/CSC411/tutorial4.pdf).**



---
# Properties

## Naive Bayes is a Linear Classifier

We start with the multinomial features case:
$$
\begin{aligned}
    P(y \mid \mathbf{x}) &\propto P(\mathbf{x} \mid y) * P(y) \\
    P(y=+1 \mid \mathbf{x}) &> P(y=-1 \mid \mathbf{x}) \\
    P(\mathbf{x} \mid y=+1) * P(y=+1) &> P(\mathbf{x} \mid y=-1) * P(y=-1) \\
    \frac{m!}{x_1! \cdot x_2! \cdot \dots \cdot x_d!} \prod_{\alpha = 1}^d
\left(\theta_{\alpha c=+1}\right)^{x_\alpha} * P(y=+1) &> \frac{m!}{x_1! \cdot x_2! \cdot \dots \cdot x_d!} \prod_{\alpha = 1}^d
\left(\theta_{\alpha c=-1}\right)^{x_\alpha} * P(y=-1) \\
    \log P(y=+1) + \sum_{\alpha = 1}^d
{x_\alpha} \log \left(\theta_{\alpha c=+1}\right) &> \log P(y=-1) + \sum_{\alpha = 1}^d
{x_\alpha} \log \left(\theta_{\alpha c=-1}\right) \\
    \log P(y=+1) - \log P(y=-1) + \sum_{\alpha = 1}^d
{x_\alpha} \log \left(\theta_{\alpha c=+1}\right) - \sum_{\alpha = 1}^d
{x_\alpha} \log \left(\theta_{\alpha c=-1}\right) &> 0 \\
    \underbrace{\log \frac{P(y=+1)}{P(y=-1)} }_{\vec{b} } + \sum_{\alpha = 1}^d {x_\alpha} \underbrace{\log \frac{\left(\theta_{\alpha c=+1}\right)}{\left(\theta_{\alpha c=-1}\right)} }_{\vec{w} } &> 0 \\
    \mathbf{w}^\top\mathbf{x} + b &> 0
\end{aligned}
$$

- Notice that we don't have any loops, therefore unlike the Perceptron, we won't loop forever, because there's no loop!
- Naive Bayes does not find the best hyperplane to separate the positive from negative points, it finds the best hyperplane that separates the **positive distribution** from the **negative distribution**!
    - Because of this, if we do indeed have linearly separable data, but there exists a few positive points that are very close to the negative points and very far away from the rest of the positive points, perceptron will perform better by separating the data, but naive bayes will find a decision boundary nearer to the middle of the two clusters, misclassifying the positive points near the negatives.
    
Now we'll go through the continuous (gaussian) features case:
$$
\begin{aligned}
    P(y=+1\mid x) &= \frac{P(y=+1)*P(x\mid y=+1)}{P(x)} \\
    &= \frac{P(y=+1)*P(x\mid y=+1)}{P(x,y=+1) + P(x,y=-1)} \because \text{ marginalization} \\
    &= \frac{P(y=+1)*P(x\mid y=+1)}{P(y=+1) * P(x \mid y=+1) + P(y=-1) * P(x \mid y=-1)} \\
    &= \frac{1}{1 + \frac{P(y=-1) * P(x \mid y=-1)}{P(y=+1) * P(x \mid y=+1)} } \\
    &= \frac{1}{1 + {e}^{\ln\frac{P(y=-1) * P(x \mid y=-1)}{P(y=+1) * P(x \mid y=+1)} } } \\
    &= \frac{1}{1 + {e}^{\ln\frac{P(y=-1)}{P(y=+1)} + \ln\frac{P(x \mid y=-1)}{P(x \mid y=+1)} } } \\
    &= \frac{1}{1 + {e}^{\ln\frac{P(y=-1)}{P(y=+1)} + \ln\prod_{i=1}^{n}\frac{P(x_i \mid y=-1)}{P(x_i \mid y=+1)} } } \\
    &= \frac{1}{1 + {e}^{\ln\frac{P(y=-1)}{P(y=+1)} + \sum_{i=1}^{n}\ln\frac{P(x_i \mid y=-1)}{P(x_i \mid y=+1)} } } \\
    &= \frac{1}{1 + {e}^{\ln\frac{P(y=-1)}{P(y=+1)} + \ldots } } \\
    &= \vdots \\
    &= \frac{1}{1 + e^{yw^\top x} }
\end{aligned}
$$

Find full derivation [here](https://www.cs.toronto.edu/~urtasun/courses/CSC411/tutorial4.pdf), Naive Bayes is only a linear classifier if the class conditional covariance matrices are the same for both classes and the likelihood probabilities $P(\mathbf{x}_i \mid y)$ come from exponential families. Otherwise, it'll be quadratic.



---
## Resources:
- [Clear explanation of MLE estimates of Naive Bayes with EM to combat missing values](http://www.cs.columbia.edu/~mcollins/em.pdf)
- [Naive Bayes Class conditional independence](https://www.saedsayad.com/naive_bayesian.htm)
- [Kilian Weinberger's Naive Bayes lecture](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html)
- [CMU Naive Bayes](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)
- [Naive Bayes with different feature distributions](https://www.quora.com/How-can-Should-I-create-a-Naive-Bayes-model-with-different-feature-distributions)
- [When is Naive Bayes a linear classifier?](https://stats.stackexchange.com/questions/142215/how-is-naive-bayes-a-linear-classifier)

