---
interact_link: content/machine-learning/miscellaneous-topics/handling-missing-data.ipynb
kernel_name: python3
has_widgets: false
title: 'How do we handle Missing Data?'
prev_page:
  url: /machine-learning/miscellaneous-topics/feature-importance
  title: 'Feature Importance'
next_page:
  url: /machine-learning/miscellaneous-topics/linear-algebra-review
  title: 'Linear Algebra Review'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# How can we handle Missing Data?

In this notebook we'll provide an overview of the techniques and APIs used for imputing missing data.



---
# Missing Data Mechanisms

Why do we have missing data in the first place?

### 1. Structurally missing data
- Structurally missing data is data that is missing for a logical reason.

### 2. Missing completely at random (MCAR)
- Assumes that whether or not the person has missing data is completely unrelated to the other information in the data.

### 3. Missing at random (MAR)
- Assumes that we can predict the value that is missing based on the other data.

### 4. Missing not at random (nonignorable)
- E.g., people with very low incomes and very high incomes tend to refuse to answer. Or there could be some other reason we just do not know. Hence, we cannot use any of the standard methods for dealing with missing data (e.g., imputation, or algorithms specifically designed for missing values).



---
# Missing Data Patterns

What cases of missing data exist?

Little and Rubin (2002) classify these into the following techincal categories.

We shall illustrate with a case of cross-classification of Sex, Race, Admission and Department, $S$, $R$, $A$, $D$.

### 1. Univariate:
- $Mij = 0$ unless $j = j*$, e.g. an unmeasured response. 
- Example: $R$ unobserved for some, but data otherwise complete.

### 2. Multivariate:
- $Mij = 0$ unless $j \in J \subset V$ , as above, just with multivariate response, e.g. in surveys. 
- Example: For some subjects, both $R$ and $S$ unobserved.

### 3. Monotone:
- There is an ordering of $V$ so $M_{ik} = 0$ implies $M_{ij} = 0$ for $j < k$, e.g. drop-out in longitudinal studies. 
- Example: For some, $A$ is unobserved, others neither $A$ nor $R$, but data otherwise complete.

### 4. Disjoint: Two subsets of variables never observed together. Controversial. Appears in Rubin’s causal model.
- Example: $S$ and $R$ never both observed.

### 5. General:
- none of the above. Haphazardly scattered missing values. 
- Example: $R$ unobserved for some, $A$ unobserved for others, $S$, $D$ for some.

### 6. Latent:
- A certain variable is never observed. Maybe it is even unobservable. 
- Example: $S$ never observed, but believed to be important for explaining the data.



---
# Methods to deal with Missing Data

### 1. Complete case analysis
- analyse only cases where all variables are observed. Can be adequate if most cases are present, but will generally give serious biases in the analysis. In survey’s, for example, this corresponds to making inference about the population of responders, not the full population

### 2. Weighting methods
- For example, if a population total µ = E(Y ) should be estimated and unit i has been selected with probability πi a standard method is the Horwitz–Thompson estimator
$$
$$
To correct for non-response, one could let ρi be the
response-probability, estimate this in some way as ρˆi
and then let
$$

$$

### 3. Imputation methods
- Find ways of estimating the values of the unobserved values as Yˆmis, then proceed as if there were complete data. Without care, this can give misleading results, in particular because the ”sample size” can be grossly overestimated.
    1. Listwise (complete case) deletion
    2. Single Imputation
        - Hot Deck (Last Observation Carried Forward)
            - Fill missing value with last observed value
        - Regression
        - Recommender System Methods
            - Matrix Factorization
    3. Multiple Imputation by Chained Equations 
    
### 4. Model-based likelihood methods
- Model the missing data mechanism and then proceed to make a proper likelihood-based analysis, either via the method of maximum-likelihood or using Bayesian methods. This appears to be the most sensible way.
- Typically this approach was not computationally feasible in the past, but modern algorithms and computers have changed things completely. Ironically, the efficient algorithms are indeed based upon imputation of missing values, but with proper corrections resulting.



---
# 4. Model-based likelihood methods

## Expectation Maximization

### Expectation Maximization: Naive Bayes with Missing Values



<a id='resources'></a>

---
## Resources
- [Different Types of Missing Data](https://www.displayr.com/different-types-of-missing-data/)
- [Types of Data Imputation](https://en.wikipedia.org/wiki/Imputation_(statistics))
- Visualization of Missing Data
    - [missingno](https://github.com/ResidentMario/missingno)
- Handling Missing Data
    - [fancyimpute](https://github.com/iskandr/fancyimpute)
- [Missing Data and the EM Algorithm](http://www.stats.ox.ac.uk/~steffen/teaching/fsmHT07/fsm407c.pdf)
- [EM with Naive Bayes](http://www.cs.columbia.edu/~mcollins/em.pdf)

