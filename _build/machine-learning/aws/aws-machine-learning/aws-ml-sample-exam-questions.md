---
interact_link: content/machine-learning/aws/aws-machine-learning/aws-ml-sample-exam-questions.ipynb
kernel_name: python3
has_widgets: false
title: 'Sample Exam Questions'
prev_page:
  url: /machine-learning/aws/aws-machine-learning/aws-machine-learning-specialty-exam
  title: 'AWS Machine Learning Specialty Exam'
next_page:
  url: /machine-learning/miscellaneous-topics/README
  title: 'Miscellaneous Topics'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# AWS Machine Learning  - Specialty Sample Exam Questions

In this notebook, we will go through the sample exam questions provided [here](https://d1.awsstatic.com/training-and-certification/docs-ml/AWS%20Certified%20Machine%20Learning%20-%20Specialty_Sample%20Questions.pdf).



---
## 1.

**A Machine Learning team has several large CSV datasets in Amazon S3. Historically, models built with the Amazon SageMaker Linear Learner algorithm have taken hours to train on similar-sized datasets. The team’s leaders need to accelerate the training process.**

**What can a Machine Learning Specialist do to address this concern?**

- A. Use Amazon SageMaker Pipe mode.
- B. Use Amazon Machine Learning to train the models.
- C. Use Amazon Kinesis to stream the data to Amazon SageMaker.
- D. Use AWS Glue to transform the CSV dataset to the JSON format.

A - Amazon SageMaker Pipe mode streams the data directly to the container, which improves the performance of training jobs. (Refer to this [link](https://aws.amazon.com/blogs/machine-learning/now-use-pipe-mode-with-csv-datasets-for-faster-training-on-amazon-sagemaker-built-in-algorithms/) for supporting information.) In Pipe mode, your training job
streams data directly from Amazon S3. Streaming can provide faster start times for training jobs and better throughput. With Pipe mode, you also reduce the size of the Amazon EBS volumes for your training instances. B would not apply in this scenario. C is a streaming ingestion solution, but is not applicable in this scenario. D transforms the data structure.

[Using Pipe mode VS File mode](https://aws.amazon.com/blogs/machine-learning/accelerate-model-training-using-faster-pipe-mode-on-amazon-sagemaker/)
```
There are a few situations where Pipe mode may not be the optimum choice for training. In that case you should stick to using File mode:

If your algorithm needs to backtrack or skip ahead within an epoch. This isn’t possible in Pipe mode because the underlying FIFO cannot support lseek() operations.
If your training dataset is small enough to fit in memory and you need to run multiple epochs. In this case it might be quicker and easier just to load it all into memory and iterate.
If it is not easy to parse your training dataset from a streaming source.
In all other scenarios, if you have an I/O-bound training algorithm, switching to Pipe mode should give you a significant throughput-boost as well as reduce the size of the disk volume required. This should result in both saving you time and reducing training costs.
```



---
# 2.

**A term frequency–inverse document frequency (tf–idf) matrix using both unigrams and bigrams is built from a text corpus consisting of the following two sentences:**

1. **Please call the number below.**
2. **Please do not call us.**

**What are the dimensions of the tf–idf matrix?**

- A. (2, 16)
- B. (2, 8)
- C. (2, 10)
- D. (8, 10)

A - There are 2 sentences, 8 unique unigrams, and 8 unique bigrams, so the result would be (2,16). The phrases are “Please call the number below” and “Please do not call us.” Each word individually (unigram)
is “Please,” “call,” ”the,” ”number,” “below,” “do,” “not,” and “us.” The unique bigrams are “Please call,” “call the,” ”the number,” “number below,” “Please do,” “do not,” “not call,” and “call us.” The tf–idf
vectorizer is described at this [link](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).



---
# 3.

**A company is setting up a system to manage all of the datasets it stores in Amazon S3. The company would like to automate running transformation jobs on the data and maintaining a catalog of the
metadata concerning the datasets. The solution should require the least amount of setup and maintenance.**

**Which solution will allow the company to achieve its goals?**

- A. Create an Amazon EMR cluster with Apache Hive installed. Then, create a Hive metastore and a script to run transformation jobs on a schedule.
- B. Create an AWS Glue crawler to populate the AWS Glue Data Catalog. Then, author an AWS Glue ETL job, and set up a schedule for data transformation jobs.
- C. Create an Amazon EMR cluster with Apache Spark installed. Then, create an Apache Hive metastore and a script to run transformation jobs on a schedule.
- D. Create an AWS Data Pipeline that transforms the data. Then, create an Apache Hive metastore and a script to run transformation jobs on a schedule

B - AWS Glue is the correct answer because this option requires the least amount of setup and maintenance since it is serverless, and it does not require management of the infrastructure. Refer to this [link](https://aws.amazon.com/glue/) for supporting information. A, C, and D are all solutions that can solve the problem, but require more steps for configuration, and require higher operational overhead to run and maintain.



---
# 4.

**A Data Scientist is working on optimizing a model during the training process by varying multiple parameters. The Data Scientist observes that, during multiple runs with identical parameters, the loss function converges to different, yet stable, values.**

**What should the Data Scientist do to improve the training process?**

- A. Increase the learning rate. Keep the batch size the same.
- B. Reduce the batch size. Decrease the learning rate.
- C. Keep the batch size the same. Decrease the learning rate.
- D. Do not change the learning rate. Increase the batch size.

B - It is most likely that the loss function is very curvy and has multiple local minima where the training is getting stuck. Decreasing the batch size would help the Data Scientist stochastically get out of the local minima saddles. Decreasing the learning rate would prevent overshooting the global loss function minimum. Refer to the paper at this [link](https://arxiv.org/pdf/1609.04836.pdf) for an explanation.

### On Large-Batch Training for deep learning: Generalization Gap and sharp minima Summary

Firstly, SGD = minibatch gradient descent when batch size = 1 and that sample is randomly chosen. Batch gradient descent = minibatch gradient descent when batch size = m (total size of training set). What changes with this is how we average the losses. 
In BGD - we divide the sum of losses of each sample by m
In SGD - we dont divide the loss, just use it as is
In MBGD - we divide it by the batch size

Motivation:
- SGD uses a small sample of training data to compute approx. gradient 
- Using larger batch degrades quality of model
- Paper aims to support that large batch decreases generalizability of model because they tend to converge to sharp minimas

Problem with SGD:
- Training neural nets is a **non-convex** optimization problem
- Because of sequential nature of SGD, there is limited opportunity for parallelization, decreasing our ability to speed up training
- Using large batch sizes will increase parallelizability, but we risk the model overfitting and hence not being able to generalize

Main observation:
- The lack of generalization ability is due to the fact that large-batch methods tend to converge to sharp minimizers of the training function.
- These minimizers are characterized by a significant number of large positive eigenvalues in ${\nabla}^{2}f(x)$, and tend to generalize less well. In contrast, small-batch methods converge to flat minimizers characterized by having numerous small eigenvalues of ${\nabla}^{2}f(x)$. 
- We have observed that the loss function landscape of deep neural networks is such that large-batch methods are attracted to regions with sharp minimizers and that, unlike small-batch methods, are unable to escape basins of attraction of these minimizers.

How to improve large-batch methods:
- Data augmentation to regularize the network
- Adding a [conservative regularizer](https://www.cs.cmu.edu/~muli/file/minibatch_sgd.pdf) to objective function for mini-batch
- Robust Training (e.g. Adversarial Training)
- [Optimizing gradient descent](http://ruder.io/optimizing-gradient-descent/index.html)



---
# 5.

**A Data Scientist is evaluating different binary classification models. A false positive result is 5 times more expensive (from a business perspective) than a false negative result.**

**The models should be evaluated based on the following criteria:**

1. Must have a recall rate of at least 80%
2. Must have a false positive rate of 10% or less
3. Must minimize business costs

**After creating each binary classification model, the Data Scientist generates the corresponding confusion matrix.**

**Which confusion matrix represents the model that satisfies the requirements?**

- A. 
    - TN = 91, FP = 9
    - FN = 22, TP = 78
- B. 
    - TN = 99, FP = 1
    - FN = 21, TP = 79
- C. 
    - TN = 96, FP = 4
    - FN = 10, TP = 90
- D. 
    - TN = 98, FP = 2
    - FN = 18, TP = 82
 
D - The following calculations are required:

- Recall = TP / (TP + FN)
- False Positive Rate (FPR) = FP / (FP + TN)
- Cost = 5 * FP + FN

| Metric | A | B | C | D |
|:------:|:-:|:-:|:-:|:-:|
| Recall | 78 / (78 + 22) = 0.78 | 79 / (79 + 21) = 0.79 | 90 / (90 + 10) = 0.9 | 82 / (82 + 18) = 0.82 |
| False Positive Rate | 9 / (9 + 91) = 0.09 | 1 / (1 + 99) = 0.01 | 4 / (4 + 96) = 0.04 | 2 / (2 + 98) = 0.02 |
| Costs | 5 * 9 + 22 = 67 | 5 * 1 + 21 = 26 | 5 * 4 + 10 = 30 | 5 * 2 + 18 = 28 |

Options C and D have a recall greater than 80% and an FPR less than 10%, but D is the most cost effective. For supporting information, refer to this [link](https://docs.aws.amazon.com/machine-learning/latest/dg/binary-model-insights.html).



---
# 6.

**A Data Scientist uses logistic regression to build a fraud detection model. While the model accuracy is 99%, 90% of the fraud cases are not detected by the model.**

**What action will definitively help the model detect more than 10% of fraud cases?**

- A. Using undersampling to balance the dataset
- B. Decreasing the class probability threshold
- C. Using regularization to reduce overfitting
- D. Using oversampling to balance the dataset

B - Decreasing the class probability threshold makes the model more sensitive and, therefore, marks more cases as the positive class, which is fraud in this case. This will increase the likelihood of fraud detection. However, it comes at the price of lowering precision. This is covered in the Discussion section of the paper at this [link](https://academic.oup.com/bib/article/14/1/13/304457).



---
# 7.

**A company is interested in building a fraud detection model. Currently, the Data Scientist does not have a sufficient amount of information due to the low number of fraud cases.**

**Which method is MOST likely to detect the GREATEST number of valid fraud cases?**

- A. Oversampling using bootstrapping
- B. Undersampling
- C. Oversampling using SMOTE
- D. Class weight adjustment

C – With datasets that are not fully populated, the Synthetic Minority Over-sampling Technique (SMOTE) adds new information by adding synthetic data points to the minority class. This technique would be the
most effective in this scenario. Refer to Section 4.2 at this [link](https://www.jair.org/index.php/jair/article/view/10302) for supporting information.



---
# 8.

**A Machine Learning Engineer is preparing a data frame for a supervised learning task with the Amazon SageMaker Linear Learner algorithm. The ML Engineer notices the target label classes are highly imbalanced and multiple feature columns contain missing values. The proportion of missing values across the entire data frame is less than 5%.**

**What should the ML Engineer do to minimize bias due to missing values?**

- A. Replace each missing value by the mean or median across non-missing values in same row.
- B. Delete observations that contain missing values because these represent less than 5% of the data.
- C. Replace each missing value by the mean or median across non-missing values in the same column.
- D. For each feature, approximate the missing values using supervised learning based on other features.

D – Use supervised learning to predict missing values based on the values of other features. Different supervised learning approaches might have different performances, but any properly implemented supervised learning approach should provide the same or better approximation than mean or median approximation, as proposed in responses A and C. Supervised learning applied to the imputation of missing values is an active field of research. Refer to this [link](https://www.omicsonline.org/open-access/a-comparison-of-six-methods-for-missing-data-imputation-2155-6180-1000224.php?aid=54590) for an example.



---
# 9.

**A company has collected customer comments on its products, rating them as safe or unsafe, using decision trees. The training dataset has the following features: id, date, full review, full review summary, and a binary safe/unsafe tag. During training, any data sample with missing features was dropped. In a few instances, the test set was found to be missing the full review text field.**

**For this use case, which is the most effective course of action to address test data samples with missing features?**

- A. Drop the test samples with missing full review text fields, and then run through the test set.
- B. Copy the summary text fields and use them to fill in the missing full review text fields, and then run through the test set.
- C. Use an algorithm that handles missing data better than decision trees.
- D. Generate synthetic data to fill in the fields that are missing data, and then run through the test set.

B – In this case, a full review summary usually contains the most descriptive phrases of the entire review
and is a valid stand-in for the missing full review text field. For supporting information, refer to page 1627
at this [link](http://jmlr.csail.mit.edu/papers/volume8/saar-tsechansky07a/saar-tsechansky07a.pdf), and this [link](https://www.annualreviews.org/doi/10.1146/annurev.publhealth.25.102802.124410) and this [link](https://docs.aws.amazon.com/machine-learning/latest/dg/feature-processing.html).



---
# 10.

**An insurance company needs to automate claim compliance reviews because human reviews are expensive and error-prone. The company has a large set of claims and a compliance label for each. Each claim consists of a few sentences in English, many of which contain complex related information. Management would like to use Amazon SageMaker built-in algorithms to design a machine learning supervised model that can be trained to read each claim and predict if the claim is compliant or not.**

**Which approach should be used to extract features from the claims to be used as inputs for the downstream supervised task?**

- A. Derive a dictionary of tokens from claims in the entire dataset. Apply one-hot encoding to tokens found in each claim of the training set. Send the derived features space as inputs to an Amazon SageMaker builtin supervised learning algorithm.
- B. Apply Amazon SageMaker BlazingText in Word2Vec mode to claims in the training set. Send the derived features space as inputs for the downstream supervised task.
- C. Apply Amazon SageMaker BlazingText in classification mode to labeled claims in the training set to derive features for the claims that correspond to the compliant and non-compliant labels, respectively.
- D. Apply Amazon SageMaker Object2Vec to claims in the training set. Send the derived features space as inputs for the downstream supervised task.

D – Amazon SageMaker Object2Vec generalizes the Word2Vec embedding technique for words to more complex objects, such as sentences and paragraphs. Since the supervised learning task is at the level of
whole claims, for which there are labels, and no labels are available at the word level, Object2Vec needs be used instead of Word2Vec. For supporting information, refer to this [link](https://docs.aws.amazon.com/sagemaker/latest/dg/object2vec.html) and this [link](https://aws.amazon.com/blogs/machine-learning/introduction-to-amazon-sagemaker-object2vec/).

