---
interact_link: content/machine-learning/01-supervised-learning/classification/perceptron.ipynb
kernel_name: python3
has_widgets: false
title: 'Perceptron'
prev_page:
  url: /machine-learning/01-supervised-learning/classification/k-nearest-neighbours
  title: 'K-Nearest Neighbours'
next_page:
  url: /machine-learning/01-supervised-learning/classification/svm-inductive-derivation
  title: 'SVM (Inductive Derivation)'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Perceptron

In this algorithm cookbook, we'll go over the perceptron algorithm.

$$h(x_i) = \textrm{sign}(\mathbf{w}^\top \mathbf{x}_i + b) \rightarrow \textrm{sign}(\mathbf{w}^\top \mathbf{x}_i) \text{ by making } \mathbf{x}_i \leftarrow \begin{bmatrix} \mathbf{x}_i \\ 1  \end{bmatrix} \text{ and } \mathbf{w} \leftarrow \begin{bmatrix} \mathbf{w} \\ b  \end{bmatrix} \\ $$ (*$h$ stands for hyopothetis and this "absorbing" of the bias term just increases the dimension that we're working on by 1, e.g. originally our data was in 2-D, and now it's shifted in the z-axis to live in 3-D*)

![adding_dimension_perceptron][adding_dimension_perceptron]

[adding_dimension_perceptron]: http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/Perceptron/PconstantDim.png "adding_dimension_perceptron"



---
# Brief Overview

What is it used for?
- Classification

Assumption: 
- It is a binary classification problem
- Data is linearly separable



---
# Training

1. Initialize ${\mathbf{w} }$ as ${\vec{0} }$
2. Calculate the un-normalized cosine similarities / inner product / dot product between ${\mathbf{w}^\top}$ and every other data point ${\mathbf{w}^\top \cdot \mathbf{x}_i}$.
$$
\text{similarity} = 
\cos(\theta)= 
\frac{ 
\left\langle
    \begin{bmatrix} w_0 \\ \vdots \\ w_n \\ b \end{bmatrix},
    \begin{bmatrix} x_0 \\ \vdots \\ x_n \\ 1 \end{bmatrix}
\right\rangle =
{\mathbf{w}^\top \cdot \mathbf{x}_i} = \sum_{i=1}^n w_i x_i = w_0 x_0 + \cdots + w_n x_n }
{ {\vert \mathbf{w}\vert }_2 {\vert \mathbf{x}\vert }_2 = \sqrt{\sum\limits_{i=1}^{n}{w_i^2} }\sqrt{\sum\limits_{i=1}^{n}{x_i^2} } }$$

3. Check what the value of $y_i \mathbf{w}^\top \mathbf{x}_i$ is. If datapoint is correctly classified, $y_i(\mathbf{w}^\top \mathbf{x}_i) > 0$, while misclassified points are $y_i(\mathbf{w}^\top \mathbf{x}_i) \leq 0$
    - For each misclassified datapoint ${(y_i = \{+1 / -1\}) \cdot (\mathbf{w}^\top = \begin{bmatrix} w_0, w_1 ... b \end{bmatrix}) \cdot (\mathbf{x}_i} = \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ 1 \end{bmatrix}) \leq 0$, we update $\mathbf{w}$ by 
$$\mathbf{w} \leftarrow \mathbf{w} + y_i\mathbf{x}_i$$
    - When we classify a negative point as positive, it means the un-normalized cosine similarity / inner product / dot product ${\mathbf{w}^\top \cdot \mathbf{x}_i}$ is falsely positive (vectors are pointing in the same direction when they are supposed to be in opposite directions). To correct this, we __add__ $\mathbf{x}_i$ in the __opposite__ direction to increase the angle between vectors > 90 degrees and correctly classify it in the next iteration.
    - When we classify a positive point as negative, it means the un-normalized cosine similarity / inner product / dot product ${\mathbf{w}^\top \cdot \mathbf{x}_i}$ is falsely negative (vectors are pointing in the opposite direction when they are supposed to be in the same direction). To correct this, we __add__ $\mathbf{x}_i$ in the __same__ direction to decrease the angle between vectors < 90 degrees and correctly classify it in the next iteration.
4. Stop iteration when all points are correctly classified.

![perceptron update][perceptron_geometry]

(*Unlike SVMs, the perceptron finds __any__ hyperplane that linearly separates the positive and negative data points, not necessarily the __best__ one (which maximizes the margin between the hyperplane and the closest positive and negative data points).*)

[perceptron_geometry]: http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/Perceptron/PerceptronUpdate.png "perceptron_geometry"



---
# Prediction

Just plug in ${\mathbf{x}_{test} }$ with the trained / fitted ${\mathbf{w}^\top}$ in ${\hat{y} = \mathbf{w}^\top \cdot \mathbf{x}_{test} }$




---
# Problems
- Algorithm will never converge if data isn't linearly separable.



---
# Implementation




---
# Advantages

- Convergence Guarantee
    - We can prove that if the data set is linearly separable, we can find a hyperplane that separates the positive and negative data points by at most ${\frac{1}{\gamma^2} }$ iterations, where ${\gamma}$ is the minimum distance from hyperplane ${\mathbf{w}^*}$ to the closest data point, $\gamma = \min_{(\mathbf{x}_i, y_i) \in D}\mid\mathbf{x}_i^\top \mathbf{w}^*\mid > 0$.
        1. Suppose $\exists \mathbf{w}^*: y_i(\mathbf{x}^\top \mathbf{w}^*  ) > 0 \forall (\mathbf{x}_i, y_i) \in D$ (*One of the many hyperplanes that linearly separates the positive and negative datapoints. Recall also that $y_i(\mathbf{w}^\top \mathbf{x}_i) > 0$ means correctly classified and $y_i(\mathbf{w}^\top \mathbf{x}_i) \leq 0$ means incorrectly classified*)
        2. Rescale __(A).__ $\mathbf{w}^*$ and __(B).__ all the $\mathbf{x}_i$ so that $\large{\vert\vert\mathbf{w}^*\vert\vert = {1} } \text{ and } \mid\mid\mathbf{x}_i\mid\mid \le 1\forall \mathbf{x}_i \in D$ by $\mathbf{x}_i \leftarrow \frac{\mathbf{x}_i}{\max_j{\mid\mid\mathbf{x}_j\mid\mid} }$.
        3. Update $\mathbf{w} \leftarrow \mathbf{w} + y_i\mathbf{x}_i$ when $y_i(\mathbf{w}^\top \mathbf{x}_i) \leq 0$.
            1. To measure how much of an effect the update has made, we have to consider $\mathbf{w}^\top \mathbf{w}^*$ (This value increases if we correctly made an update and will be highest and angle between vectors $\theta=0$, AKA $= {\mid\mathbf{w}^*\mid}^2_2$), and ensure that $\mathbf{w}^\top \mathbf{w}$ does not grow really fast because $\mathbf{w}^\top \mathbf{w}^*$ can trivially increase by scaling, i.e. $\mathbf{w} \leftarrow 2 * \mathbf{w}$.
            2. After one update:
                1. $\mathbf{w}^\top \mathbf{w}^* \leftarrow ({\mathbf{w} + y\mathbf{x} })^\top \mathbf{w}^*$
                    - $= \mathbf{w}^\top \mathbf{w}^* + y(\mathbf{x}^\top  \mathbf{w}^*) \ge \mathbf{w}^\top \mathbf{w}^* + \gamma$, because we defined $\gamma = \min_{(\mathbf{x}_i, y_i) \in D}\mid\mathbf{x}_i^\top \mathbf{w}^*\mid > 0$. Also, $y(\mathbf{x}^\top  \mathbf{w}^*) > 0$ because we defined $\mathbf{w}^*$ as a hyperplane that linearly separates the positive and negative datapoints. 
                    - Therefore, after one update, our $\mathbf{w}^\top \mathbf{w}^*$ grows by __at least__ $\gamma$.
$$\mathbf{w}^\top\mathbf{w}^*\geq M\gamma$$
                - $\mathbf{w}^\top \mathbf{w} \leftarrow (\mathbf{w} + y\mathbf{x})^\top   (\mathbf{w} + y\mathbf{x})$ 
                    - $= \mathbf{w}^\top \mathbf{w} + \underbrace{2y(\mathbf{w}^\top\mathbf{x})}_{<0} + \underbrace{y^2(\mathbf{x}^\top  \mathbf{x})}_{0\leq \ \ \leq 1} \le \mathbf{w}^\top \mathbf{w} + 1$, because $2y(\mathbf{w}^\top  \mathbf{x}) < 0$ as we only updated when $\mathbf{x}$ was misclassified, making $y_i(\mathbf{w}^\top \mathbf{x}_i) \leq 0$, and that $0\leq y^2(\mathbf{x}^\top  \mathbf{x}) \le 1$ as $y^2 = 1$ and $\mathbf{x}^\top  \mathbf{x}\leq 1$ because we rescaled it as such in Step 2.
                    - Therefore, after one update, our $\mathbf{x}^\top  \mathbf{x}$ grows by __at most__ 1.
$$\mathbf{w}^\top \mathbf{w}\leq M$$
        4. After $M$ updates,
        $$
        \begin{align}
        M\gamma &\le \mathbf{w}^\top \mathbf{w}^* &&\text{By (1)} \\
        &= \mid\mathbf{w}^\top \mathbf{w}^*\mid &&\text{Simply because $M\gamma \geq 0$} \\
        &\le \mid\mid\mathbf{w}\mid\mid\  \mid\mid\mathbf{w}^*\mid\mid &&\text{By Cauchy-Schwartz inequality$^*$} \\
        &= \mid\mid\mathbf{w}\mid\mid &&\text{As $\mid\mid\mathbf{w}^*\mid\mid = 1$} \\
        &= \sqrt{\mathbf{w}^\top \mathbf{w} } && \text{by definition of $\mid\mathbf{w}\mid$} \\
        &\le \sqrt{M} &&\text{By (2)} \\ 
        & \textrm{ }\\
        &\Rightarrow M\gamma \le \sqrt{M} \\
        &\Rightarrow M^2\gamma^2 \le M \\
        &\Rightarrow M \le \frac{1}{\gamma^2} && \text{And hence, the number of updates $M$ is bounded from above by a constant.}
        \end{align}
        $$
        
![rescaled_perceptron][rescaled_perceptron]

[rescaled_perceptron]: http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/perceptron/perceptron_img3.png "rescaled_perceptron"



---
## Resources:
- [Kilian Weinberger's Perceptron lecture](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote03.html)

