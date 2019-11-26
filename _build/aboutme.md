---
interact_link: content/aboutme.ipynb
kernel_name: python3
has_widgets: false
title: 'About Me'
prev_page:
  url: /intro.html
  title: 'Home'
next_page:
  url: https://github.com/jeffchenchengyi/jeffchenchengyi.github.io
  title: 'GitHub repository'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# About Me





---
### Hi there, I'm Jeff!

<img src='./images/header.jpg' style='border: 5px solid black; border-radius: 5px;' />



---
### Who am I?

I am a rising junior @ University of Southern California, majoring in B.S. Computer Science and Business and pursuing an M.S. in Analytics. I love learning anything related to data science / machine learning / artificial intelligence.

<img src='https://16mhpx3atvadrnpip2kwi9or-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/USC-Viterbi-School-of-Engineering.png' style='border: 5px solid black; border-radius: 5px;' />



---
### Where I'm From?
I come from sunny Singapore.

<img src='https://media.giphy.com/media/AyIYq8NfX07sc/giphy.gif' style='border: 5px solid black; border-radius: 5px;' />



---
### What's my favourite song?



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from IPython.display import HTML

HTML('<iframe width="710" height="399" src="https://www.youtube.com/embed/5CorhbkYWGg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<iframe width="710" height="399" src="https://www.youtube.com/embed/5CorhbkYWGg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>


</div>
</div>
</div>



---
### What do I dream to be?
"That" guy that can solve *any* problem using technology and explain *anything* given a reasonable amount of time.



---
### Purpose of this Blog
I started my journery watching simple tutorials about the **intuition** behind the more "introductory" machine learning algorithms such as linear regression, logistic regression, ..., support vector machines... But soon I realized that the "**intuition**" that I got from those videos were really purely "**intuition**". There is so much math (overlaps between statistics, probability theory, linear algebra, multivariable calculus, optimization theory, ...) underlying all those "simple" algorithms and each of them can be derived from multiple perspectives (simple linear regression can be thought of as $\hat{y} = r(\frac{s_y}{s_x})x + [\bar{y} - r(\frac{s_y}{s_x})\bar{x}]$) where $s$ is the sample standard deviation and $r$ is the pearson correlation in the eyes of the statistician, but when there are multiple independent variables, $y = \hat{\beta}x$, where $\hat{\beta} = {(X^\top X)}^{-1} X^\top y$ (MLE estimate / OLS solution) in the probabilistic perspective by assuming the $y$s are drawn from a gaussian / normal distribution, and even with $y = \hat{\beta}x$, there's also a geometric interpretation / derivation through linear algebra by projecting $y$ orthogonally onto the column space of $x$. Given that there are so many different but overlapping concepts, I really wanted to curate a notebook filled with all my notes to help me track the relationships between concepts and organize the resources that have helped me along the way in this journey to master data science. Hence, this will be less of a *"Here's an explanantion of what XXX is"* and more of a *"Here's how XXX is organized, some of my own notes and a bunch of resources that have helped me"* kinda thing ...

---
### Random Thoughts
After about a year of diving into Machine Learning with basically 0 experience except for some programming / high school math, I've realized just how wide and deep your skillsets have to be in order to become an **Elite** data scientist / machine learning / AI person. During this journey, I've learnt that the toughest part of learning data science or any subject for that matter is really about finding out **just how much you don't know you don't know**. However, we'll take it one step at a time. Data science / Machine Learning / Artificial Intelligence is extremely broad. So what do you actually need to be an **Elite** data scientist / machine learning / AI person?

