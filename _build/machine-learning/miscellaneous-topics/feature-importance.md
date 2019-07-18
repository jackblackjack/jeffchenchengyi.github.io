---
interact_link: content/machine-learning/miscellaneous-topics/feature-importance.ipynb
kernel_name: python3
has_widgets: false
title: 'Feature Importance'
prev_page:
  url: /machine-learning/miscellaneous-topics/handling-class-imbalance
  title: 'How do we handle Class Imbalance?'
next_page:
  url: /machine-learning/miscellaneous-topics/handling-missing-data
  title: 'How do we handle Missing Data?'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Feature Importance



---
## Permutation Importance

Steps:
1. Get a trained model.

2. Shuffle the values in a single column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.

3. Return the data to the original order (undoing the shuffle from step 2). Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# import eli5
# from eli5.sklearn import PermutationImportance

# perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
# eli5.show_weights(perm, feature_names = val_X.columns.tolist())

```
</div>

</div>



---
## Resources:
- [Dans Becker on Permutation Importance](https://www.kaggle.com/dansbecker/permutation-importance)

