# Capstone
Capstone Project for PCMLAI Berkeley Haas

This project aims to build and compare classification models to predict whether a client will default on their credit card payment.

Files for PCMLAI Assignment 20.1 (“Capstone: Initial Report and EDA”)

## Description of files in repository
* 'train.csv' –the database containing information on client attributes, along with values of the target variable ('credit_card_default').
* 'CapstoneAssignment20_1DelilMartinez.ipynb' -- Jupyter Notebook containing the code and analysis for the project
* 'OptimalDecisionTree.png' – image of the resulting optimal decision tree (the best classifier)


**Link to Jupyter notebook**: [Capstone Jupyter Notebook](https://github.com/delilx/Capstone/blob/main/CapstoneAssignment20_1_DelilMartinez.ipynb )

# Credit Default Prediction
## Project Description
The objective of the project will be to build the best possible classification model to predict whether a person will default on a loan, based on a set of personal and contextual attributes.
The data set used was provided by American Express in the context of a contest known as AmExpert 2021 Code Lab, and it contains information on a variety of client attributes, along with whether they default on their credit card payment.
The purpose of the project is to work with and compare a variety of classification models that make use of the explanatory features to predict the default variable.

## Data and Task Understanding 
The file train.csv was obtained from **AmExpert 2021 Code Lab** (https://www.kaggle.com/datasets/pradip11/amexpert-codelab-2021/data), and it contains the following features:

```
Column Name	Description
1. customer_id: unique identification of customer
2. name: name of customer
3. age:	age of customer (Years)
4. gender: gender of customer (M or F)
5. owns_car: whether a customer owns a car (Y or N)
6. owns_house: whether a customer owns a house (Y or N)
7. no_of_children: number of children of a customer
8. net_yearly_income: net yearly income of a customer (USD)
9. no_of_days_employed: no. of days employed
10. occupation_type: occupation type of customer
11. total_family_members: no. of family members of customer
12. migrant_worker: customer is migrant worker (Yes or No)
13. yearly_debt_payments:	yearly debt of customer (USD)
14. credit_limit:	credit limit of customer (USD)
15. credit_limit_used(%):	credit limit used by customer
16. credit_score: credit score of customer
17. prev_defaults: no. of previous defaults
18. default_in_last_6months: whether a customer has defaulted (Yes or No)
19. credit_card_default: whether there will be credit card default (Yes or No)  **target variable**
```

In preparation for the analysis, rows with missing values (an extremely low percentage of the dataset) were dropped, as well as an outlier identified when exploring the income and credit_limt variables. Additionally, the two features income and credit_limit, which were found to be highly skewed, were processed with a logarithmic transformation to alleviate the issue. Finally, two other features were dropped due to being highly correlated with others in the dataset.


**Key Issue: Class imbalance**: The target variable is not balanced, with only about 8\% of the observations in the set corresponding to clients who default their credit card payments. As a result, accuracy, which tends to be the default metric for comparing classifiers, is not really adequate. We will need to decide whether to focus on the precision or on the recall of each of the models. 
|      | Predicted label 0      | Predicted label 1 |
| ------------- | ------------- |---|
|True label 0 | true negative (TN)|false positive (FP) |
| True label 1 | false negative (FN) |true positive (TP) |

<u>Classifier Errors</u>:

* FP = false positive: the model predicts that a client will default on their credit card payment when they actually won't $\rightarrow$ a loan may be denied to a client who would actually be a good payer (lost business!).

* FN = false negative: the model predicts that a client will not default on their payment when in fact they will $\rightarrow$ lost money due to inability to collect loan repayment.

<u> Metric Candidates</u>:

* **Precision** = $\frac{TP}{TP + FP}$ = proportion of all the 'default' predictions that are actually correct.


* **Recall** = $\frac{TP}{TP + FN}$ = proportion of all the actual 'default' labels that the model was able to classify correctly.

* **f1score**  = harmonic mean of precision and recall (a combined metric that aims to balance the previous two).

In this credit card default classification problem, minimizing the false negatives means reducing the amount of money lost by the credit card issuer due to the inability to collect payment from clients who default, whereas minimizing the false positives means being able to grant more loans that will be recovered because good payers won't be denied the loan. 

So as to take a position, I posit that focusing on minimizing the false negatives appears to be a valid goal from this type of institution's point of view. In this case, the criterion that determines the optimal model is maximizing recall. Nonetheless, I will also consider the f1 score, which aims to strike a balance between precision and recall, thus combining the importance of controlling (rather than minimizing) both the false negatives and the false positives, as a metric of interest.


Hence, this analysis focuses on identifying the classifier that maximizes recall, as well as presenting the one that maximizes the f1 score.

##Baseline Classifier and Models Compared
A dummy classifier that predicts exclusively the majority class is used as baseline, and compared against simple and hyperparametrized versions of Logistic Regression, K Nearest Neighbors, Decision Tree, and Support Vector Machine.

##Results


|Criterion      |Simple Model Winner      |Improved Model Winner | Notes |
| ------------- | ------------- |---| ---|
|Simplicity| Dummy classifier|NA | baseline model, no training required|
| ------------------ | -------------------------------------- |------------------------------------| ---------------------------------------------------------|
|Recall |Logistic Regression |Decision Tree | |
| | recall = 0.7678 | recall = 0.749 | this is the metric to |
| | training time: 0.11s | training time: 3.73s |focus on |
| ------------------ | -------------------------------------- |------------------------------------| ---------------------------------------------------------|
|f1 score |Decision Tree |Decision Tree | |
| | f1 = 0.8565 | f1 = 0.8565 | this is a combined metric |
| | training time: 0.14s | training time: 3.73s |showing DT as winner as well |

The resulting best hyperparametrized classification model for this application is the optimized decision tree (with hyperparameters criterion = gini and max_depth = 2), with the added advantage that it is also extremely efficient with a training time of 3.73 seconds.

It is also worth mentioning that the optimized decision tree classifier not only maximizes the recall (among the hyperparametrized models), but it also produces an accuracy that exceeds that of the basic, majority-based dummy classifier at 97.85% (compared to 91.44% for the dummy classifier). This optimal decision tree also maximizes the f1 score, the harmonic mean of precision and recall.

**Findings**: the resulting decision tree only tests the (standardized) credit score: any client whose credit score is more than 1.328 standard deviations below the average credit score is predicted to default on their credit card payment.
This extremely simple model achieves 74.9\% precision and an f1 score of 85.65\%.

###Worth exploring: Logistic Regression Model
The Jupyter notebook presents an additional discussion on the simple and hyperparametrized logistic regression models obtained with this dataset. While more complex than the decision tree, these produced solid recall and f1-score values, and they offer the possibility of expanding the understanding and interpretation of how each of the features in the dataset affect the target variable.

