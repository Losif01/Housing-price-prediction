# Housing price prediction

## Overview 
This Data science project analyses the [Housing Price Dataset]([House Prices - Advanced Regression Techniques | Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)), predicting the `SalePrice` based on all other features in the dataset
**Technologies used:**
- `python 3.12`
- `matplotlib`
- `pandas`
- `numpy`
- `seaborn`
- `XGBRegressor`
- `Sklearn`
## Step 1: Exploratory Data Analysis 

- Know the null and unique values in each column in the dataset 
- Differentiating between object and numeric values for their corresponding suitable data cleaning technique 
- Plotted the numeric data correlation with the target column `SalePrice`
- Plotted a frequency histogram for each numeric column in the dataset
![[Pasted image 20250211205718.png]]

- Created a Correlation `Heatmap` 
![[Pasted image 20250211205635.png]]

- Created a function to find all null values in a column, which was useful in the long run
```python
def getMissingValues(df):
    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_values = missing_values[missing_values > 0]
    missing_values = missing_values / len(df)
    return [missing_values], missing_values.__len__()
```

- Created a function to find all unique values in the dataset
```python
def getUniqueVaules(df):

    unique_values = []
    for i in df.columns:
        unique_values.append((i , df[i].nunique()))
    return f" {unique_values} "
```

- Plotted a frequency histogram for each categorical column, and filled the null values with suitable 
- Handled numeric null records with `Measure of Center` techniques (mean, median, mode)
- Export cleaned data to new files

--- 
## Step 2: Feature Engineering
- Added the following columns that are derived from other columns:
```python 
train['TotalArea'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

test['TotalArea'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

# Age of the house when sold
train['HouseAge'] = train['YrSold'] - train['YearBuilt']

test['HouseAge'] = test['YrSold'] - test['YearBuilt']

# Interaction term: Overall quality * Living area
train['QualLivArea'] = train['OverallQual'] * train['GrLivArea']

test['QualLivArea'] = test['OverallQual'] * test['GrLivArea']
```

- used the `log transform` to convert skewed data into an acceptable domain for training
```python
skewed_features = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF']

  

for col in skewed_features:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])

  

# Log-transform the target variable
y_train = np.log1p(y_train)  # Reverse later with np.expm1()
```

- used `Ordinal encoding` for some columns 
```python 
# Define ordinal mappings (customize based on your data)

ordinal_mappings = {

    'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},

    'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0},

    'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}

}

  

for col, mapping in ordinal_mappings.items():

    train[col] = train[col].map(mapping)

    test[col] = test[col].map(mapping).fillna(0)  # Handle missing/unseen categories in test
```

- Saved the processed data for the next Stage: *Model building*

---

## **Step 3: Model Building _`XGBregressor`_  

## **Introduction**

`XGBRegressor` is a regression model from the XGBoost (Extreme Gradient Boosting) library, which is optimized for speed and performance. It is an implementation of gradient boosting designed to be highly efficient, flexible, and portable. The model builds an ensemble of decision trees to minimize a loss function iteratively.

## **Core Concept**

XGBoost is a boosting algorithm that builds models in an additive manner by minimizing a differentiable loss function. It uses gradient descent to optimize the model.

### **Mathematical Formulation**

Given a dataset ${(xi,yi)}i=1n\{(x_i, y_i)\}_{i=1}^{n}$, where:

- $x_i$ is the feature vector for the $i^{th}$ data point,
- $y_i$ is the target value.

We aim to predict $y^i\hat{y}_i$ using an ensemble of KK regression trees:

$y^i=∑k=1Kfk(xi)\hat{y}_i=\sum_{k=1}^{K} f_k(x_i)$

where $fk(x)$ $f_k(x)$ represents the $k^{th}$ regression tree.

### **Objective Function**

XGBoost optimizes the following objective function:

$L(Θ)=∑i=1nl(yi,y^i)+∑k=1KΩ(fk)\mathcal{L}(\Theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$
where:

- $l(yi,y^i) l(y_i, \hat{y}_i)$ is a differentiable convex loss function that measures the difference between the actual and predicted values (e.g., Mean Squared Error).
- $Ω(fk)\Omega(f_k)$ is a regularization term to prevent overfitting.

#### **Loss Function (Squared Error for Regression)**

For regression, the most common loss function is the squared error:

$l(yi,y^i)=(yi−y^i)2l(y_i, \hat{y}_i) = (y_i - \hat{y}_i)^2$

which leads to gradient boosting minimizing:

$∑i=1n(yi−y^i)2+∑k=1KΩ(fk)\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \sum_{k=1}^{K} \Omega(f_k)$

#### **Regularization Term**

To control model complexity, XGBoost includes a regularization term:

$Ω(f)=γT+12λ∑jwj2\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j} w_j^2$

where:

- TT is the number of leaf nodes in the tree.
- $w_j$ is the weight of leaf $jj$.
- $γ$ and $λ$ are regularization parameters.

### **Tree Growth & Optimization**

At each iteration, a new tree is added to the model to minimize the residuals. The weights of the tree are computed using the second-order Taylor expansion:

$gi=∂l(yi,y^i)∂y^i,hi=∂2l(yi,y^i)∂y^i2g_i = \frac{\partial l(y_i, \hat{y}_i)}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$
where:

- $g_i$ is the gradient (first derivative of loss).
- $h_i$ is the Hessian (second derivative of loss).

For each leaf $jj$, the optimal weight $w_j$ is given by:

$wj∗=−∑i∈Ijgi∑i∈Ijhi+λw_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$
where $I_j$ represents the set of samples in leaf $jj$.

### **Final Prediction**

The final prediction is computed as:

$y^i=FK(xi)=FK−1(xi)+fK(xi)\hat{y}_i = F_{K}(x_i) = F_{K-1}(x_i) + f_K(x_i)$

where $FK(x)F_K(x)$ is the cumulative model up to the $K^{th}$ tree.

## **Hyperparameters of `XGBRegressor`**

Key hyperparameters include:

- `n_estimators`: Number of trees.
- `learning_rate` (η\eta): Step size shrinkage.
- `max_depth`: Maximum depth of trees.
- `lambda`: L2 regularization term.
- `gamma`: Minimum loss reduction for a split.
- `subsample`: Fraction of samples used per tree.
- `colsample_bytree`: Fraction of features used per tree.

## **Conclusion**

`XGBRegressor` is a powerful and efficient gradient boosting algorithm designed for regression tasks. It minimizes a loss function using additive tree models while incorporating regularization for better generalization.

---
