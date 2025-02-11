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
`XGBRegressor` is a regression model from the XGBoost (Extreme Gradient Boosting) library, optimized for speed and performance. It builds an ensemble of decision trees iteratively to minimize a loss function.  

## **Core Concept**  
XGBoost is a boosting algorithm that builds models additively by minimizing a differentiable loss function using gradient descent.  

### **Mathematical Formulation**  
Given a dataset `{(x_i, y_i)}_{i=1}^{n}`, where:  
- `x_i` is the feature vector for the `i`th data point,  
- `y_i` is the target value.  

We aim to predict `ŷ_i` using an ensemble of `K` regression trees:  

`ŷ_i = Σ_{k=1}^{K} f_k(x_i)`,  

where `f_k(x)` represents the `k`th regression tree.  

### **Objective Function**  
XGBoost optimizes the following objective function:  

`L(Θ) = Σ_{i=1}^{n} l(y_i, ŷ_i) + Σ_{k=1}^{K} Ω(f_k)`,  

where:  
- `l(y_i, ŷ_i)` is a differentiable convex loss function (e.g., Mean Squared Error).  
- `Ω(f_k)` is a regularization term to prevent overfitting.  

#### **Loss Function (Squared Error for Regression)**  
For regression, the most common loss function is the squared error:  

`l(y_i, ŷ_i) = (y_i - ŷ_i)^2`,  

which leads to gradient boosting minimizing:  

`Σ_{i=1}^{n} (y_i - ŷ_i)^2 + Σ_{k=1}^{K} Ω(f_k)`.  

#### **Regularization Term**  
To control model complexity, XGBoost includes a regularization term:  

`Ω(f) = γT + (1/2) λ Σ_{j} w_j^2`,  

where:  
- `T` is the number of leaf nodes in the tree.  
- `w_j` is the weight of leaf `j`.  
- `γ` and `λ` are regularization parameters.  

### **Tree Growth & Optimization**  
At each iteration, a new tree is added to the model to minimize residuals. The weights of the tree are computed using the second-order Taylor expansion:  

`g_i = ∂ l(y_i, ŷ_i) / ∂ ŷ_i,`  
`h_i = ∂² l(y_i, ŷ_i) / ∂ ŷ_i²,`  

where:  
- `g_i` is the gradient (first derivative of loss).  
- `h_i` is the Hessian (second derivative of loss).  

For each leaf `j`, the optimal weight `w_j` is given by:  

`w_j* = - (Σ_{i ∈ I_j} g_i) / (Σ_{i ∈ I_j} h_i + λ)`,  

where `I_j` represents the set of samples in leaf `j`.  

### **Final Prediction**  
The final prediction is computed as:  

`ŷ_i = F_K(x_i) = F_{K-1}(x_i) + f_K(x_i)`,  

where `F_K(x)` is the cumulative model up to the `K`th tree.  

## **Hyperparameters of `XGBRegressor`**  
Key hyperparameters include:  
- `n_estimators`: Number of trees.  
- `learning_rate` (`η`): Step size shrinkage.  
- `max_depth`: Maximum depth of trees.  
- `lambda`: L2 regularization term.  
- `gamma`: Minimum loss reduction for a split.  
- `subsample`: Fraction of samples used per tree.  
- `colsample_bytree`: Fraction of features used per tree.  

## **Conclusion**  
`XGBRegressor` is a powerful and efficient gradient boosting algorithm designed for regression tasks. It minimizes a loss function using additive tree models while incorporating regularization for better generalization.  

---

This version should now be fully compatible with GitHub Markdown preview. Let me know if you need any tweaks! 🚀

## **Conclusion**  
`XGBRegressor` is a powerful and efficient gradient boosting algorithm designed for regression tasks. It minimizes a loss function using additive tree models while incorporating regularization for better generalization.  

---

