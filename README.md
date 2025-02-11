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
# XGBRegressor Functionality

XGBRegressor is a gradient boosting implementation designed for regression tasks. It builds an ensemble of decision trees sequentially, optimizing a user-defined loss function with regularization. Below is an explanation of its key components and mathematical formulation.

## Key Components

- **Gradient Boosting**: Builds trees sequentially, each correcting errors from previous trees.
- **Loss Function**: Typically squared error (`reg:squarederror`) for regression.
- **Regularization**: Includes L1 (LASSO) and L2 (Ridge) penalties on leaf weights, and complexity control via tree structure.
- **Additive Training**: Predictions are the sum of outputs from all trees.

## Mathematical Formulation

### Objective Function
The objective function combines loss and regularization:
$$
\text{Obj}(\theta) = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
$$
- \( L(y_i, \hat{y}_i) \): Loss function (e.g., squared error: \( \frac{1}{2}(y_i - \hat{y}_i)^2 \)).
- \( \Omega(f_k) \): Regularization term for tree \( f_k \).
- \( K \): Total number of trees.

### Additive Model
At iteration \( t \), the prediction is:
$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)
$$
- \( \eta \): Learning rate (shrinkage to prevent overfitting).
- \( f_t \): Tree added at iteration \( t \).

### Taylor Approximation
The loss is approximated using Taylor expansion up to the second order:
$$
\text{Obj}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
$$
- \( g_i = \partial_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)}) \): First-order gradient.
- \( h_i = \partial_{\hat{y}^{(t-1)}}^2 L(y_i, \hat{y}^{(t-1)}) \): Second-order Hessian.

For squared error loss:
$$
g_i = \hat{y}^{(t-1)} - y_i, \quad h_i = 1
$$

### Regularization
For a tree with \( T \) leaves and weights \( w \):
$$
\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|
$$
- \( \gamma \): Minimum loss reduction to split a node.
- \( \lambda \): L2 regularization on leaf weights.
- \( \alpha \): L1 regularization on leaf weights.

### Optimal Leaf Weight
For leaf \( j \) with instance set \( I_j \), the optimal weight is:
$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

### Split Criteria
The gain for splitting a node into left (\( L \)) and right (\( R \)) children:
$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$
- \( G_{L/R} = \sum_{i \in I_{L/R}} g_i \)
- \( H_{L/R} = \sum_{i \in I_{L/R}} h_i \)

A split is made if the gain is positive.

## Key Hyperparameters
| Parameter | Description |
|-----------|-------------|
| `eta` (η) | Learning rate (shrinks tree weights). |
| `gamma` (γ) | Minimum loss reduction for a split. |
| `lambda` (λ) | L2 regularization on leaf weights. |
| `alpha` (α) | L1 regularization on leaf weights. |
| `max_depth` | Maximum tree depth. |
| `subsample` | Fraction of samples used per tree. |

## Conclusion
XGBRegressor optimizes a regularized objective function using gradient boosting. It builds trees greedily, selecting splits that maximize gain, and applies shrinkage to reduce overfitting. The mathematical formulation leverages Taylor expansion and regularization to balance model complexity and accuracy.

