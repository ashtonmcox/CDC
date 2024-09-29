# Carolina Data Challenge 2024 Cheat Sheet

## Table of Contents

1. [Setup](#setup)
2. [Table Set Values & Key Functions](#table-set-values)
4. [Simple Statistical Formulas](#simple-statistical-formulas)
5. [Setup](#setup)
6. [Data Preprocessing](#data-preprocessing)
7. [Categorical Data Classification with Pandas](#categorical-data-classification)
8. [Model Evaluation Metrics](#model-evaluation-metrics)
9. [Model Selection and Cross-Validation](#model-selection)
10. [Feature Selection and Dimensionality Reduction](#feature-selection)

## <a name="setup"></a> 1. Setup

- **Windows** users can download and install Python from the [official website](https://www.python.org/downloads/).

I recommend upgrading `pip` before proceeding to ensure smooth installations.

## Upgrading Pip

Before installing the required packages, upgrade `pip`, `setuptools`, and `wheel` to avoid installation issues:

- Run the following command:
  ```bash
  python3 -m pip install --upgrade pip setuptools wheel
  ```

  - On **macOS**, you might need to use `python3`:
    ```bash
    python3 -m pip install --upgrade pip setuptools wheel
    ```

  - On **Windows**, use:
    ```bash
    python -m pip install --upgrade pip setuptools wheel
    ```

## Installing Required Packages

1. Ensure your system has the necessary tools to install packages with compiled code:

   - **macOS**: You may need to install Xcode command-line tools. Run the following command:
     ```bash
     xcode-select --install
     ```

   - **Windows**: Some Python packages may require additional build tools (like `Visual Studio Build Tools`). You can install them from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

2. Install the packages listed in the `requirements.txt` file by running:

   - **macOS**:
     ```bash
     python3 -m pip install -r requirements.txt
     ```

   - **Windows**:
     ```bash
     python -m pip install -r requirements.txt
     ```

3. **Note for macOS M1/M2 Users**: If you are on an Apple Silicon Mac, you may need to run the installation with architecture flags:
   ```bash
   arch -arm64 python3 -m pip install -r requirements.txt
   ```

4. **Binary wheels for problematic packages**: If you encounter issues with certain packages like `numpy`, `matplotlib`, or `scikit-learn`, try installing them using precompiled binaries to avoid errors related to system architecture:

   ```bash
   pip install --only-binary=:all: numpy pandas scikit-learn
   ```

## Troubleshooting

- **macOS**: If you get any errors about missing libraries or compilers, ensure that Xcode command-line tools are installed (`xcode-select --install`).
  
- **Windows**: Ensure that `Visual Studio Build Tools` are installed and that your system is up to date.

## Uninstalling Packages

If you need to remove the installed packages at any point, run the following command:

- **macOS**:
  ```bash
  python3 -m pip uninstall -r requirements.txt
  ```

- **Windows**:
  ```bash
  python -m pip uninstall -r requirements.txt
  ```

## <a name="table-set-values"></a> 2. Table Set Values & Key Functions

### Data Types in Pandas:
| Data Type | Description                                        | Example            |
|-----------|----------------------------------------------------|--------------------|
| `int`     | Integer values                                     | `42, -1`           |
| `float`   | Floating-point values                               | `3.14, -0.001`     |
| `object`  | Text or mixed types                                 | `'apple', '123abc'`|
| `category`| Categorical data with defined categories           | `'low', 'medium'`  |
| `bool`    | Boolean values                                      | `True, False`      |
| `datetime`| Date and time data                                  | `2024-09-20 12:00` |

### Key Functions by Library:

1. **Pandas**  
   - `pd.read_csv()`: Load data from CSV.
   - `df.describe()`: Summary statistics.
   - `df.groupby()`: Group data and apply aggregations.

2. **NumPy**  
   - `np.mean()`: Calculate mean.
   - `np.median()`: Calculate median.
   - `np.std()`: Calculate standard deviation.

3. **Scikit-Learn**  
   - `train_test_split()`: Split data into training and test sets.
   - `GridSearchCV`: Optimize model hyperparameters.
   - `classification_report()`: Comprehensive model evaluation.

4. **SciPy**  
   - `stats.ttest_ind()`: Perform a t-test.
   - `stats.pearsonr()`: Calculate Pearson correlation.
   - `stats.norm.cdf()`: Cumulative distribution function of a normal distribution.

5. **Matplotlib/Seaborn**  
   - `plt.plot()`: Basic plotting.
   - `sns.heatmap()`: Visualize data as a heatmap.
   - `sns.pairplot()`: Plot pairwise relationships in a dataset.

6. **Sympy**  
   - `prime()`: Generates the nth prime number.
   - `factorint()`: Factorizes an integer into its prime factors.
   - `isprime()`: Checks if a number is prime.

## <a name="simple-statistical-formulas"></a> 3. Simple Statistical Formulas

1. **Mean (Average)**:
   $$\text{Mean} = \frac{\sum_{i=1}^{n} x_i}{n}$$
   - `x_i`: Value of each observation
   - `n`: Number of observations

2. **Median**:
   - The middle value of a dataset when arranged in ascending order. If `n` is even, the median is the average of the two middle numbers.

3. **Mode**:
   - The most frequently occurring value in a dataset.

4. **Variance (σ²)**:
   $$text{Variance} = \frac{\sum_{i=1}^{n} (x_i - \mu)^2}{n}$$
   - `μ`: Mean of the dataset

5. **Standard Deviation (σ)**:
   $$text{Standard Deviation} = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \mu)^2}{n}}$$

6. **Correlation Coefficient (Pearson's r)**:
   $$\frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$
   - $$x_i$$, $$y_i$$: Individual sample points
   - $$\bar{x}$$, $$\bar{y}$$: Means of the datasets

7. **Linear Regression Formula**:
   $$y = mx + b$$
   - `y`: Dependent variable
   - `x`: Independent variable
   - `m`: Slope of the line
   - `b`: Intercept

### 1. **Handling Missing Values**

#### 1.1. **Detecting Missing Values**
Check for missing values in the dataset.

```python
import pandas as pd

# Example data
data = {'A': [1, 2, None], 'B': [None, 2, 3]}
df = pd.DataFrame(data)

# Detecting missing values
print(df.isnull())
print("Missing values per column:\n", df.isnull().sum())
```

#### 1.2. **Removing Missing Values**
Remove rows or columns with missing values.

```python
# Dropping rows with missing values
df_dropped_rows = df.dropna()
print(df_dropped_rows)

# Dropping columns with missing values
df_dropped_cols = df.dropna(axis=1)
print(df_dropped_cols)
```

#### 1.3. **Imputing Missing Values**
Fill missing values with a specified value, mean, median, or mode.

```python
# Fill with a specified value
df_filled = df.fillna(0)

# Fill with column mean
df_filled_mean = df.fillna(df.mean())

# Fill with column median
df_filled_median = df.fillna(df.median())

# Fill with column mode
df_filled_mode = df.fillna(df.mode().iloc[0])
print(df_filled, df_filled_mean, df_filled_median, df_filled_mode)
```

### 2. **Handling Duplicates**

#### 2.1. **Detecting Duplicates**
Find duplicate rows in the dataset.

```python
# Detecting duplicates
print("Duplicate rows:\n", df[df.duplicated()])
```

#### 2.2. **Removing Duplicates**
Remove duplicate rows, keeping the first occurrence.

```python
# Removing duplicates
df_unique = df.drop_duplicates()
print(df_unique)
```

### 3. **Data Transformation**

#### 3.1. **Renaming Columns**
Rename columns to improve readability or comply with naming conventions.

```python
# Renaming columns
df_renamed = df.rename(columns={'A': 'Feature_A', 'B': 'Feature_B'})
print(df_renamed)
```

#### 3.2. **Changing Data Types**
Convert data types for better memory usage and accurate computations.

```python
# Converting data types
df['A'] = df['A'].astype('float')
df['B'] = df['B'].astype('category')
print(df.dtypes)
```

### 4. **Handling Outliers**

#### 4.1. **Detecting Outliers**
Use statistical methods like IQR to detect outliers.

```python
# Example data with an outlier
data = {'Value': [1, 2, 3, 4, 100]}
df_outliers = pd.DataFrame(data)

# Detecting outliers using IQR
Q1 = df_outliers['Value'].quantile(0.25)
Q3 = df_outliers['Value'].quantile(0.75)
IQR = Q3 - Q1

outliers = df_outliers[(df_outliers['Value'] < (Q1 - 1.5 * IQR)) | (df_outliers['Value'] > (Q3 + 1.5 * IQR))]
print("Outliers:\n", outliers)
```

#### 4.2. **Handling Outliers**
Remove or transform outliers based on business logic.

```python
# Removing outliers
df_no_outliers = df_outliers[(df_outliers['Value'] >= (Q1 - 1.5 * IQR)) & (df_outliers['Value'] <= (Q3 + 1.5 * IQR))]
print(df_no_outliers)
```

### 5. **Feature Engineering**

#### 5.1. **Creating New Features**
Generate new features based on existing data.

```python
# Example data
data = {'Length': [2, 3, 4], 'Width': [1, 1.5, 2]}
df_features = pd.DataFrame(data)

# Creating a new feature 'Area'
df_features['Area'] = df_features['Length'] * df_features['Width']
print(df_features)
```

#### 5.2. **Binning**
Convert continuous data into categorical bins.

```python
#

 Binning example
bins = [0, 1, 2, 3, 4]
labels = ['Very Low', 'Low', 'Medium', 'High']
df_features['Length_Binned'] = pd.cut(df_features['Length'], bins=bins, labels=labels)
print(df_features)
```

## <a name="categorical-data-classification"></a> 5. Categorical Data Classification with Pandas

### 1. **Encoding Categorical Variables**

#### 1.1. **Label Encoding**
Convert categorical labels into numerical form.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])
print(df)
```

#### 1.2. **One-Hot Encoding**
Create binary columns for each category.

```python
# One-hot encoding
df_one_hot = pd.get_dummies(df, columns=['Category'])
print(df_one_hot)
```

## <a name="model-evaluation-metrics"></a> 6. Model Evaluation Metrics

### 1. **Confusion Matrix**
Evaluate classification models with a confusion matrix.

```python
from sklearn.metrics import confusion_matrix

y_true = [0, 1, 0, 1]
y_pred = [0, 0, 1, 1]
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)
```

### 2. **Classification Report**
Get a detailed report on precision, recall, F1-score, and support.

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

### 3. **ROC Curve and AUC**
Visualize the performance of a binary classifier.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Sample probabilities for positive class
y_scores = [0.1, 0.4, 0.35, 0.8]
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

## <a name="model-selection"></a> 7. Model Selection and Cross-Validation

### 1. **Cross-Validation**
Use cross-validation to evaluate the model's performance.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", scores)
```

### 2. **Grid Search for Hyperparameter Tuning**
Optimize model hyperparameters using grid search.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
```

## <a name="feature-selection"></a> 8. Feature Selection and Dimensionality Reduction

### 1. **Feature Importance**
Identify important features using model-based approaches.

```python
model.fit(X, y)
importances = model.feature_importances_

# Visualize feature importance
plt.bar(range(len(importances)), importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
```

### 2. **Principal Component Analysis (PCA)**
Reduce dimensionality while retaining variance.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```

## <a name="advanced-preprocessing"></a> 9. Advanced Preprocessing Techniques

### 1. **Scaling Features**
Standardize or normalize features for model training.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Normalization
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)
```

### 2. **Polynomial Features**
Create polynomial features to capture interactions.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print(X_poly)
```

## 10. Univariate Linear Regression

### 1. **Overview**
Univariate linear regression involves predicting a single dependent variable based on one independent variable. The goal is to fit a line that best explains the relationship between the variables.

The linear regression equation is:
$$ y = mx + b $$
- `y`: Dependent variable
- `x`: Independent variable
- `m`: Slope of the line
- `b`: Intercept

### 2. **Fitting a Model in Scikit-learn**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Fitting the model
model = LinearRegression()
model.fit(X, y)

# Coefficients
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

# Predicting
y_pred = model.predict([[6]])
print("Prediction for X=6:", y_pred)
```

### 3. **Evaluating the Model**

- **R-squared (Coefficient of Determination)**:
  $$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} $$
  where:
  - \( SS_{res} \) is the residual sum of squares
  - \( SS_{tot} \) is the total sum of squares

```python
from sklearn.metrics import r2_score

# Calculate R-squared
r2 = r2_score(y, model.predict(X))
print("R-squared:", r2)
```

### 4. **Plotting the Regression Line**

```python
import matplotlib.pyplot as plt

# Plot the data and regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.title('Univariate Linear Regression')
plt.show()
```

---

## 11. Clustering Models

### 1. **Overview**
Clustering involves grouping data points based on similarity without labeled outputs. Popular algorithms include **K-means**, **DBSCAN**, and **Hierarchical Clustering**.

### 2. **K-means Clustering**

#### 2.1. **How it Works**
K-means assigns each data point to one of `K` clusters based on minimizing the variance within clusters. The objective is to minimize the within-cluster sum of squares.

#### 2.2. **Fitting K-means Model in Scikit-learn**

```python
from sklearn.cluster import KMeans
import numpy as np

# Example data
X = np.array([[1, 2], [2, 3], [3, 4], [8, 8], [9, 10]])

# Fitting the model
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Cluster labels
print("Cluster labels:", kmeans.labels_)

# Cluster centers
print("Cluster centers:", kmeans.cluster_centers_)
```

#### 2.3. **Elbow Method for Choosing K**

The elbow method helps determine the optimal number of clusters (`K`) by plotting the within-cluster sum of squares (WCSS) for different values of `K`.

```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
```

### 3. **DBSCAN Clustering**

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) forms clusters based on density, making it robust to outliers.

```python
from sklearn.cluster import DBSCAN

# Fitting the model
dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan.fit(X)

# Cluster labels
print("DBSCAN labels:", dbscan.labels_)
```

### 12. **Chi-Squared Tests for Independence**

#### 1. **Overview**
A Chi-Squared Test for Independence determines if there is a significant association between two categorical variables.

#### 2. **Performing a Chi-Squared Test in Python**

```python
from scipy.stats import chi2_contingency
import pandas as pd

# Example contingency table
data = [[10, 20], [20, 40]]
chi2, p, dof, expected = chi2_contingency(data)

print(f"Chi-Squared: {chi2}, p-value: {p}")
```

#### 3. **Interpreting Results**
- If the p-value is less than the significance level (e.g., 0.05), reject the null hypothesis (indicating a relationship between the variables).

---

### 13. **Advanced Feature Engineering**

#### 1. **Feature Scaling Techniques**
- **RobustScaler**: Use this scaler when there are many outliers, as it scales data based on percentiles.
  
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

#### 2. **Interaction Features**
Creating interaction terms between variables can improve model performance.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_interaction = poly.fit_transform(X)
```

---

### 14. **Time Series Data**

#### 1. **Overview**
Time series data is a sequence of data points indexed by time. It's essential to account for the temporal structure in data analysis.

#### 2. **Rolling Mean and Window Functions**

```python
import pandas as pd

# Example data
df['Rolling_Mean'] = df['Value'].rolling(window=3).mean()
print(df['Rolling_Mean'])
```

#### 3. **Time Series Decomposition**
Decompose time series into trend, seasonality, and residuals.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Value'], model='additive', period=12)
result.plot()
```

---

### 16. **Pavlov's CP (Critical Period) for Model Evaluation**

#### 1. **Overview**
Pavlov's Critical Period (CP) is a model evaluation technique that focuses on how a model performs during specific, crucial time windows or periods. This is especially important in applications where timing is a critical factor (e.g., stock price predictions, sales forecasting, or emergency systems).

#### 2. **Purpose**
- To assess model performance during high-impact or high-stakes periods, such as peak sales hours, financial crises, or specific seasons.
- Helps to evaluate if a model remains reliable under pressure or during critical events.

#### 3. **Application**
You define the critical periods (CPs) and evaluate model performance during these periods separately from overall performance. For example, you could focus on evaluating:
- A predictive model's accuracy during market open/close hours.
- A sales forecast model during Black Friday or the holiday season.

#### 4. **Implementation Example**

Let's assume we have sales data and our critical period is the holiday season (November and December). We can filter the data for those months and evaluate the model's performance.

```python
# Define the critical period (e.g., November and December)
critical_period = df[(df['Date'].dt.month == 11) | (df['Date'].dt.month == 12)]

# Evaluate the model's performance only on the critical period data
from sklearn.metrics import mean_squared_error

y_true_cp = critical_period['True_Values']
y_pred_cp = model.predict(critical_period['Features'])

mse_cp = mean_squared_error(y_true_cp, y_pred_cp)
print(f"MSE during Critical Period: {mse_cp}")
```

#### 5. **Why It Matters**
- **Context-specific performance:** A model that performs well overall may struggle during critical periods, which could be more important in practice.
- **Risk Mitigation:** By focusing on high-stakes periods, you can catch potential failures before they occur in crucial moments.
