# Titanic Solution: Random Forests and Feature Engineering

The journey through the Titanic dataset encompasses three pivotal stages: Exploratory Data Analysis (EDA), Feature Engineering, and Model Building.

### Dataset Information

- **Training Set:** 891 rows, 12 features (including the target variable 'Survived')
- **Test Set:** 418 rows, 11 features
- **Target Variable:** 'Survived' (0 or 1)

## A. Exploratory Data Analysis:

#### 1. Handling Missing Values
The initial step in our analysis involved addressing missing values in both the training and test datasets. Utilizing `isnull().sum()`, we assessed the extent of missingness across features. With missing values identified in 'Age,' 'Cabin,' 'Embarked,' and 'Fare,' we strategically filled these gaps through imputation techniques tailored to each feature.
Multiple columns in both training and test sets presented missing values. Addressed using `isnull().sum()` to assess column-wise missing values.

#### 1.1 Missing Values Summary
- **Training Set:** Missing values in 'Age,' 'Cabin,' 'Embarked'
- **Test Set:** Missing values in 'Age,' 'Cabin,' 'Fare'

#### 1.2 Concatenated Dataset Approach
Mitigated overfitting concerns by handling missing values in the concatenated training and test sets.

#### 1.3 Filling Missing Values
- **Age:** Imputed with median age based on Pclass and Sex groups.
- **Embarked:** Filled categorical values with 'S' based on specific passenger information.
- **Fare:** Imputed one missing value leveraging assumptions tied to family size, Pclass, and gender.
- **Cabin:** Introduced a new 'Deck' feature, replacing 'Cabin.'

### 2. Correlations
Exploring feature correlations revealed insights crucial for subsequent model building. Notably, we observed a significant correlation between 'Fare' and 'Pclass,' underscoring the socio-economic dynamics aboard the Titanic. These correlations guided feature selection and transformation strategies.

- **Highest Correlation:**
  - Training Set: 0.5495 ('Fare' and 'Pclass')
  - Test Set: 0.5771 ('Fare' and 'Pclass')

### 3. Target Distribution in Features

#### 3.1 Continuous Features
Identified split points and spikes in 'Age' and 'Fare' suitable for a decision tree model. Noted higher survival rates for children and tails of the 'Fare' distribution.

#### 3.2 Categorical Features
Explored survival rates based on boarding location, family size, and other categorical features.

#### 3.3 Feature correlations
Feature correlations indicate opportunities for transformation and interaction. Proposed target encoding for features with high correlations. Distinct distributions in categorical features (Pclass and Sex) with varying survival rates. Introduced 'Deck' as a feature to capture survival rates on different decks.

## B. Feature Engineering
Drawing insights from passenger names, we extracted titles ('Mr,' 'Mrs,' etc.) and inferred marital status ('Is_Married'). Additionally, we engineered features like 'Family_Size' and leveraged target encoding to encapsulate survival rates associated with family units and ticket groups.

### 2.1 Binning Continuous Features
To enhance model performance and interpretability, we binned continuous features like 'Age' and 'Fare' into quantile-based bins. This transformation facilitated capturing nonlinear relationships and identifying groups with differential survival rates.
#### 2.1.1 Fare
Binned 'Fare' into 13 quantile-based bins, revealing varied survival rates.

#### 2.1.2 Age
Applied binning to 'Age' with 10 quantile-based bins, capturing groups with distinct survival rates.

### 2.2 Frequency Encoding
Created 'Family_Size' by summing 'SibSp,' 'Parch,' and 1, categorizing family sizes.

### 2.3 Title & Is Married
Extracted 'Title' from passenger names and introduced 'Is_Married' based on the 'Mrs' title.

### 2.4 Target Encoding
Introduced features like 'Family_Survival_Rate' and 'Ticket_Survival_Rate' using target encoding.

### 2.5 Feature Transformation
Transforming categorical features through label encoding and one-hot encoding ensured compatibility with machine learning algorithms. This step streamlined the representation of categorical data while preserving valuable information for predictive modeling.

#### 2.5.1 Label Encoding
Applied label encoding to non-numerical features such as 'Embarked,' 'Sex,' 'Deck,' etc.

#### 2.5.2 One-Hot Encoding
Transformed categorical features into one-hot encoded features.

## C. Model Configuration:

###  Modeling - Random Forest Algorithm

Utilized a tuned Random Forest Classifier for its robustness and predictive capabilities.
I created a powerful machine learning model, the `single_best_model`, which is a Random Forest Classifier. This model uses the Gini criterion for decision-making and consists of 1100 decision trees. Each tree has a maximum depth of 5 levels, with a minimum of 4 samples required to split a node and 5 samples required for a leaf node.

To enhance its accuracy and prevent overfitting, I set it to consider a maximum of 'auto' features for each split. Additionally, I incorporated out-of-bag (OOB) scoring for better evaluation. The model is designed to maintain consistency across runs by setting a random seed (SEED) and utilizing parallel processing with '-1' jobs for optimal performance.

### D. Conclusion:

In wrapping up my work on the Titanic dataset, I followed a three-step approach: exploring the data, enhancing features, and building a model. The top-performing Random Forest Classifier I used reached a fantastic accuracy of 83.73%. I dealt with missing data smartly, created new features, and understood how different factors influenced survival rates.
