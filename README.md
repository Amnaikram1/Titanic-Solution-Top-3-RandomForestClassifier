# Introduction to Kaggle and the Titanic Competition

Have you ever wondered how data scientists predict things like who survived the Titanic disaster? Well, Kaggle is where the magic happens! It's like a big online playground for people who love playing with data.

The Titanic competition on Kaggle is super famous because it's perfect for beginners. You get to use real data from the Titanic to try and predict who survived. It's kind of like solving a mystery with numbers!

I decided to give it a shot myself, even though I'm pretty new to this whole data science thing. What followed was a journey of exploration and learning, as I dove into the Titanic dataset to see what I could uncover.

## Background EDA

The sinking of the Titanic remains one of the most captivating tales in maritime history. The tragic events of April 15, 1912, saw the "unsinkable" RMS Titanic meet its demise after a fateful encounter with an iceberg. Among the 2224 souls aboard, only 722 would live to tell the tale. It's a story steeped in tragedy, heroism, and the stark realities of class distinction.

### The Challenge: Predicting Survival

Fast forward to today, and the mystery of the Titanic endures. Thanks to platforms like Kaggle, we have the opportunity to delve into this enigma using data science. The challenge is clear: can we uncover patterns in the passenger data to answer the age-old question, "what sorts of people were more likely to survive?"

### Decoding the Dataset

Before we dive into the depths of our analysis, let's acquaint ourselves with the treasure trove of information at our disposal. Our dataset is a treasure map of sorts, guiding us through the intricate details of each passenger's journey. Here's a snapshot of what we're working with:

- **PassengerId:** A unique identifier for each passenger.
- **Survived:** Our target variable, indicating whether a passenger survived (1) or not (0).
- **Pclass:** A window into the socio-economic class of each passenger.
- **Demographics:** Name, sex, and age provide insights into the individual characteristics of each passenger.
- **Family Ties:** SibSp (siblings and spouse) and Parch (parents and children) shed light on the family dynamics aboard.
- **Logistics:** Ticket number, fare, and cabin offer logistical details of the voyage.
- **Embarkation:** Port of embarkation reveals the starting point of each passenger's journey.

Armed with this wealth of information, we're ready to set sail on our data science expedition. Our mission is clear: to build a predictive model that unlocks the secrets of survival aboard the Titanic. So, join me as we navigate the turbulent waters of data analysis, seeking to unravel the mysteries of history's most infamous shipwreck.

## MISSING VALUES

Some columns in both the training and test sets contain missing values, notably Age, Cabin, and Embarked in the training set, and Age, Cabin, and Fare in the test set. To address this, missing Age values are filled with the median age of respective passenger class groups due to its high correlation with Age and Survived. This approach is more logical and less prone to overfitting compared to using the overall dataset's median age. With only two missing values in the Embarked column, and the passengers sharing those missing values being female, upper-class, and traveling together with the same ticket number, we can infer they likely embarked from the same port. While "C" (Cherbourg) is the most common port for upper-class female passengers, to maintain consistency, the missing values have been filled with "S" (Southampton), the most common port of embarkation overall.

### Filling Missing Fare Values

To address the single missing Fare value, we employed a method based on related features such as passenger class (Pclass) and family size (Parch and SibSp). We calculated the median Fare for a male traveler with a third-class ticket and no family. This approach ensures a logical estimation for the missing value, promoting consistency within the dataset.

### Analyzing Cabin and Deck Information

The first letter of each cabin indicates its deck location, serving as a proxy for socio-economic status and proximity to life-saving resources. Decks B, C, D, and E, primarily for 1st class passengers, had the highest survival rates, while Deck M, designated for 2nd and 3rd class, had the lowest. To streamline analysis, we grouped similar decks: ABC for 1st class, DE for mixed-class, and FG for shared decks. Missing cabin values were labeled 'M'. Adding deck information enriches our dataset, providing insights into passenger demographics and aiding in understanding survival factors during the disaster. We also dropped the Cabin column for simplicity and to avoid redundancy.

Age and Fare display distinct split points and spikes, ideal for decision tree learning. However, differences in distribution smoothness between the training and test sets may hinder model generalization. Notably, children under 15 exhibit higher survival rates in the Age distribution, while survival rates are higher on the distribution tails in the Fare feature.

Categorical features like Pclass and Sex offer valuable insights, with each class revealing distinct mortality rates. Passengers from Southampton show lower survival rates, contrasting with higher survival rates for those from Cherbourg. Additionally, passengers with only one family member aboard tend to have higher survival rates, as indicated by Parch and SibSp features.

Features show correlations, aiding in potential feature transformation and interaction creation. Decision trees excel in capturing split points in continuous features. Categorical features' distinct distributions with varying survival rates suggest one-hot encoding and possible feature combination. The addition of Deck and removal of Cabin features streamline dataset preparation.

## FEATURE ENGINEERING

The Fare feature exhibits positive skewness, with extremely high survival rates on the right end. Using 13 quantile-based bins captures variations in survival rates across different fare groups, revealing low survival rates on the left and high survival rates on the right. Notably, an unusual group with high survival rates (15.742, 23.25] is identified.

Age distribution follows a normal pattern with spikes and bumps, segmented into 10 quantile-based bins. The first bin shows the highest survival rate, while the fourth bin exhibits the lowest. Additionally, an anomalous group (34.0, 40.0] with high survival rates is identified.

Family_size, derived from SibSp and Parch, indicates that larger family sizes correlate with higher survival rates. Categorizing family sizes as Alone, Small, Medium, and Large further illustrates this trend. Similarly, Ticket_Frequency, which accounts for group travel, reveals higher survival rates for groups of 2, 3, and 4, while solo travelers show the lowest survival rates. Unlike Family_Size, Ticket_Frequency doesn't group values to avoid redundancy and maintain information gain.

### Title & Is Married

Titles are extracted from the Name feature, with some rare titles corrected or grouped for accuracy. Titles like Miss, Mrs, and Ms are consolidated as they represent females, while others like Dr and Rev are categorized based on similar characteristics. The Is_Married feature is binary and indicates whether a passenger holds the Mrs title, which typically correlates with higher survival rates among females. Additionally, surnames are extracted from the Name feature to create the Family feature, essential for grouping passengers within the same family.

Non-numeric features like Embarked, Sex, Deck, Title, and Family_Size_Grouped are label encoded using LabelEncoder to facilitate model learning. Categorical features are further converted to one-hot encoded features, except for Age and Fare, which remain ordinal.

Binning of Age and Fare features aids in handling outliers and reveals homogeneous groups within the data. Family_Size and Ticket_Frequency features are created to capture family and group travel dynamics. Utilizing the Name feature, additional features like Title, Is_Married, and Family are generated, while target encoding produces features like Family_Survival_Rate and Ticket_Survival_Rate. Finally, non-numeric features are label encoded, categorical features are one-hot encoded, and redundant features are dropped post-encoding. This comprehensive feature engineering approach enhances model performance and predictive accuracy.

### Model Building

I created a powerful machine learning model, the single_best_model, which is a Random Forest Classifier. This model uses the Gini criterion for decision-making and consists of 1100 decision trees. Each tree has a maximum depth of 5 levels, with a minimum of 4 samples required to split a node and 5 samples required for a leaf node.

To enhance its accuracy and prevent overfitting, I set it to consider a maximum of 'auto' features for each split. Additionally, I incorporated out-of-bag (OOB) scoring for better evaluation. The model is designed to maintain consistency across runs by setting a random seed (SEED) and utilizing parallel processing with '-1' jobs for optimal performance.

In wrapping up my work on the Titanic dataset, I followed a three-step approach: exploring the data, enhancing features, and building a model. The top-performing Random Forest Classifier I used reached a fantastic accuracy of 83.73%. I dealt with missing data smartly, created new features, and understood how different factors influenced survival rates.

