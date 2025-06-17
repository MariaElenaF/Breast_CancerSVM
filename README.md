### Key Features ###

#Preprocessing
-Cleans column names and removes the non-informative Id column.
-Replaces missing values (?) with 0 for consistency.
-Converts all values to numeric format.

#Feature Scaling
Standardizes all input features using StandardScaler from scikit-learn.

#Modeling
Implements a Support Vector Classifier (SVC) with:

->RBF kernel
->gamma = 1
->C = 10

Aims to balance margin maximization and misclassification tolerance.

#Evaluation
Repeats the full training and testing pipeline 30 times with different random seeds.

Stores accuracy from each run to measure model stability.

#Performance
-Average Accuracy: 95.10%
-Standard Deviation: ±1.25%

Consistent predictive performance across multiple random splits.

#Dependencies
->pandas
->numpy
->scikit-learn
