import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
url = "breast-cancer-wisconsin.csv"
df = pd.read_csv(url)

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove the first column (Id)-> irrelevant
df = df.drop(columns=['Id'])

# Replace missing values (?) with 0
df.replace('?', 0, inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Separate features and target variable
X = df.drop(columns=['Class'])  # Ensure 'Class' is the correct column name
y = df['Class']

# Number of repetitions
n_repeats = 30
accuracies = []

# Repeat the process 30 times
for i in range(n_repeats):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the SVM classifier
    svm_classifier = SVC(kernel='rbf', gamma=1, C= 10)

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Calculate average accuracy and standard deviation
average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print(f"Average Accuracy over {n_repeats} runs: {average_accuracy:.4f} ± {std_accuracy:.4f}")