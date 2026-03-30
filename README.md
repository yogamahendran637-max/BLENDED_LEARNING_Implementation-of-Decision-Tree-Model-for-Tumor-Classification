# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries Import pandas, scikit-learn, seaborn, and matplotlib for data handling, modeling, and visualization.
   
2.Load Dataset Load the dataset from the provided URL and explore its structure.

3.Feature Selection Define features (X) by excluding irrelevant columns like id and set diagnosis as the target (y).

4.Split Dataset Split the data into training and testing sets (70-30 ratio).

5.Train Model Train a Decision Tree Classifier on the training data.

6.Evaluate Model Predict using the test set and calculate accuracy, generate a classification report, and confusion matrix.

7.Visualize Results Plot a heatmap of the confusion matrix to assess prediction performance.
    

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by:Yogamahendran G
RegisterNumber: 212225040500


*/
Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

Step 1: Data Loading
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/tumor.csv"
data = pd.read_csv(url)

Step 2: Data Exploration
Display the first few rows and column names for verification
print(data.head())
print(data.columns)

Step 3: Feature Selection
Assume that the 'diagnosis' column is the target, with other columns as features
Drop 'id' and any unrelated columns
X = data.drop(columns=['id', 'diagnosis'])  # Adjust if there are any other non-feature columns
y = data['diagnosis']  # 'diagnosis' column indicates benign or malignant

Step 4: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Step 5: Model Training
Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

Step 6: Model Evaluation
Predicting on the test set
y_pred = model.predict(X_test)

Calculate accuracy and print classification metrics
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
*/
```

## Output:


<img width="1037" height="500" alt="513186515-9b1866fb-744b-4387-b08f-52e1df73b347" src="https://github.com/user-attachments/assets/a4bc153b-5799-4193-abdd-7ccb3796950f" />



<img width="1025" height="584" alt="513186621-64ec0353-0c7b-433b-ae11-f61a2b19783c" src="https://github.com/user-attachments/assets/8adf1fe3-b9db-4388-b02f-fa00faa4f13f" />









## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
