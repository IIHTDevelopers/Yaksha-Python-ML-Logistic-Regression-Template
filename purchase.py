import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Function 1: Preprocess data (load and split)
def preprocess_data(filename='purchase_data.csv'):
    # TODO: Implement data loading and splitting
    # Return dummy data that will allow tests to run but fail
    import numpy as np
    X_train = np.array([[0, 0]])
    X_test = np.array([[0, 0]])
    y_train = np.array([0])
    y_test = np.array([0])
    return X_train, X_test, y_train, y_test

# Function 2: Train logistic regression model
def train_model(X_train, y_train):
    # TODO: Implement model training
    # Return None instead of a LogisticRegression instance to make the test fail
    return None

# Function 3: Evaluate the model
def evaluate_model(model, X_test, y_test):
    # TODO: Implement model evaluation
    # Print dummy output that will allow tests to run but fail
    print("Evaluation Results:")
    print("Accuracy: 0.00")

# Function 4: Predict new samples
def predict_new(model, age, salary):
    # TODO: Implement prediction for new samples
    # Print dummy output
    print(f"Person with Age={age} and Salary={salary} will not purchase the product.\n")

# Main program
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    predict_new(model, 35, 70000)
    predict_new(model, 25, 30000)
