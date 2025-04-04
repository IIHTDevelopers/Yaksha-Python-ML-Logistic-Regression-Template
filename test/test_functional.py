import unittest
from test.TestUtils import TestUtils
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from purchase import preprocess_data, train_model, evaluate_model, predict_new
import io
import sys

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()
        
        # Prepare test data
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data()
        self.model = train_model(self.X_train, self.y_train)
        
        # Test data for predictions
        self.high_purchase_probability = (35, 70000)  # Age, Salary with high purchase probability
        self.low_purchase_probability = (25, 30000)   # Age, Salary with low purchase probability

    def test_preprocess_data(self):
        """
        Test case for preprocess_data() function.
        """
        try:
            X_train, X_test, y_train, y_test = preprocess_data()
            
            # Check if data is split correctly
            expected_train_size = 24  # 80% of 30 records
            expected_test_size = 6    # 20% of 30 records
            
            if (len(X_train) == expected_train_size and 
                len(X_test) == expected_test_size and 
                len(y_train) == expected_train_size and 
                len(y_test) == expected_test_size):
                self.test_obj.yakshaAssert("TestPreprocessData", True, "functional")
                print("TestPreprocessData = Passed")
            else:
                self.test_obj.yakshaAssert("TestPreprocessData", False, "functional")
                print("TestPreprocessData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPreprocessData", False, "functional")
            print(f"TestPreprocessData = Failed | Exception: {e}")

    def test_train_model(self):
        """
        Test case for train_model() function.
        """
        try:
            model = train_model(self.X_train, self.y_train)
            
            # Check if model is a LogisticRegression instance
            if isinstance(model, LogisticRegression):
                self.test_obj.yakshaAssert("TestTrainModel", True, "functional")
                print("TestTrainModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainModel", False, "functional")
                print("TestTrainModel = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainModel", False, "functional")
            print(f"TestTrainModel = Failed | Exception: {e}")

    def test_evaluate_model(self):
        """
        Test case for evaluate_model() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            evaluate_model(self.model, self.X_test, self.y_test)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if evaluation output contains expected strings
            output = captured_output.getvalue()
            if ("Evaluation Results:" in output and 
                "Accuracy:" in output and 
                "Classification Report:" in output):
                self.test_obj.yakshaAssert("TestEvaluateModel", True, "functional")
                print("TestEvaluateModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
                print("TestEvaluateModel = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
            print(f"TestEvaluateModel = Failed | Exception: {e}")

    def test_predict_new_purchase(self):
        """
        Test case for predict_new() function with high purchase probability.
        """
        try:
            age, salary = self.high_purchase_probability
            
            # Create a sample DataFrame for prediction
            sample = pd.DataFrame({'Age': [age], 'EstimatedSalary': [salary]})
            
            # Get the model's prediction
            prediction = self.model.predict(sample)[0]
            
            # Check if prediction is 1 (will purchase)
            if prediction == 1:
                self.test_obj.yakshaAssert("TestPredictNewPurchase", True, "functional")
                print("TestPredictNewPurchase = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictNewPurchase", False, "functional")
                print("TestPredictNewPurchase = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictNewPurchase", False, "functional")
            print(f"TestPredictNewPurchase = Failed | Exception: {e}")

    def test_predict_new_no_purchase(self):
        """
        Test case for predict_new() function with low purchase probability.
        """
        try:
            age, salary = self.low_purchase_probability
            
            # Create a sample DataFrame for prediction
            sample = pd.DataFrame({'Age': [age], 'EstimatedSalary': [salary]})
            
            # Get the model's prediction
            prediction = self.model.predict(sample)[0]
            
            # Check if prediction is 0 (will not purchase)
            if prediction == 0:
                self.test_obj.yakshaAssert("TestPredictNewNoPurchase", True, "functional")
                print("TestPredictNewNoPurchase = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictNewNoPurchase", False, "functional")
                print("TestPredictNewNoPurchase = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictNewNoPurchase", False, "functional")
            print(f"TestPredictNewNoPurchase = Failed | Exception: {e}")

if __name__ == '__main__':
    unittest.main()
