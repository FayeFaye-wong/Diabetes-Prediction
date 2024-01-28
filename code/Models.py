from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np

'''
Index in Evaluation: 
    TP — True Positive: Predicted as P (Positive), actually is P (True Positive).
    TN — True Negative: Predicted as N (Negative), actually is N (True Negative).
    FP — False Positive: Predicted as P (Positive), actually is N (False Positive).
    FN — False Negative: Predicted as N (Negative), actually is P (False Negative).
    Precision = TP / (TP + FP)
    Recall = TP / (TP + TN)
    F1 Score (Harmonic mean of Precision and Recall) = 2 * Precision * Recall / (Precision + Recall)
'''
class models:
    def SGD():
        # Here, the loss function must be specified as 'log' for logistic regression.
        # This is necessary to support probability outputs for ROC calculation.
        sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=6, loss="log")  # Create a classifier object
        return sgd_clf

    def train_model(X_train, y_train, model):
        # Display column names in the training data
        for col in X_train.columns:
            print(col)

        # Train the model using the provided training data
        model.fit(X_train, y_train)
        return model

    def SVM():
        # Create a pipeline to encapsulate and manage all steps in a streamlined manner
        pipe_svc = Pipeline([('svc', SVC(probability=True, random_state=6, kernel='poly', degree=8, C=0.5))])

        # Fit the pipeline on the training data
        svc_linear = pipe_svc.fit(train_x, train_y)
        
        # Evaluate the model on the test set
        svc_linear.score(test_x, test_y)

        # Make predictions on both training and test sets
        svc_linear_train_pred = svc_linear.predict(train_x)
        svc_test_pred = svc_linear.predict(test_x)

        # Use the pipeline as the classifier
        svc_clf = svc_linear
        return svc_clf

    def RFC():
        # Create a pipeline to encapsulate and manage all steps in a streamlined manner
        pipe_forest = Pipeline([('forest', RandomForestClassifier(n_estimators=100, random_state=6))])

        # Fit the pipeline on the training data
        rf_clf = pipe_forest
        rf_clf.fit(train_x, train_y)

        # Fit the RandomForestClassifier on the training data again (redundant code)
        rf_clf.fit(train_x, train_y)

        # Make predictions on both training and test sets
        forest_clf_train_pred = rf_clf.predict(train_x)
        forest_clf_test_pred = rf_clf.predict(test_x)
        return

    def make_predictions(model, X_test):
        # Use the trained model to make predictions on new data
        predictions = model.predict(X_test)
        return predictions

    def evaluate(test_preds, X_test, y_test):
        preds_df = X_test.copy()
        preds = np.round(test_preds).astype(int) + 1
        y_test = y_test + 1
        print("Number of predictions: ", len(preds_df))

        precision = (preds == y_test).sum() / len(preds)
        print("Precision: ", precision)

        recall = (preds == y_test).sum() / len(y_test)
        print("Recall: ", recall)

        f1 = 2 * (precision * recall) / (precision + recall)
        print("F1: ", f1)

