# Diabetes-Prediction

[Notebook](https://www.kaggle.com/code/wongfeiyeung/diabetes-prediction-from-iflytek)

This project aims to predict whether a person has diabetes through numerous features.

**Preprocessing**

In the preprocessing part, a feature selection is performed to reduce the dimension of the data. Some features, such as BMI, have been segmented into discrete variables according to some relevant literature, a common feature engineering technique known as binning or discretization. Segmenting continuous variables can reduce the influence of outliers on the model. Outliers can introduce significant bias to the model's predictions, but placing them within specific bins limits their impact to that particular range rather than affecting the entire variable. Discretizing continuous variables simplifies the model by treating them as categorical variables. Handling discrete variables can be more efficient and intuitive for certain models, such as decision trees and random forests.

**Mode**l

This project first classifies using SGD, SVM, and Random Forest, then ensembles these models through a voting system. Ensemble models can help mitigate both bias and variance issues. Different models in the ensemble may have different biases, and by combining them, the ensemble can reduce the overall bias. Additionally, ensemble models tend to have lower variance than individual models, which can improve generalization performance and reduce the risk of over-fitting.
