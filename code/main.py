from dataprocessing import dataproc
from FeatureSelection import selection
from Models import models
from sklearn.ensemble import VotingClassifier

def main():
    # Load the data
    x_train = dataproc.load_and_clean_data()
    y_train = dataproc.takey()
    x_valid = dataproc.valid_dataset()
    # Perform feature selection
    selection.perform_feature_selection()
    # get only 30% of the data for training
    x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.7, random_state=6)
    
    sgd_model = models.SGD()

    # Train the SGD model
    sgd_model = models.train_model(train_x, train_y, sgd_model)

    # Call the SVM method
    svm_model = models.SVM()

    # Call the RFC method
    rfc_model = models.RFC()

    # Make predictions using the trained models
    sgd_preds = models.make_predictions(sgd_model, test_x)
    svm_preds = models.make_predictions(svm_model, test_x)
    rfc_preds = models.make_predictions(rfc_model, test_x)

    # Evaluate the models
    print("SGD Model Evaluation:")
    models.evaluate(sgd_preds, test_x, test_y)

    print("\nSVM Model Evaluation:")
    models.evaluate(svm_preds, test_x, test_y)

    print("\nRFC Model Evaluation:")
    models.evaluate(rfc_preds, test_x, test_y)
    
    #ensemble models
    eclf = VotingClassifier(estimators=[('SGD',sgd_model),('SVM',svm_model),('rf', rfc_model)], voting='soft', weights=[1,1,2])
    eclf.fit(train_x, train_y)
    for clf, label in zip([sgd_model,svm_model,rfc_model], ['SGD','SVM', 'Random Forest','Ensemble']):
        scores = cross_val_score(clf, x_train, y_train, cv=3, scoring='f1')
        print("score: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))
    eclf_pre=eclf.make_predictions(eclf,test_x)
    cm = metrics.confusion_matrix(test_y, eclf_pre)
    sns.heatmap(cm, annot=True, fmt='d', linewidths=0.5)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predict')
    plt.show()

    return

main()

