import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from AdditionalFeatures import *

if __name__ == '__main__':
    predic_model_age()
    train_df = pd.read_csv('train.csv') #read the downloaded training data set.
    train_df.iloc[np.random.permutation(len(train_df))] # shuffle the training dataset.
    train_df.replace({'male':1, 'female':0}, inplace=True) # using numbers to represent sex,
    #  as strings can not be used as input to classifier.
    train, test = train_test_split(train_df, test_size=0.5) # split the training set to training and testing set.
    X_train = train[['Pclass', 'Sex']]
    y_train = train['Survived']
    X_test = test[['Pclass', 'Sex']]
    y_test = test['Survived']
    # In the following we will try different approaches. 1). further split the training set to training and cross validation.
    #2). use cross validation.
    #2). use cross validation.
    #1). Split Training into training and validation data. 25% validation.
    # Thus, overall, 60% training, 20% validation, and 20% testing.
    train_split, validation_split = train_test_split(train, test_size=0.25)
    X_train_split= train_split[['Pclass', 'Sex']]
    y_train_split = train_split['Survived']
    X_validation_split = validation_split[['Pclass', 'Sex']]
    y_validation_split = validation_split['Survived']
    clf = RandomForestClassifier()
    clf.fit(X_train_split, y_train_split)
    y_predicted_validation_split = clf.predict(X_validation_split)
    y_predicted_test = clf.predict(X_test)
    print ("\n Training accuracy score for RF validation = ")
    print (accuracy_score(y_validation_split ,y_predicted_validation_split))
    print ("\n Training accuracy score for RF testing = ")
    print (accuracy_score(y_test ,y_predicted_test))
    clf_cv = RandomForestClassifier()
    scores = cross_val_score(clf_cv, X_train, y_train, cv=10)
    print("\n Cross Validation Score")
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    clf_cv.fit(X_train, y_train)
    y_predicted_test = clf_cv.predict(X_test)
    y_predicted_train = clf_cv.predict(X_train)
    print("\n Training accuracy score using RF CV for training data = ")
    print(accuracy_score(y_train, y_predicted_train))
    print("\n Training accuracy score using RF CV for testing data = ")
    print(accuracy_score(y_test, y_predicted_test))
    #using gridsearchCV to tune the parameter of randomforest
    # param_grid = {
    #     'n_estimators': [1, 2, 5, 10, 15, 20, 25, 30],
    #     'max_depth': [none, 1, 2, 3, 4, 6, 8, 10, 12]
    # }
    # clf_cv = randomforestclassifier()
    # clf_cv_mean = gridsearchcv(estimator=clf_cv, param_grid=param_grid, cv=10, n_jobs=-1)
    # clf_cv_mean.fit(x_train, y_train)
    # #print (clf_cv_mean.cv_results_)
    # y_predicted_test = clf_cv_mean.predict(x_test)
    # y_predicted_train = clf_cv_mean.predict(x_train)
    # print("\n training accuracy score using rf gridsearchcv for training data = ")
    # print(accuracy_score(y_train, y_predicted_train))
    # print("\n training accuracy score using rf gridsearchcv for testing data = ")
    # print(accuracy_score(y_test, y_predicted_test))
    # SVM
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    y_predicted_test = clf.predict(X_test)
    y_predicted_train = clf.predict(X_train)
    print("\n Training accuracy score using SVM for training data = ")
    print(accuracy_score(y_train, y_predicted_train))
    print("\n Training accuracy score using SVM for testing data = ")
    print(accuracy_score(y_test, y_predicted_test))


    truetestdata = pd.read_csv('test.csv')
    truetestdata.replace({'male': 1, 'female': 0}, inplace=True)
    X_truetest = truetestdata[['Pclass', 'Sex']]
    y_truetest = clf_cv.predict(X_truetest)
    Passenger_ID = truetestdata['PassengerId']
    pd_result = pd.concat([Passenger_ID, pd.DataFrame(y_truetest)], axis=1)
    pd_result.to_csv('my_submission.csv')
