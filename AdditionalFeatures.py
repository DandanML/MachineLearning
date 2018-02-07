import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
def predic_model_age():
    train_df = pd.read_csv('train.csv', keep_default_na = False) #read the downloaded training data set.
    train_df.replace({'male': 1, 'female': 0, '': 0}, inplace=True)  # using numbers to represent sex,
    train_df.iloc[np.random.permutation(len(train_df))] # shuffle the training dataset.
    train_df_age = train_df.loc[train_df['Age'] != -1]
    train, test = train_test_split(train_df_age, test_size=0.5) # split the training set to training and testing set.
    X_train = train[['Pclass', 'Sex', 'Age']]
    y_train = train['Survived']
    X_test = test[['Pclass', 'Sex', 'Age']]
    y_test = test['Survived']
    clf_cv = RandomForestClassifier()
    scores = cross_val_score(clf_cv, X_train, y_train, cv=10)
    print("\n Cross Validation Score with age")
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    clf_cv.fit(X_train, y_train)
    y_predicted_test = clf_cv.predict(X_test)
    print("\n Training accuracy score using RF CV for testing data with age = ")
    print(accuracy_score(y_test, y_predicted_test))
    truetestdata = pd.read_csv('test.csv', keep_default_na = False)
    truetestdata.replace({'male': 1, 'female': 0, '':0}, inplace=True)
    #truetestdata_age = truetestdata.loc[train_df['Age'] != -1]
    X_truetest_age = truetestdata[['Pclass', 'Sex','Age']]
    y_truetest_age = clf_cv.predict(X_truetest_age)
    Passenger_ID = truetestdata['PassengerId']
    pd_result = pd.concat([Passenger_ID, pd.DataFrame(y_truetest_age)], axis=1)
    pd_result.to_csv('my_submission.csv')
