from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import logloss
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

def main():
    #read in  data, parse into training and target sets
    dataset = pd.read_csv('Data/train.csv')
    target = dataset.Activity.values
    train = dataset.drop('Activity', axis=1).values
    imp = Imputer(missing_values = 'NaN',strategy='mean',axis=0)
    new_train_data = imp.fit_transform(train)

    #In this case we'll use a random forest, but this could be any classifier
    cfr = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    #Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(new_train_data), n_folds=5, indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = cfr.fit(new_train_data[traincv], target[traincv]).predict_proba(new_train_data[testcv])
        results.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() )

if __name__=="__main__":
    main()
