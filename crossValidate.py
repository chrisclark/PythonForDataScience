from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import logloss
import numpy as np

def main():
    #read in  data, parse into training and target sets
    train = csv.reader(open("Data/train.csv", "rb"))
    target = np.array( [x[0] for x in train] )
    train = np.array( [x[1:] for x in train] )

    #In this case we'll use a random forest, but this could be any classifier
    cfr = RandomForestClassifier(n_estimators=100)

    #Simple K-Fold cross validation. 10 folds.
    cv = cross_validation.KFold(len(train), k=10, indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() )

if __name__=="__main__":
    main()