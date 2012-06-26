from sklearn.ensemble import RandomForestClassifier
import csv

def main():
    #read in the training and test data
    train = csv.reader(open("Data/train.csv", "rb"))
    #the first column of the training set will be the target for the random forest classifier
    target = [x[0] for x in train]
    train = [x[1:] for x in train]
    test = csv.read_csv("Data/test.csv")

    #create and train the random forest
    #if you have a multi-core CPU you can train the model in parallel
    #by changing the below line to:
    #rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    predicted_probs = rf.predict_proba(test)
    predicted_probs = [["%f" % x[1]] for x in predicted_probs]

    #write to a results CSV file
    myWriter = csv.writer(open("Data/my_first_submission.csv"))
	myWriter.writerows[predicted_probs]

if __name__=="__main__":
    main()