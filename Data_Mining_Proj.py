import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class CreditCardFraudPrediction:
    def __init__(self, data, sample_frac):
        self.data = data
        self.sample_frac = sample_frac

    def sample(self):
        return self.data.sample(frac=self.sample_frac)

    def LocalOutlierFactor(self, print_to_terminal):
        df = self.sample()
        # data frames with class exlusivity
        fraud = df[df['Class'] == 1] # Number of fraudulent transactions
        valid = df[df['Class'] == 0] # Number of valid transactions
        #sample from both or exclusively to see if model its changed

        outlier_fraction = len(fraud)/float(len(valid))

        X = df.drop('Class',axis = 1) # X is input
        y = df['Class'] # y is output

        a = LocalOutlierFactor(n_neighbors = 20,contamination = outlier_fraction)
        y_prediction1 = a.fit_predict(X) # Fitting the model.
        y_prediction1[y_prediction1 == 1] = 0 # Valid transactions are labelled as 0.
        y_prediction1[y_prediction1 == -1] = 1 # Fraudulent transactions are labelled as 1.
        errors1 = (y_prediction1 != y).sum() # Total number of errors is calculated.
        if(print_to_terminal):
            print("---Local Outlier Factor---")
            print(errors1)
            print(accuracy_score(y_prediction1,y))
            print(classification_report(y_prediction1,y))

        return accuracy_score(y_prediction1,y)

    def IsolationForest(self, print_to_terminal):
        df = self.sample()
        # data frames with class exlusivity
        fraud = df[df['Class'] == 1] # Number of fraudulent transactions
        valid = df[df['Class'] == 0] # Number of valid transactions
        #sample from both or exclusively to see if model its changed

        outlier_fraction = len(fraud)/float(len(valid))

        X = df.drop('Class',axis = 1) # X is input
        y = df['Class'] # y is output

        b = IsolationForest(max_samples = len(X),contamination = outlier_fraction).fit(X) # Fitting the model.
        y_prediction2 = b.predict(X) # Prediction using trained model.
        y_prediction2[y_prediction2 == 1] = 0 # Valid transactions are labelled as 0.
        y_prediction2[y_prediction2 == -1] = 1 # Fraudulent transactions are labelled as 1.
        errors2 = (y_prediction2 != y).sum() # Total number of errors is calculated.
        if(print_to_terminal):
            print("---Isolation Forest---")
            print(errors2)
            print(accuracy_score(y_prediction2,y))
            print(classification_report(y_prediction2,y))

        return accuracy_score(y_prediction2,y)





# read data into dataframe
data = pd.read_csv('creditcard.csv')

clf = CreditCardFraudPrediction(data, 0.1)
sum_accuracy = 0
for i in range(1):
    sum_accuracy += clf.LocalOutlierFactor(print_to_terminal = True)
avg_accuracy = sum_accuracy/1
print("Local Outlier Factor Accuracy Score:", avg_accuracy)

sum_accuracy = 0
for i in range(1):
    sum_accuracy += clf.IsolationForest(print_to_terminal = True)
avg_accuracy = sum_accuracy/1
print("Isolation Forest Accuracy Score:", avg_accuracy)



# clf.IsolationForest()



#sample from df
# df = df.sample(frac=0.1) # sample 10% of credit card csv

# # data frames with class exlusivity
# fraud = df[df['Class'] == 1] # Number of fraudulent transactions
# valid = df[df['Class'] == 0] # Number of valid transactions
# #sample from both or exclusively to see if model its changed

# outlier_fraction = len(fraud)/float(len(valid))

# X = df.drop('Class',axis = 1) # X is input
# y = df['Class'] # y is output

# a = LocalOutlierFactor(n_neighbors = 20,contamination = outlier_fraction)
# y_prediction1 = a.fit_predict(X) # Fitting the model.
# y_prediction1[y_prediction1 == 1] = 0 # Valid transactions are labelled as 0.
# y_prediction1[y_prediction1 == -1] = 1 # Fraudulent transactions are labelled as 1.
# errors1 = (y_prediction1 != y).sum() # Total number of errors is calculated.
# print(errors1)
# print(accuracy_score(y_prediction1,y))
# print(classification_report(y_prediction1,y))

# b = IsolationForest(max_samples = len(X),contamination = outlier_fraction).fit(X) # Fitting the model.
# y_prediction2 = b.predict(X) # Prediction using trained model.
# y_prediction2[y_prediction2 == 1] = 0 # Valid transactions are labelled as 0.
# y_prediction2[y_prediction2 == -1] = 1 # Fraudulent transactions are labelled as 1.
# errors2 = (y_prediction2 != y).sum() # Total number of errors is calculated.
# print(errors2)
# print(accuracy_score(y_prediction2,y))
# print(classification_report(y_prediction2,y))
