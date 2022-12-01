import time
start_time = time.time()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import resample
warnings.filterwarnings("ignore")
df = pd.read_csv('creditcard.csv')
#print(df.head())

# Scaling the amount and Time, as those two features do not seem to be normalized
robust_scaler = RobustScaler()
df['Amount'] = robust_scaler.fit_transform(df[['Amount']])
df['Time'] = robust_scaler.fit_transform(df[['Time']])

# print(df.head())

#Printing current stats of the data
print(df['Class'].value_counts(normalize=True))
print(df['Class'].value_counts())

#Creating the data into X, which holds all the features of the dataset, 
# and Y which holds the class labels of the dataset
y_final = df['Class']
X_final = df.drop(['Class'], axis=1)

X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(X_final, y_final, test_size=0.20, random_state=100, stratify=y_final)
df = pd.concat([X_final_train, y_final_train], axis=1)
df.reset_index(drop=True, inplace=True)

class CreditCardFraudPrediction:
    def __init__(self, data, sample_frac):
        self.data = data
        self.sample_frac = sample_frac

    def sample(self):
        return self.data.sample(frac=self.sample_frac)

    def LocalOutlierFactor(self, print_to_terminal=True):
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

    def IsolationForest(self, print_to_terminal=True):
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


    def logistic_regression(self, X, y, folds=5, print_to_terminal=True):
        model_name = "Logistic Regression"
        self.run_model(X, y, LogisticRegression(), model_name, folds)
        

    def k_nearest(self, X, y, folds=5, print_to_terminal=True):
        model_name = "K Nearest Neighbors"
        self.run_model(X, y, KNeighborsClassifier(), model_name, folds)
        
    def decision_tree(self, X, y, folds=5, print_to_terminal=True):
        model_name = "Decision Tree"
        self.run_model(X, y, DecisionTreeClassifier(),model_name, folds)

    def naive_bayes(self, X, y, folds=5, print_to_terminal=True):
        model_name = "Naive Bayes"
        self.run_model(X, y, GaussianNB(), model_name,  folds)

    def random_forests(self, X, y, folds=5, print_to_terminal=True):
        model_name = "Random Forests"
        self.run_model(X, y, RandomForestClassifier(), model_name,  folds)

    def run_model(self, X, y, model, model_name, folds=5):
        k_fold = StratifiedKFold(n_splits=folds, random_state=100, shuffle=True)
        cross_val_scores = []
        precision_scores = []
        recall_scores = []
        roc_auc_scores = []
        f1_scores = []
        estimators = []
        for train_index, test_index in k_fold.split(X, y):
            X_train, X_test = pd.DataFrame(data=X, index=train_index), pd.DataFrame(data=X, index=test_index)
            y_train, y_test = pd.DataFrame(data=y, index=train_index), pd.DataFrame(data=y, index=test_index)
            # model = GaussianNB()
            model.fit(X_train, y_train)
            estimators.append(model)
            scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=5)
            cross_val_scores.append(scores)
            y_pred = model.predict(X_test)
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            roc_auc_scores.append(roc_auc_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        print(f'============================= {model_name} =============================')
        print('Mean cross validation score: {}'.format(np.array([cross_val_scores]).mean()))
        print('Mean precision score: {}'.format(np.array([precision_scores]).mean()))
        print('Mean Recall score: {}'.format(np.array([recall_scores]).mean()))
        print('Mean ROC-AUC score: {}'.format(np.array([roc_auc_scores]).mean()))
        print('Mean F1 score: {}'.format(np.array([f1_scores]).mean()))
        best_iteration = recall_scores.index(max(recall_scores))
        best_estimator = estimators[best_iteration]
        print('******* Real test dataset metrics *******')
        y_final_pred = best_estimator.predict(X_final_test)
        print('Accuracy score for the real test set:\n', accuracy_score(y_final_test, y_final_pred))
        print('confusion matrix for the real test set:\n', confusion_matrix(y_final_test, y_final_pred))
        print('Classification report for the real test set:\n', classification_report(y_final_test, y_final_pred))

    def mlp(self, X, y):
        scaler = StandardScaler()
        scaler.fit(X_final_train)
        X_train = scaler.transform(X)
        X_test = scaler.transform(X_final_test)

        MLP = MLPClassifier(learning_rate_init = 0.0001,hidden_layer_sizes = (25,25,25,25))

        MLP.fit(X_train,y)
        y_final_pred = MLP.predict(X_test)
        precision_scr = precision_score(y_final_test, y_final_pred)
        recall_scr = recall_score(y_final_test, y_final_pred)
        roc_auc_scr = roc_auc_score(y_final_test, y_final_pred)
        f1_scr = f1_score(y_final_test, y_final_pred)
        print(f'============================= Multilayer perceptron =============================')
        print('******* Real test dataset metrics *******')
        print(f'Precision score: {precision_scr}')
        print(f'Recall score: {recall_scr}')
        print(f'ROC-AUC score: {roc_auc_scr}')
        print(f'F1 score: {f1_scr}')
        print('Accuracy score for the real test set:\n', accuracy_score(y_final_test, y_final_pred))
        print('confusion matrix for the real test set:\n', confusion_matrix(y_final_test, y_final_pred))
        print('Classification report for the real test set:\n', classification_report(y_final_test, y_final_pred))
        


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



def do_sampling(sampler):
    X_sampled, y_sampled = sampler.fit_resample(X_final_train, y_final_train)
    df_X_sampled = pd.DataFrame(X_sampled, columns=X_final_train.columns)
    df_y_sampled = pd.DataFrame(y_sampled, columns=['Class'])

    df_sampler_data = pd.concat([df_X_sampled, df_y_sampled], axis=1)
    print(df_y_sampled['Class'].value_counts())
    del df_X_sampled, df_y_sampled

    y = df_sampler_data['Class']
    X = df_sampler_data.drop(['Class'], axis=1)

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    return (X, y)

def run_models(X, y):
    clf = CreditCardFraudPrediction(data, 0.1)
    clf.logistic_regression(X = X, y = y)
    clf.k_nearest(X = X, y = y)
    clf.decision_tree(X = X, y = y)
    clf.naive_bayes(X = X, y = y)
    clf.random_forests(X = X, y = y)
    clf.mlp(X = X, y = y)


marking = "*********************************"
print(f'{marking} Random under sampling {marking}')
X, y = do_sampling(RandomUnderSampler())
run_models(X, y)
rUS = time.time()
print("Process finished --- %s seconds ---" % (rUS - start_time))

print(f'{marking} Under sampling using Nearmiss algorithm {marking}')
X, y = do_sampling(NearMiss())
run_models(X, y)
nmUS = time.time()
print("Process finished --- %s seconds ---" % (nmUS - rUS))


print(f'{marking} Random over sampling {marking}')
X, y = do_sampling(RandomOverSampler())
run_models(X, y)
rOS = time.time()
print("Process finished --- %s seconds ---" % (rOS - nmUS))

print(f'{marking} Random over sampling using SMOTE {marking}')
X, y = do_sampling(SMOTE(sampling_strategy='minority'))
run_models(X, y)
rOSS = time.time()
print("Process finished --- %s seconds ---" % (rOSS - rOS))

print("Process finished --- %s seconds ---" % (time.time() - start_time))