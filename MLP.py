from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
###read in X_train
file = open("creditcardedited.csv",'r')
raw_data = []
raw_true = []
raw_false = []
raw_true_class = []
raw_false_class = []
raw_class = []
temp = []
fraud = -1
for line in file:
    temp.append([x for x in line.split(',')])
for line in temp:    
    temp2 = []
    for i in range(0,31,1):
        if i != 30:
            temp2.append(float(line[i]))
        elif line[i] == '"0"\n':
            fraud = 0
            raw_class.append(0)
        elif line[i] == '"1"\n':
            fraud = 1
            raw_class.append(1)
    raw_data.append(temp2)
    #seperate place that holds both values
    if fraud == 1:
        raw_true.append(temp2)
        raw_true_class.append(1)
    elif fraud == 0:
        raw_false.append(temp2)
        raw_false_class.append(0)
file.close()
print("# of features is")
print(len(raw_data[0]))
print("# of samples is ")
print(str(len(raw_data)))
Xtrain = []
Xtest = []
Ytrain = []
Ytest = []

for i in range(0,int(len(raw_true)/2),1):
    Xtrain.append(raw_true[i])
    Ytrain.append(raw_true_class[i])

for i in range(int(len(raw_true)/2),len(raw_true),1):
    Xtest.append(raw_true[i])
    Ytest.append(raw_true_class[i])

for i in range(0,int(len(raw_false)/2),1):
    Xtrain.append(raw_false[i])
    Ytrain.append(raw_false_class[i])

for i in range(int(len(raw_false)/2),len(raw_false),1):
    Xtest.append(raw_false[i])
    Ytest.append(raw_false_class[i])

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

MLP = MLPClassifier(learning_rate_init = 0.0001,hidden_layer_sizes = (25,25,25,25))

MLP.fit(Xtrain,Ytrain)
Ypred = MLP.predict(Xtest)
total = 0.0
right = 0.0
right1 = 0.0
right0 = 0.0
guess0 = 0.0
guess1 = 0.0
total1 = 0.0
total0 = 0.0
#these are used to calculate precision and recall
for i in range(0,len(Ypred),1):
    total += 1
    if Ypred[i] == 1:
        guess1 += 1
    if Ypred[i] == 0:
        guess0 += 1
    if Ytest[i] == 1:
        total1 += 1
    if Ytest[i] == 0:
        total0 += 1
    if Ypred[i] == Ytest[i]:
        right += 1
        if Ypred[i] == 1:
            right1 += 1
        if Ypred[i] == 0:
            right0 += 1
precision1 = float(right1)/float(guess1)
recall1 = float(right1)/float(total1)
F11 = 2.0 * (precision1*recall1)/(precision1+recall1)
precision0 = float(right0)/float(guess0)
recall0 = float(right0)/float(total0)
F10 = 2.0 * (precision0*recall0)/(precision0+recall0)
print("# of features Xtrain")
print(len(Xtrain[0]))
print("# of samples is ")
print(str(len(Xtrain)))
print("# of features Xtest")
print(len(Xtest[0]))
print("# of samples is ")
print(str(len(Xtest)))
print("ACCURACY")
print(float(right)/float(total))
print("PRECISION FRAUD")
print(precision1)
print("RECALL FRAUD")
print(recall1)
print("F1 FRAUD")
print(F11)
print("PRECISION VALID")
print(precision0)
print("RECALL VALID")
print(recall0)
print("F1 VALID")
print(F10)