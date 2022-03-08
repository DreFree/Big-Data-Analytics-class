import numpy as numpy
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.multiclass import OneVsOneClassifier
from sklearn import decomposition
from sklearn import svm
from sklearn import datasets

iFILE="metadata.csv"
iTEST="test_data.csv"
PATH="./Dataset/"
data=[[]]
Y=[]
test=[[]]
test_Y=[]

if not(os.path.isfile(PATH+iFILE)):
    print("Directory not found:"+PATH+iFILE)
    os.kill(1)

def ReadMetaData():
    global data
    with open(PATH+iFILE,"r") as f:
        counter=0
        for temp in f:
            if counter!=0:
                data.append([])
            temps=temp.split(",")
            for item in temps:
                item=item.rstrip('\n')
                if (item.isnumeric()):
                    data[counter].append(int(item))
                elif(item.isalpha()):
                    data[counter].append(item)
                else:
                    data[counter].append(float(item))
            counter+=1
        f.close()
def ReadTestData():
    global test
    with open(PATH+iTEST,"r") as f:
        counter=0
        for temp in f:
            if counter!=0:
                test.append([])
            temps=temp.split(",")
            for item in temps:
                item=item.rstrip('\n')
                if (item.isnumeric()):
                    test[counter].append(int(item))
                elif(item.isalpha()):
                    test[counter].append(item)
                else:
                    test[counter].append(float(item))
            counter+=1
        f.close()


ReadMetaData()
data=pd.DataFrame(data)
l,nof=data.shape
Y=data[nof-1].astype(int)
del data[nof-1]
nof-=1
print("Data dimensions: ",l,nof)

ReadTestData()
test=pd.DataFrame(test)
t_l,t_nof=test.shape
test_y=test[t_nof-1].astype(int)
del test[t_nof-1]
t_nof-=1
print("Test dimensions: ",t_l,t_nof)
def writeTOfile(file,data,Y):
    OUTFILE=open("./Dataset/"+file,'w',encoding='utf8')
    i=0
    for item in data:
        counter=0
        for value in item:
            if (counter!=0):
                OUTFILE.write(",")
            OUTFILE.write(str(value))
            counter+=1
        OUTFILE.write(',')
        OUTFILE.write(str(Y[i]))
        OUTFILE.write('\n')
        i+=1   
    OUTFILE.close()
    print(file,"Write complete")

def applyPCA():
    global data
    global test
    print("DATA:")
    print(data)
    to_fit=data.iloc[:,5:34]
    to_fit=to_fit.join(data.iloc[:,2])
    print(to_fit)
    data=data.drop(to_fit.columns,axis=1)
    print("DATA:")
    print(data)
    print(data.dtypes)

    print("TEST:")
    test_to_fit=test.iloc[:,5:34]
    test_to_fit=test_to_fit.join(test.iloc[:,2])
    test=test.drop(test_to_fit.columns,axis=1)
    print(test)

    PCA=decomposition.PCA(n_components=3)
    PCA2=decomposition.PCA(n_components=3)

    fit_tran=PCA.fit_transform(to_fit.values)
    data=data.join(pd.DataFrame(fit_tran,index=data.index,columns=["t1","t2","t3"]))
    print("DATA:")
    print(data)

    fit_test=PCA2.fit_transform(test_to_fit)
    test=test.join(pd.DataFrame(fit_test,index=test.index,columns=["t1","t2","t3"]))
    print("TEST:")
    print(test)

applyPCA()
writeTOfile("final_meta.csv",data.values,Y)
writeTOfile("final_test_meta.csv",test.values,test_y)
print(data)
print(test)
clf=svm.SVC(decision_function_shape='ovo', kernel='linear')
clf.fit(data.values,Y)
print("Train Complete.")
print("Predicting Data...")
count=0

t_l,t_nof=test.shape
ans=clf.predict(test.values)
for i in range(t_l):
    if ans[i]!=test_y[i]:
        count+=1

print("Number of train data: ",l)
print("Number of test data:  ",t_l)
print("Prediction Accuracy:   ",(1-(count/t_l))*100,"%",sep='')