import pandas as pd
import os

FILE="CompleteDataset.csv"
PATH="./Dataset/"
nROWS=15

if not(os.path.isfile(PATH+FILE)):
    print("Directory not found:"+PATH+FILE)
    os.kill(1)

if nROWS>0:
    df=pd.read_csv(PATH+FILE,nrows=nROWS)
else:
    df=pd.read_csv(PATH+FILE)



print("Sample DAta")
print (df.iloc[:nROWS])
head=df.columns

del df["Unnamed: 0"]
del df["ID"]
del df["Photo"]
del df["Name"]
del df["Flag"]
del df["Wage"]
del df["Club Logo"]
del df["Overall"]
del df['GK diving']
del df['GK handling']
del df['GK kicking']
del df['GK positioning']
del df['GK reflexes']
del df['CAM']
del df['CB']
del df['CDM']
del df['CF']
del df['CM']
del df['LAM']
del df['LB']
del df['LCB']
del df['LCM']
del df['LDM']
del df['LF']
del df['LM']
del df['LS']
del df['LW']
del df['LWB']
del df['RAM']
del df['RB']
del df['RCB']
del df['RCM']
del df['RDM']
del df['RF']
del df['RM']
del df['RS']
del df['RW']
del df['RWB']
del df['ST']

print("Sample DAta")
print (df.iloc[:nROWS])

print ("Wage Summary:")
print (df.describe(include="all"))
print ("Dimensions ",df.shape)
print("\nIndexes to be used: ")

counter=0
for item in head:
    if item in df.columns:
        print(counter,end=',')
    counter+=1
print()
print(df.columns)

print (df['Preferred Positions'].unique)