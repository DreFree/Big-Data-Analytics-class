import os
import numpy as np
import pandas as pd

iFILE="CompleteDataset.csv"
oFILE="metadata.csv"
oTEST="test_data.csv"
cFILE="corr_matrix.csv"
oBIN="bins.csv"
oNAT="nationality.csv"
oPOS="position.csv"
oLABEL="labels.csv"
PATH="./Dataset/"
nROWS=-1
FILE_WRITE=True

#using_cols={4,5,7,11,12,13,14,15,63}
#using_cols={4,5,8,9,12,13,14,15,16,17,18,19,20,21,22,23,24,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,63} #Wage
using_cols={4,5,8,9,11,13,14,15,16,17,18,19,20,21,22,23,24,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,63} #Value
DIV=10

if not(os.path.isfile(PATH+iFILE)):
    print("Directory not found:"+PATH+iFILE)
    os.kill(1)

data=[[]]                   #Data array
head=[[]]                   #Labels
Nat=[]                      #Nationality
Cl=[]                       #Bined Classes
Pos=[]                      #Preferred Postions
cor =[]                     #Correlation matrix
Club=[]                     #Clubs List
nNat=0
l=[]                        #Single from class only coreelation

def natTOnum(value):
    counter=0
    for item in Nat:
        if item==value:
            return counter
        counter+=1
    Nat.append(value)
    return counter +1
def rangeTOnum(value):
    counter=0
    for item in Cl:
        if item==value:
            return counter
        counter+=1
    Cl.append([value.left,value.right])
    return counter +1
def posTOnum(value):
    counter=0
    for item in Pos:
        if item==value:
            return counter
        counter+=1
    Pos.append(value)
    return counter +1
def clubTOnum(value):
    counter=0
    for item in Club:
        if item==value:
            return counter
        counter+=1
    Club.append(value)
    return counter +1

def translate(value,typ):
    if isinstance(value,str):  ##if is str
        
        if typ=="Value" or typ=="Wage":
            factor=1
            if value[len(value)-1]=='K':
                factor*=1000
                ans=float(value[1:len(value)-1])
            elif value[len(value)-1]=='M':
                factor*=1000000
                ans=float(value[1:len(value)-1])
            else:
                ans=float(value[1:len(value)])
         
            return int(ans*factor)
        elif typ=="Preferred Positions":
            lis=value.split(' ')
            return posTOnum(lis[0])
        elif (typ=="Nationality"):
            return natTOnum(value)
        elif (typ=="Club"):
            return clubTOnum(value)

        elif (value.find('+')>=0):
            nums=value.split('+')
            sum=int(nums[0])
            for i in range (1,len(nums)):
                sum+=int(nums[i])
            return int(sum)
        elif (value.find('-')>=0 ):
            nums=value.split('-')
            sum=int(nums[0])
            for i in range (1,len(nums)):
                sum-=int(nums[i])
            return int(sum)
        elif (value.isnumeric()):
            return int(value)
        
    
    return value
            

def readFile():
    global head
    global data
    global using_cols
    INFILE=open(PATH+iFILE,'r',encoding='utf8')
    temp=INFILE.readline()
    temp=temp.split(",")
    

    counter=0
    for i in using_cols:
        if counter!=0:
            head.append([])
        head[counter].append(temp[i])
        head[counter].append(i)
        counter+=1

    head=pd.DataFrame(head)             ## Make head dataframe
    counter=0
    while (True):
        if (counter>=nROWS and nROWS!=-1):
            break
        
        temp=INFILE.readline()
        if not(temp):
            break
        temp=temp.split(",")
        if (counter!=0):
            data.append([])
        
        for i in using_cols:
            for t in range(len(head[1])):
                if head[1][t]==i:
                    
                    data[counter].append(translate(temp[i],head[0][t]))
                    #if head[0][t]=="Aggression" and data[counter][t]>100:
                        #print("GOTCHA:" ,data[counter])
                        #a=input()
        print(data[counter])
        counter+=1
    INFILE.close()

def writeFile(F,d):
    OUTFILE=open(PATH+F,'w',encoding='utf8')
    for item in d:
        counter=0
        for value in item:
            if (counter!=0):
                OUTFILE.write(",")
            OUTFILE.write(str(value))
            counter+=1
        OUTFILE.write('\n')   
    OUTFILE.close()
    print(F,"Write to file complete...")
def writeFile2(F,d):
    OUTFILE=open(PATH+F,'w',encoding='utf8')
    for item in d:
        OUTFILE.write(str(item)+'\n')
    OUTFILE.close()
    print(F,"Write to file complete...")
    
readFile()
d=pd.DataFrame(data, columns=head[:][0])
head=pd.DataFrame(head)
total_size,w=d.shape
def wage2bin_transform():
    global d
    global head
    global total_size
    num=round(total_size*.8)
    print("Sadsdasdd")
    #print(d["Value"][:num].rank(method='first'))
    d["Class"]=pd.qcut(d["Value"].rank(method='first'),q=DIV,precision=4)
    d["Class"]=d["Class"].transform(rangeTOnum)
    d["Class"]=d["Class"].astype('int64')
    del d["Value"]
   # print("asassas",head.rows)
    head=head.drop(4,axis=0)

    #print(d["Wage"].rank(method='first'))
    #print(pd.qcut(d["Wage"].rank(method='first'),q=DIV,precision=4).value_counts())
    #print("")
    #print(pd.cut(d["Wage"],bins=DIV,precision=4).value_counts())
wage2bin_transform()

def single_corr():
    global d
    global cor
    global l
    cor=d.corr()
    cor=cor.sort_values(by=['Class'],ascending=False)
    l=cor["Class"]
    
    print("Single Correltion to feature target")
    print(l)

single_corr()

print (d.describe(include="all"))
print(d.dtypes)
#print(d)

#print(pd.DataFrame(Pos))

for inter in Cl:
    print(round(inter[0]),round(inter[1]))

def WriteTestnTrainData():
    global total_size
    global d

    limit=int(round(total_size*0.8))
    train_data=d.values[:limit][:]
    test_data=d.values[limit:][:]
    writeFile(oFILE,train_data)
    writeFile(oTEST,test_data)

if(FILE_WRITE):
    WriteTestnTrainData()
    writeFile(oLABEL,head.values)
    ##writeFile(cFILE,cor.values)
    writeFile(cFILE,[l.index,l.values])
    writeFile2(oNAT,Nat)
    writeFile2(oPOS,Pos)
    writeFile2(oBIN,Cl)
