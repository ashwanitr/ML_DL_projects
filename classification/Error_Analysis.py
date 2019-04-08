#Import Functions


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import sys
import shutil
import Error_Analysis_User as E

# Set These Paths

CSV_In=E.CSV_In
CSV_Out=E.CSV_Out
Images_SRC=E.Images_SRC
Images_Dest=E.Images_Dest


# Dataframe Generation

df=pd.read_csv(CSV_In)
#df.head()
df.drop(columns=["Unnamed: 0"],inplace=True)
df.reset_index(inplace=True)
df.drop(columns=['index'],inplace=True)
l1=df['Predictions']
l2=df['Actual']
l3=[]
for i in range(len(l1)):
    if l1[i]==l2[i]:
        l3.append("Correct")
    else:
        l3.append("Wrong")
df['Correctness']=l3
labels=list(set(df['Actual']))
df['Correctness'].value_counts()

#Confusion Table Matrix

Actual=df['Actual']
Predicted=df['Predictions']
conf=pd.crosstab(Actual,Predicted,rownames=['Actual'], colnames=['Predicted'], margins=True)
conf.to_csv(CSV_Out+"Confusion_Matrix.csv")


#Error Analysis Total

wrong=df.iloc[np.where(df['Correctness']=="Wrong")]
wrong.reset_index(inplace=True)
wrong.drop(columns=['index'],inplace=True)
for i in labels:
    wrong['Filename']=wrong['Filename'].str.replace(i+"/","")
wrong.to_csv(CSV_Out+"All_Errors.csv")
print("Total Number Of Wrong Images Per Class")
print(wrong['Actual'].value_counts())

#Error Analysis Individual

for i in labels:
    Label=wrong.iloc[np.where(wrong['Actual']==i)]
    Label.to_csv(CSV_Out+i+".csv")
    print("Wrongly Predicted Images Of Class "+i+" are:")
    print(Label['Predictions'].value_counts())

# Copy Error Images

if not os.path.isdir(Images_Dest+"Error_Images"):
    print("No Folder Found")
    os.mkdir(Images_Dest+"Error_Images")
dest=Images_Dest+"Error_Images/"
image=wrong['Filename']
prediction=wrong['Predictions']
for i in range(len(wrong['Filename'])):
    for j in labels:
        src_image=Images_SRC+j+"/"+image[i]
        if not os.path.isdir(dest+j):
            os.mkdir(dest+j)
        if not os.path.isdir(dest+j+"/"+prediction[i]):
            os.mkdir(dest+j+"/"+prediction[i])
        dest_image=dest+j+"/"+prediction[i]+"/"+image[i]
        if os.path.isfile(src_image):
            shutil.copy(src_image,dest_image)





