# DATA CLEANING AND PRE-PROCESSING

#GET DATA-----

import numpy as np
import pandas as pd


#df=pd.read_csv('Data_2012.csv') #read csv from local source
df1=pd.read_csv('Data_sample.csv')

#print(df.head(5)) #print first 5 of data
print(df1.head(5))
print()
print("---------Data Types---------")
print(df1.dtypes) #feature/columns data types 
print()
print("---------Data Summary---------")
print(df1.describe()) #summary of data (mean,median,max,etc....)

#SHOW OUTCOME VARIABLE/CLASSIFIER INFO-----

print()
print()
print("---------Classifier---------")
print (df1['Status'].value_counts()) #show information for the classifier/outcome


#DATA CLEANING-----

#separates the attributes and the classifier 

X=df1.drop(columns=['Name','Status']) #drop the name and status attributes (for this data, the classifier)
Y=df1.Status
print()
print()
#print(X['School'].head(5)) #printing school categories
#print(pd.get_dummies(X['School']).head(5)) #creating dummies and printing them, turning non-numeric data to numeric (1,0's)

#function for checking the unique categories and decide which categorical variables to use.

def catCheckUnique(a): 

 for col_name in X.columns:
    dummy = a
    if X[col_name].dtypes == 'object': #check if data type of attributes is not numeric
        dummy.extend([col_name])
        unique_cat = len(X[col_name].unique()) #check and count the number of unique categories in an attribute
        print("Attribute '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat)) #print the attributes and the number of its unique values
        if unique_cat > 5:
            print(X[col_name].value_counts().sort_values(ascending=False)) #Select attributes with lots of unique categories then low frequency categories changes to "Other"
            print()
 return dummy    

#function for creating dummy list for each not numeric categories

def dummy_df(df,todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x],prefix=x, dummy_na=False)
        df = df.drop(x,1)
        df = pd.concat([df,dummies], axis=1)
    return df    

a = []
b = catCheckUnique(a)
X = dummy_df(X,b)
print()
print()
print(X)
print()

#handling missing data

print(X.isnull().sum().sort_values(ascending=False)) # How much of the data are missing

