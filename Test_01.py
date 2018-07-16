
# HANDLE MISSING DATA

# DATA CLEANING AND PRE-PROCESSING

# GET DATA-----
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

#Decision Trees
import pydotplus
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Dataframe
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df1=pd.read_csv('NBA_2010.csv')
df2=pd.read_csv('NBA_2011.csv')
df3=pd.read_csv('NBA_2012.csv')
df4=pd.read_csv('NBA_2013.csv')
df5=pd.read_csv('NBA_2014.csv')
df6=pd.read_csv('NBA_2015.csv')
frames = [ df1, df2, df3, df4, df5, df6]
combine = pd.concat(frames, axis=0, ignore_index=True, join='inner')
df = combine.reset_index(drop=True)
df.index += 1
print(df.head(5))
print()
print(df.tail(5))
print()

input("Press Enter to Continue...")

# INITIAL DATA SUMMARY-----

print("--------- Data Summary ---------")
print(df.info())
print()

input("Press Enter to Continue...")

# REPLACE COLUMN NAMES----

df.columns = ['Name', 'Position', 'Country', 'Number of Seasons',
              'College', 'Height', 'Wingspan', 'Vertical Reach',
              'Games Played', 'Games Started', 'Minutes per Game',
              'Field Goals Attempted', 'Field Goals Made', 'Field Goals Percentage',
              '3-Points Attempted', '3-Points Made', '3-Points Percentage',
              'Free-throws Percentage', 'Rebounds per Game', 'Assists per Game',
              'Steals per Game', 'Blocks per Game', 'Points per Game', 'Turnovers per Game',
              'Fouls per Game', 'Class']
print(df.info())
print()

input("Press Enter to Continue...")

# REMOVE ROWS WITH NO STAT VALUES----

df.dropna(how='all', subset=['Height', 'Position'], inplace=True)
df.dropna(how='all', subset=['Games Played', 'Games Started', 'Minutes per Game',
                             'Field Goals Attempted', 'Field Goals Made', 'Field Goals Percentage',
                             '3-Points Attempted', '3-Points Made', '3-Points Percentage',
                             'Free-throws Percentage', 'Rebounds per Game', 'Assists per Game',
                             'Steals per Game', 'Blocks per Game', 'Points per Game'], inplace=True)
df.reset_index(inplace=True)
df.index += 1
print(df.head(5))
print()
print(df.tail(5))
print()

input("Press Enter to Continue...")

# HANDLING WRONG DATA TYPES----

df.replace({'-': ''}, regex=True, inplace=True)  # remove unnecessary character in strings


def inches(prime_str):
    try:
        result = re.match(r"([0-9]+)'([0-9]{0,2}?)[\"]?\Z", prime_str)
        feet = int(result.group(1))
        if result.group(2) == "":
            inches = 0
        else:
            inches = int(result.group(2))

        return feet * 12 + inches
    except:
        return None


df["Height"] = df["Height"].apply(inches)
df["Wingspan"] = df["Wingspan"].apply(inches)
df["Vertical Reach"] = df["Vertical Reach"].apply(inches)
print(df.head(5))
print()
print(df.info())

input("Press Enter to Continue...")

# HANDLE CATEGORICAL DATA----

for col_name in df.columns:
    if df[col_name].dtypes == 'object':  # check if data type of attributes is not numeric
        unique_cat = len(df[col_name].unique())  # check and count the number of unique categories in an attribute
        print("Attribute '{col_name}' has {unique_cat} unique categories".format(col_name=col_name,
                                                                                 unique_cat=unique_cat))  # print the attributes and the number of its unique values

print()
print(df['Position'].value_counts(ascending=False))
print()
print(df['Country'].value_counts(ascending=False))
print()
print(df['College'].value_counts(ascending=False))

input("Press Enter to Continue...")

# reduce unique categories of country
df['Country'] = ['USA' if x == 'USA' else 'Others' for x in df['Country']]
print()
print(df['Country'].value_counts(ascending=False))

# reduce unique categories of position
df.replace(['PG', 'SG', 'PG/SG', 'SG/PG', 'SG/SF', 'SF', 'SF/SG', 'SF/PF', 'PF', 'PF/SF', 'PF/C', 'C/PF', 'C'],
           ['Guard', 'Guard', 'Guard', 'Guard', 'Guard', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward',
            'Forward', 'Center', 'Center'], inplace=True)
print()
print(df['Position'].value_counts(ascending=False))

# reduce unique categories of college
df.loc[~df.College.isin(['Kentucky', 'Duke']), 'College'] = 'Others'
print()
print(df['College'].value_counts(ascending=False))

# change object datatypes to categorical
for col in ['Country', 'Position', 'College']:
    df[col] = df[col].astype('category')

# change some datatypes
df['Number of Seasons'] = df['Number of Seasons'].fillna(0)
df['Number of Seasons'] = df['Number of Seasons'].astype('int32')
df['Height'] = df['Height'].astype('float64')

print()
print(df.info())

# Fill missing data (Vertical Reach)
df['Vertical Reach']= df['Vertical Reach'].fillna(0)

# HANDLE MISSING DATA

print(df.isnull().sum().sort_values(ascending=False))

#Relationship (Correlation R)
print()
print("Correlation Between Points per Game and Minutes Per Game")
print(df['Points per Game'].corr(df['Minutes per Game']))
print("Correlation Between Points per Game and Games Played")
print(df['Points per Game'].corr(df['Games Played']))
print("Correlation Points per Game and Field Goals Attempted")
print(df['Points per Game'].corr(df['Field Goals Attempted']))
print("Correlation Vertical Reach and Blocks Per Game")
print(df['Vertical Reach'].corr(df['Blocks per Game']))
print("Correlation Games Played and Games Started")
print(df['Games Played'].corr(df['Games Started']))
print("Correlation Height and Rebounds per Game")
print(df['Height'].corr(df['Rebounds per Game']))
print()

input("Press Enter to Continue...")


#SEABORN  (EDA)
#Single plot for Positions
'''fg = sns.factorplot('Position', data=df, kind='count', aspect=1.5)
fg.set_xlabels('Position')
plt.show(fg)

#Single plot for Colleges
tg = sns.factorplot('College', data=df, kind='count', aspect=1.5)
tg.set_xlabels('College')
plt.show(tg)

#print(df.groupby('Number of Seasons')['Number of Seasons'].count())

#Single plot of all seasons played by each player
sg = sns.factorplot('Number of Seasons', data=df, kind='count', aspect=1.5)
sg.set_xlabels('Number of Seasons')
plt.show(sg)

#Points made per minutes played
mpg = sns.lmplot('Points per Game', 'Minutes per Game', data=df)
mpg.set_xlabels('Points per Game')
plt.show(mpg)

#Points made per games played
gpg = sns.lmplot('Points per Game', 'Games Played', data=df)
gpg.set_xlabels('Points per Game')
plt.show(gpg)

#Points made and field goals attempt
spg = sns.lmplot('Points per Game', 'Field Goals Attempted', data=df, hue='Position')
spg.set_xlabels('Points per Game')
plt.show(spg)

#Vertical Reach and blocks accumulated
wpg = sns.lmplot('Vertical Reach', 'Blocks per Game', hue='Position', data=df)
wpg.set_xlabels('Vertical Reach')
plt.show(wpg)

#Kde Plot, Distribution of Field Goals Percentage by position
figA = sns.FacetGrid(data=df, hue='Position', aspect=4)
figA.map(sns.kdeplot, 'Field Goals Percentage', shade=True)
highestA = df['Field Goals Percentage'].max()
figA.set(xlim=(0,highestA))
figA.set(title='Distribution of Field Goals Percentage by Position')
figA.add_legend()
plt.show(figA)

#Kde Plot, Distribution of minutes per game by position
figB = sns.FacetGrid(data=df, hue='Position', aspect=4)
figB.map(sns.kdeplot, 'Minutes per Game', shade=True)
highestB = df['Minutes per Game'].max()
figB.set(xlim=(0,highestB))
figB.set(title='Distribution of Minutes per Game by Position')
figB.add_legend()
plt.show(figB)

#relationship of position and vertical reach
figC = sns.FacetGrid(data=df, hue='Position', aspect=4)
figC.map(sns.kdeplot, 'Vertical Reach', shade=True)
highestC = df['Vertical Reach'].max()
figC.set(xlim=(0,highestC))
figC.set(title='Distribution of Vertical Reach by Position')
figC.add_legend()
plt.show(figC)

cg = sns.lmplot('Games Played', 'Games Started', data=df, hue='College')
cg.set_xlabels('Games Played')
plt.show(cg)

#relationship of height and rebounds
hg = sns.lmplot('Height', 'Rebounds per Game', data=df, hue='Position')
hg.set_xlabels('Height')
plt.show(hg)'''

#Binarize Data
#separate array into input and output components
#bindata = preprocessing.Binarizer(threshold=1.5).transform(df)
#print('Binarized data:\n\n', bindata)

'''testcol = ['Name','Position','Country','Number of Seasons','College','Height','Wingspan','Vertical Reach','Class','Games Started', 'Minutes per Game',
                             'Field Goals Attempted', 'Field Goals Made', 'Field Goals Percentage',
                             '3-Points Attempted', '3-Points Made', '3-Points Percentage',
                             'Free-throws Percentage', 'Rebounds per Game', 'Assists per Game',
                             'Steals per Game', 'Blocks per Game', 'Points per Game', 'Turnovers per Game',
                             'Fouls per Game']

#fill NaN data avoid false-positive error
df[testcol] = df[testcol].fillna(0)'''

'''binstan_var = ['Games Played', 'Games Started', 'Minutes per Game',
                             'Field Goals Attempted', 'Field Goals Made', 'Field Goals Percentage',
                             '3-Points Attempted', '3-Points Made', '3-Points Percentage',
                             'Free-throws Percentage', 'Rebounds per Game', 'Assists per Game',
                             'Steals per Game', 'Blocks per Game', 'Points per Game', 'Turnovers per Game',
                             'Fouls per Game']

nonbinstan_var = ['Name','Position','Country','Number of Seasons','College','Height','Wingspan','Vertical Reach','Class']
X = df[binstan_var].values
Y = df[nonbinstan_var].values

binarizer = Binarizer(threshold=1).fit(X)
binaryX = binarizer.transform(X)

np.set_printoptions(precision=3)
print("Transformed Binary Data")
print(binaryX[0:5,:])

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

np.set_printoptions(precision=3)
print("Transformed Standard Data")
print(rescaledX[0:5, :])'''

#drop column
#GET DUMMIES

testcol = ['Number of Seasons','Height','Wingspan','Vertical Reach','Games Started', 'Minutes per Game',
                             'Field Goals Attempted', 'Field Goals Made', 'Field Goals Percentage',
                             '3-Points Attempted', '3-Points Made', '3-Points Percentage',
                             'Free-throws Percentage', 'Rebounds per Game', 'Assists per Game',
                             'Steals per Game', 'Blocks per Game', 'Points per Game', 'Turnovers per Game',
                             'Fouls per Game']

Country_Dummies = pd.get_dummies(df.Country, prefix="Country").iloc[:, 1:]
df = pd.concat([df, Country_Dummies], axis=1)
Position_Dummies = pd.get_dummies(df.Position, prefix="Position").iloc[:, 1:]
df = pd.concat([df, Position_Dummies], axis=1)
College_Dummies = pd.get_dummies(df.College, prefix="College").iloc[:, 1:]
df = pd.concat([df, College_Dummies], axis=1)
Class_Dummies = pd.get_dummies(df.Class, prefix="Class").iloc[:, 1:]
df = pd.concat([df, Class_Dummies], axis=1)
replace_columns = ['Name','Country','College','Position','Class']
df.drop(replace_columns, axis=1, inplace=True)
df.dropna(subset=['Number of Seasons',
              'Height', 'Wingspan', 'Vertical Reach',
              'Games Played', 'Games Started', 'Minutes per Game',
              'Field Goals Attempted', 'Field Goals Made', 'Field Goals Percentage',
              '3-Points Attempted', '3-Points Made', '3-Points Percentage',
              'Free-throws Percentage', 'Rebounds per Game', 'Assists per Game',
              'Steals per Game', 'Blocks per Game', 'Points per Game', 'Turnovers per Game',
              'Fouls per Game',], inplace=True)
#df[testcol] = df[testcol].fillna(0)
print(df.info())
print(df.head(5))
print()
print(df.tail(5))
print()




#Decision Trees
'''y=df.iloc[:,[0]]
x=df.iloc[:,[5,6,7,8]]
x = np.reshape(x.shape[29:261])
x = np.tranpose()

#Splitting the data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)'''

'''df.dropna(axis=0, inplace=True)
print(df.shape)

y = df[df.columns[0]].copy()
df.drop(df.columns[0], axis=1,inplace=True)

y = y.map({'p':0, 'e':1})

X = pd.get_dummies(df)


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=7)

clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=30,splitter="best")
clf = clf.fit(X_train, y_train)

y_pred=clf.predict(X_train)
accuracy=accuracy_score(y_test,y_pred)
print(str(accuracy*100)+"% accuracy")

#visualization of Training set results
height=pd.Series(y).value_counts(normalize=True)
plt.bar(range(3),height.tolist()[::-1],1/1.5,color='green',label="Classes", alpha=0.8)
plt.xlabel('Test')
plt.xlabel('Test2')
plt.legend()
plt.show()'''

features = np.array(df.iloc[:, 0:8])
labels = np.array(df.iloc[:, -1])

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, stratify=labels)

scaler = StandardScaler()
scaler.fit(x_train)
normalized_train = scaler.transform(x_train)
normalized_test = scaler.transform(x_test)

clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=30,splitter="best")
clf = clf.fit(x_train, y_train)

model = tree.DecisionTreeClassifier()

seed = 10
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train, validate in kfold.split(x_train, y_train):
    model.fit(x_train[train], y_train[train])
    accuracy = model.score(x_train[validate], y_train[validate])
    print('accuracy : {}'.format(accuracy))
    cvscores.append(accuracy)

print('CV accuracy : {}, CV stddev : +/- {}'.format(np.mean(cvscores), np.std(cvscores)))

tree.export_graphviz(clf, out_file='tree.dot')

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())















