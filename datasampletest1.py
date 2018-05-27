import csv #csv file module parsing file/data
import numpy as np
import pandas as pd
import requests


#upload csv
df1 = pd.read_csv('https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2012.csv')
df2 = pd.read_csv('https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2013.csv')
df3 = pd.read_csv('https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2014.csv')
df4 = pd.read_csv('https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2015.csv')
df5 = pd.read_csv('https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2016.csv')
df6 = pd.read_csv('https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2017.csv')

frames = [df1, df2, df3, df4, df5, df6]
result = pd.concat(frames, axis=0, ignore_index=True, join='inner')
result.index += 1
results = result.reset_index(drop = False)
print(results)



