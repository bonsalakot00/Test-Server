import csv #csv file module parsing file/data
import io #I/O reader
import urllib.request #module for opening URLS

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2012.csv" #URL temporary storage of data
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))  #TextIOWrapper; buffered text interface to a buffered raw stream

for row in datareader:  #loop object memory to display data
    print(row)  #print object memory

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2013.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

next(datareader)    #skip first line of data (which contains field names)

for row in datareader:
    print(row)

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2014.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

next(datareader)

for row in datareader:
    print(row)

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2015.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

next(datareader)

for row in datareader:
    print(row)

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2016.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

next(datareader)

for row in datareader:
    print(row)

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2017.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

next(datareader)

for row in datareader:
    print(row)

