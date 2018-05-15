import csv
import io
import urllib.request

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2012.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

for row in datareader:
    print(row)

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2013.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

for row in datareader:
    print(row)

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2014.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

for row in datareader:
    print(row)

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2015.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

for row in datareader:
    print(row)

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2016.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

for row in datareader:
    print(row)

url = "https://raw.githubusercontent.com/bonsalakot00/Test-Server/master/Data_2017.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

for row in datareader:
    print(row)

