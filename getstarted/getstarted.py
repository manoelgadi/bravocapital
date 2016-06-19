# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 19:15:51 2016

@author: ivanmartin
"""


import fagroupa

# DROPBOX!
import urllib.request as urllib2
import os.path
import time
import random
import pandas

while not os.path.exists('dev.csv') or not os.path.exists('oot0.csv'):
    time.sleep (3*random.random()); #Sleeping less than 3 seconds before going to Dropbox - avoid too many students at once.
    if not os.path.exists('dev.csv'):
        print ("DOWLOADING FILE dev.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://dl.dropboxusercontent.com/u/28535341/dev.csv")
        output = open('dev.csv','wb')
        output.write(csvfile.read())
        output.close()
    if not os.path.exists('oot0.csv'):
        print ("DOWLOADING FILE oot0.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://dl.dropboxusercontent.com/u/28535341/oot0.csv")
        output = open('oot0.csv','wb')
        output.write(csvfile.read())
        output.close()
print("READING DATA...")
csvfile = urllib2.urlopen("https://s3-us-west-2.amazonaws.com/martinmaseda/devfa.csv")
output = open('devfa.csv','wb')
output.write(csvfile.read())
output.close()
df = pandas.DataFrame.from_csv("./devfa.csv", sep = ";")

print("MACHINE LEARNING METHODS")
fagroupa.MachineLearningMethods()

print("MANUAL CLEANING")
fagroupa.CleaningManualClass("./dev.csv")

print("AUTO CLEANING")
fagroupa.CleaningAutoClass("./dev.csv")

print("PCA CREATION")
fagroupa.performPCAAndRatios(df)

print("AUTO GENERATING VARIABLES WITH RAND FOREST")
fagroupa.PerformRandForest(df)

print("BINNING DUMMY CREATION")
fagroupa.binningDummyCreattion(df)

print("All sections executed")





