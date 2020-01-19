# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 09:11:22 2020

@author: soumik
"""

import os
import random

from shutil import copyfile


#directory for training data
train_dir=r"C:\Users\soumik\Desktop\Data\DataShowCase\tfData\train"

#directory for test data
test_dir=r"C:\Users\soumik\Desktop\Data\DataShowCase\tfData\val"

#directory where test data resides
inp_dir = r"C:\Users\soumik\Desktop\Data\DataShowCase\CleanDataShowCase"


# create directories in the test and train folder
def createDirectory():

    for file in os.listdir(inp_dir):
        filePath = os.path.join(inp_dir,file)
        if os.path.isdir(filePath):
            valPath=os.path.join(train_dir,file)
            trainPath = os.path.join(test_dir,file)
            os.mkdir(valPath)
            os.mkdir(trainPath)      

# Get text for Label File

def getLabelFile():

    for file in os.listdir(inp_dir):
        filePath = os.path.join(inp_dir,file)
        if os.path.isdir(filePath):
            print(file)

# split data into 80-20 randomly ratio for test and train split                
def testTrainSplit():    
    for folder in os.listdir(inp_dir):
        filePath = os.path.join(inp_dir, folder)
        if os.path.isdir(filePath):
            noOfFile = len(os.listdir(filePath))
            testNo = noOfFile*0.2
            testFiles = random.sample(range(0, noOfFile), int(testNo))
            count = 0
            for file in os.listdir(filePath):
                fileOrgPath = os.path.join(filePath, file)
                if count in testFiles:
                    filesavePath = os.path.join(test_dir, folder)
                    fileabsPath = os.path.join(filesavePath, file)
                    copyfile(fileOrgPath, fileabsPath)
                else:
                    filesavePath = os.path.join(train_dir, folder)
                    fileabsPath = os.path.join(filesavePath, file)
                    copyfile(fileOrgPath, fileabsPath)
                count = count +1


def directories():

    for folder in os.listdir(inp_dir):

        print(folder)
                
if __name__ == '__main__':
    createDirectory()
    testTrainSplit()
