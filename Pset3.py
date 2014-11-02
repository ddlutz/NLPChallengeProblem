# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 22:40:23 2014

@author: Douglas
"""

import nltk.data
import sys
import random
from sklearn import svm
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer
import numpy as np

def buildVocab(data):
    vocab = []

    for line in data:
        if len(line) > 1:
            for word in line[5].split(' '):
                if word not in ['{sl}','sp']:
                    vocab.append(word)

    return set(vocab)


def loadFiles():
    firstFile = ""
    with open("data.csv", 'r') as cFile:
        firstFile = cFile.read()

    firstFile = firstFile.split('\n')

    data = []

    for line in firstFile:
        data.append(line.split(','))

    return data

def fixWord(word):
    newWord = ""
    for letter in word:
        if str.isalpha(letter) or str.isdigit(letter):
            newWord += letter

    return newWord


def getEMFeatures(vocab, textRow):
    features = {}
    textRow = textRow.split(' ')
    features['FirstWord'] = textRow[0]
    numSl = 0
    numSp = 0

    fixedTextRow = [fixWord(word) for word in textRow]
    for word in textRow:
        if word == '{sl}':
            numSl += 1
        if word == 'sp':
            numSp += 1

    for word in vocab:
        fixedWord = fixWord(word)
        if word in fixedTextRow:
            features['Vocab-' + word ] = True
        else:
            features['Vocab-' + word ] = False

    features['NumSp'] = numSp

    features['NumSl'] = numSl

    features['NumWords'] = len([w for w in textRow if w not in ['sp','{sl}']])

    return features
        
def splitData(data):

    numTrain = int(len(data) * 0.8)

    random.shuffle(data)

    return (data[:numTrain], data[numTrain:])

def getSVMFeatures(data):
    data = []

def main():
    
    """
    if len(sys.argv) < 3:
        print "train file as first argument, test file as second"
        sys.exit()
    fileName = sys.argv[1]
    """

    data = loadFiles()
    vocab = buildVocab(data)

    vocab = [fixWord(word) for word in vocab]

    data = [r for r in data if len(r) > 1]

    labeledData = [(getEMFeatures(vocab, row[5]), row[3]) for row in data]

    trainset, testset = splitData(labeledData)
    
    classifier = nltk.NaiveBayesClassifier.train(trainset)

    svmData = [getEMFeatures(vocab, row[5]) for row in data]
    svmAnswers = [row[3] for row in data]

    svmClass = svm.SVC(kernel = 'rbf', gamma=0.001, C=100)

    vec = DictVectorizer()
    svmData = vec.fit_transform(svmData).toarray()

    svmArrayAnswers = []
    for i in svmAnswers:
        if i == 'A' or i == 'E':
            svmArrayAnswers.append(0)
        else:
            svmArrayAnswers.append(1)

    svmTrainData = svmData[:330]
    svmTrainAnswers = svmArrayAnswers[:330]
    svmTestData = svmData[330:]
    svmTestAnswers = svmArrayAnswers[330:]

    svmClass.fit(svmTrainData, svmTrainAnswers)

    trainPredict = svmClass.predict(svmTrainData)
    svmTrainAccuracy = np.mean(trainPredict == svmTrainAnswers)

    predicted = svmClass.predict(svmTestData)
    svmAccuracy = np.mean(predicted == svmTestAnswers)

    print "SVM Train accuracy: " + str(svmTrainAccuracy)
    print "SVM Test accuracy: " + str(svmAccuracy)

    print "NB Train accuracy: " + str(nltk.classify.accuracy(classifier, trainset))
    print "NB Test accuracy: " + str(nltk.classify.accuracy(classifier, testset))

    
if __name__ == "__main__":
    main()