# -*- coding: utf-8 -*-
"""
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


def loadFile(fileName):
    firstFile = ""
    with open(fileName, 'r') as cFile:
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

def EnsemblePredicition(DTclassifier, NBclassifier, svmClassifier, textRow, svmData):

    svmData.append(textRow[0])
    vec = DictVectorizer()
    svmDataVectorized = vec.fit_transform(svmData).toarray()

    dtAns = DTclassifier.classify(textRow[0])
    nbAns = NBclassifier.classify(textRow[0])
    svmAns = svmClassifier.predict(svmDataVectorized[-1])

    ans0 = 0
    ans1 = 0

    if nbAns == 'A' or nbAns == 'E':
        ans0 += 1
    else:
        ans1 += 1

    if dtAns == 'A' or dtAns == 'E':
        ans0 += 1
    else:
        ans1 += 1

    if svmAns == 0:
        ans0 += 1
    else:
        ans1 += 1

    if ans0 > ans1:
        return 0
    else:
        return 1


def classifyUsingEnsemble(dtClass, nbClass, svmClass, testSet, svmData):
    print "classifying " + str(len(testSet)) + " points"
    return [EnsemblePredicition(dtClass, nbClass, svmClass, row, svmData) for row in testSet]

def main():
    
    if len(sys.argv) < 3:
        print "train file as first argument, test file as second"
        sys.exit()
    trainFileName = sys.argv[1]
    testFileName = sys.argv[2]

    data = loadFile(trainFileName)
    testData = loadFile(testFileName)

    vocab = buildVocab(data)

    vocab = [fixWord(word) for word in vocab]

    data = [r for r in data if len(r) > 1]
    testData = [r for r in testData if len(r) > 1]

    random.shuffle(data)

    labeledEMData = [(getEMFeatures(vocab, row[5]), row[4]) for row in data]
    testData = [(getEMFeatures(vocab, row[5]), row[4]) for row in testData]
    
    NBclassifier = nltk.NaiveBayesClassifier.train(labeledEMData)
    DTclassifier = nltk.DecisionTreeClassifier.train(labeledEMData)

    allData = labeledEMData + testData

    svmData = [row[0] for row in allData]
    svmAnswers = [row[1] for row in allData]

    svmClass = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 600)

    vec = DictVectorizer()

    svmDataVectored = vec.fit_transform(svmData).toarray()

    svmArrayAnswers = []
    for i in svmAnswers:
        if i == 'A' or i == 'E':
            svmArrayAnswers.append(0)
        else:
            svmArrayAnswers.append(1)

    svmClass.fit(svmDataVectored[0:len(labeledEMData)], svmArrayAnswers[0:len(labeledEMData)])

    trainCorrectAnswers = []
    for i in labeledEMData:
        if i[1] == 'A' or i[1] == 'E':
            trainCorrectAnswers.append(0)
        else:
            trainCorrectAnswers.append(1)

    testCorrectAnswers = []
    for i in testData:
        if i[1] == 'A' or i[1] == 'E':
            testCorrectAnswers.append(0)
        else:
            testCorrectAnswers.append(1)
     
    """ensembleTrainPredicted = classifyUsingEnsemble(DTclassifier, NBclassifier, svmClass, labeledEMData, svmData)
    

    ensembleTrainCorrect = 0
    

    for i in range(len(ensembleTrainPredicted)):
        if ensembleTrainPredicted[i] == trainCorrectAnswers[i]:
            ensembleTrainCorrect += 1

    
    print "Train accuracy: " + str(ensembleTrainCorrect / float(len(ensembleTrainPredicted)))"""

    ensembleTestPredicted = classifyUsingEnsemble(DTclassifier, NBclassifier, svmClass, testData, svmData)

    ensembleTestCorrect = 0

    for i in range(len(ensembleTestPredicted)):
        if ensembleTestPredicted[i] == testCorrectAnswers[i]:
            ensembleTestCorrect += 1

    print "Test accurcy: " + str(ensembleTestCorrect / float(len(ensembleTestPredicted)))


    
if __name__ == "__main__":
    main()