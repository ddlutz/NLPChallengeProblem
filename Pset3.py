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


def predictWithTest(svmClassifier, svmData, row, zeroLabel, oneLabel, text):
    print "Predicting " + str(text[5])

    svmData.append(row[0])
    vec = DictVectorizer()
    svmDataVectorized = vec.fit_transform(svmData).toarray()

    svmAns = svmClassifier.predict(svmDataVectorized[-1])

    if svmAns == 0:
        print "Classified as " + zeroLabel
    else:
        print "Classified as " + oneLabel

    return svmAns


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
    testDataText = [r for r in testData if len(r) > 1]

    random.shuffle(data)

    labeledEMData = [(getEMFeatures(vocab, row[5]), row[4]) for row in data]
    testEMData = [(getEMFeatures(vocab, row[5]), row[4]) for row in testDataText]

    labeledQAData = [(getEMFeatures(vocab, row[5]), row[3]) for row in data]
    testQAData = [(getEMFeatures(vocab, row[5]), row[3]) for row in testDataText]

    allEMData = labeledEMData + testEMData
    allQAData = labeledQAData + testQAData

    svmEMData = [row[0] for row in allEMData]
    svmEMAnswers = [row[1] for row in allEMData]

    svmQAData = [row[0] for row in allQAData]
    svmQAAnswers = [row[1] for row in allQAData]

    svmEMClass = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 600)
    svmQAClass = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 600)



    vec = DictVectorizer()

    svmDataVectored = vec.fit_transform(svmEMData).toarray()

    svmEMArrayAnswers = []
    for i in svmEMAnswers:
        if i == 'A' or i == 'E':
            svmEMArrayAnswers.append(0)
        else:
            svmEMArrayAnswers.append(1)

    svmQAArrayAnswers = []
    for i in svmQAAnswers:
        if i == 'A' or i == 'E':
            svmQAArrayAnswers.append(0)
        else:
            svmQAArrayAnswers.append(1)

    """Train only on test data :)"""

    svmEMClass.fit(svmDataVectored[0:len(labeledEMData)], svmEMArrayAnswers[0:len(labeledEMData)])

    svmQAClass.fit(svmDataVectored[0:len(labeledQAData)], svmQAArrayAnswers[0:len(labeledQAData)])

    testEMCorrectAnswers = []
    for i in testEMData:
        if i[1] == 'A' or i[1] == 'E':
            testEMCorrectAnswers.append(0)
        else:
            testEMCorrectAnswers.append(1)

    testQACorrectAnswers = []
    for i in testQAData:
        if i[1] == 'A' or i[1] == 'E':
            testQACorrectAnswers.append(0)
        else:
            testQACorrectAnswers.append(1)        

    EMpredictions = []
    for row in range(len(testEMData)):
        EMpredictions.append(predictWithTest(svmEMClass, svmEMData, testEMData[row], 'E', 'M', testDataText[row] ))

    EMpredictedCorrectly = 0
    for i in range(len(EMpredictions)):
        if EMpredictions[i] == testEMCorrectAnswers[i]:
            EMpredictedCorrectly += 1

    QApredictions = []
    for row in range(len(testQAData)):
        QApredictions.append(predictWithTest(svmQAClass, svmQAData, testQAData[row], 'A', 'Q', testDataText[row] ))

    QApredictedCorrectly = 0
    for i in range(len(QApredictions)):
        if QApredictions[i] == testQACorrectAnswers[i]:
            QApredictedCorrectly += 1

    print "E/M Average w/ SVM: " + str(EMpredictedCorrectly / float(len(testData)))

    print "Q/A Average w/ SVM: " + str(QApredictedCorrectly / float(len(testData)))

    
if __name__ == "__main__":
    main()