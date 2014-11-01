# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 22:40:23 2014

@author: Douglas
"""

import nltk.data
import sys
import random

features = []

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

def getEMFeatures(vocab, textRow):
    features = {}
    textRow = textRow.split(' ')
    features['FirstWord'] = textRow[0]
    numSl = 0
    numSp = 0
    for word in textRow:
        if word == '{sl}':
            numSl += 1
        if word == 'sp':
            numSp += 1

    for word in vocab:
        if word in textRow:
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
def main():
    
    """
    if len(sys.argv) < 3:
        print "train file as first argument, test file as second"
        sys.exit()
    fileName = sys.argv[1]
    """

    data = loadFiles()
    vocab = buildVocab(data)

    data = [r for r in data if len(r) > 1]

    labeledData = [(getEMFeatures(vocab, row[5]), row[3]) for row in data]

    trainset, testset = splitData(labeledData)


    
    classifier = nltk.NaiveBayesClassifier.train(trainset)

    print "Train accuracy: " + str(nltk.classify.accuracy(classifier, trainset))
    print "Test accuracy: " + str(nltk.classify.accuracy(classifier, testset))

    
if __name__ == "__main__":
    main()