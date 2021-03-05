import numpy as np
import math, operator, random

def prepareData(filename):
    random.seed()

    file = open(filename)
    lines = file.readlines()
    numLines = len(lines)
    numTests = math.floor(numLines * 0.2)

    dataSet = np.zeros((numLines,4))
    testSet = np.zeros((numTests,4))
    labels,testLabels = [],[]

    for i in range(0, numTests):
        randomIndex = random.randint(0, numLines-i-1)
        line = lines.pop(randomIndex).strip()
        lineList = line.split(',')
        testSet[i,:] = lineList[0:4]
        testLabels.append(lineList[-1])

    index = 0
    for line in lines:
        line = line.strip()
        lineList = line.split(',')
        dataSet[index,:] = lineList[0:4]
        labels.append(lineList[-1])
        index += 1



    return dataSet, labels, testSet, testLabels

def classify(example, dataSet, labels, k):
    dataSetSize = len(dataSet)
    differenceMatrix = np.tile(example, (dataSetSize,1)) - dataSet
    sqDifferenceMatrix = differenceMatrix ** 2
    sqDistances = sqDifferenceMatrix.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistanceIndices = distances.argsort()

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistanceIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

dataSet,labels,testSet,testLabels = prepareData('iris.data')

k = 16
errorCount = 0
testSetSize = len(testSet)

index = 0
for example in testSet:
    print(f'Classifying {index+1}/{testSetSize}...')
    result = classify(example, dataSet, labels, k)
    print(f'Result: {result}')
    print(f'Expected: {testLabels[index]}')
    if result != testLabels[index]:
        errorCount += 1
        print('**MISTAKE**')
    print()
    index += 1

errorPercentage = (errorCount / testSetSize) * 100
print(f'Final outcome: {testSetSize - errorCount}/{testSetSize} correct.')
print('Error rate: %.2f%%' % errorPercentage)
print()