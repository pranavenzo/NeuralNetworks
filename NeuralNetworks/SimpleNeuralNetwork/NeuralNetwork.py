import numpy as np
from random import *


class NeuralNetwork:
    def __init__(self, trainingSetInput, trainingSetOutput, configuration, learningTermination):
        self.trainingSetInput = trainingSetInput
        self.trainingSetOutput = trainingSetOutput
        self.configuration = configuration  # stores number of nodes per layer
        self.learningTermination = learningTermination

    def sigmoid(self, x, deriv=False):
        if (deriv == True):
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    def initialize(self):
        self.synX = []
        for i in range(0, len(self.configuration) - 1):
            self.synX.append(2 * np.random.random((self.configuration[i], self.configuration[i + 1])) - 1)

    def learn(self):
        layers = []
        for i in xrange(self.learningTermination):
            # forward propagation
            layers = []
            x = self.trainingSetInput
            layers.append(x)
            j = 0
            while j < len(self.synX):
                nextLayer = self.sigmoid(np.dot(x, self.synX[j]))
                j += 1
                x = nextLayer
                layers.append(x)

            # Back propagation
            errorMat = self.trainingSetOutput - x
            j = len(layers) - 1
            while j > 0:
                delta = errorMat * self.sigmoid(layers[j], deriv=True)
                errorMat = delta.dot(self.synX[j - 1].T)
                self.synX[j - 1] += layers[j - 1].T.dot(delta)
                j -= 1

    def predict(self, inputVector):
        x = inputVector
        j = 0
        while j < len(self.synX):
            nextLayer = self.sigmoid(np.dot(x, self.synX[j]))
            j += 1
            x = nextLayer
        return x


#######################Testing###################################
# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T
nn = NeuralNetwork(X, y, [3, 1], 10000)
nn.initialize()
nn.learn()
# print nn.predict([0, 0, 1])
# print nn.predict([0, 1, 1])
# print nn.predict([1, 0, 1])
# print nn.predict([1, 1, 1])
################################################################
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[1],
              [0],
              [0],
              [1]])
nn = NeuralNetwork(X, y, [3, 4, 1], 10000)
nn.initialize()
nn.learn()


# print nn.predict([0, 0, 1])
# print nn.predict([0, 1, 1])
# print nn.predict([1, 0, 1])
# print nn.predict([1, 1, 1])
def fileReader(filename):
    f = open(filename)
    fullText = f.read()
    lines = fullText.split("\n")
    rows = len(lines)
    total = []
    cols = 0
    for term in lines[0].split(","):
        total.append(0.0)
        cols += 1
    for line in lines:
        numbers = line.split(",")
        for i in range(0, len(numbers)):
            total[i] += float(numbers[i])

    strdataset = []
    for line in lines:
        strdataset.append(line.split(","))
    for i in xrange(rows):
        for j in range(0, cols - 1):
            if (total[j] != 0):
                strdataset[i][j] = str(float(strdataset[i][j]) / total[j])
    dataset = []
    for row in strdataset:
        dataset.append(map(float, row))

    output = []
    for row in dataset:
        output.append([row[len(row) - 1]])
    for i in range(len(dataset)):
        dataset[i].pop()
    return (dataset, output)


def accuracy():
    trainFile = "t_pima-indians-diabetes.csv"
    testFile = "pima-indians-diabetes.csv"

    nn = makeNN(*fileReader(trainFile))
    correct = 0.0
    total = 0.0
    dataset, output = fileReader(testFile)
    for i in xrange(len(dataset)):
        pred = nn.predict(dataset[i])
        classpred = [round(pred)]
        actu = output[i]
        if (classpred == actu):
            correct += 1
        total += 1
    accuracyPercentage = str(100 * correct / total)
    print 'Accuracy : ' + accuracyPercentage + '%'
    return accuracyPercentage


def makeNN(dataset, output):
    nn = NeuralNetwork(np.array(dataset), np.array(output), [8, 1], 100000)
    nn.initialize()
    nn.learn()
    return nn


accuracy()
