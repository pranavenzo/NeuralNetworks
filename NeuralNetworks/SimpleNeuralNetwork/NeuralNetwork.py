import numpy as np


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
print nn.predict([0, 0, 1])
print nn.predict([0, 1, 1])
print nn.predict([1, 0, 1])
print nn.predict([1, 1, 1])
################################################################
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])
nn = NeuralNetwork(X, y, [3, 4, 1], 10000)
nn.initialize()
nn.learn()
print nn.predict([0, 0, 1])
print nn.predict([0, 1, 1])
print nn.predict([1, 0, 1])
print nn.predict([1, 1, 1])
