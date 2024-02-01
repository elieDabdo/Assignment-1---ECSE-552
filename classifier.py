import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv

class Data:

    def generateData(self, numSamples, fileName):

        data = [["x1", "x2", "class"]]

        for i in range(numSamples):
            #x1, class 0
            r1_0 = random.gauss(0.0, 1.0)
            t1_0= random.uniform(0, 2*np.pi)
            x1_class0 = r1_0*np.cos(t1_0)

            #x2, class 0
            r2_0 = random.gauss(0.0, 1.0)
            t2_0= random.uniform(0, 2*np.pi)
            x2_class0 = r2_0*np.sin(t2_0)

            data.append([x1_class0, x2_class0, 0.0])

            #x1, class 1
            r1_1 = random.gauss(0.0, 1.0)
            t1_1= random.uniform(0, 2*np.pi)
            x1_class1 = (r1_1+5)*np.cos(t1_1)

            #x2, class 1
            r2_1 = random.gauss(0.0, 1.0)
            t2_1= random.uniform(0, 2*np.pi)
            x2_class1 = (r2_1+5)*np.cos(t2_1)

            data.append([x1_class1, x2_class1, 1.0])

        
        with open(fileName, mode="w", newline ="") as file:
            w = csv.writer(file)
            w.writerows(data)

        print("The data was saved to the file" + fileName)


    def loadData(self, fileName):
        data = np.loadtxt(fileName, delimiter=',', skiprows=1)
        return data

class Classifier(nn.Module):

    def __init__(self, inSize=2, numNodes=30):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(inSize, numNodes)
        self.relu = nn.ReLU()
        self.out = nn.Linear(numNodes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x 
    

class Learner:

    def __init__(self, learningRate=0.01, batchSize=32, numEpochs = 150, splitTrain = 70, dataFile ="gen_data.csv", dataSize = 300):

        self.learningRate = learningRate
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.splitTrain = splitTrain
        self.dataSize = dataSize
        self.dataFile = dataFile
        # initialize the classifier
        self.classifier = Classifier()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.learningRate)

        self.trainingAccuracies = []
        self.validationAccuracies = []

    def retrieveSplitData(self):

        d = Data()
        data = d.loadData(self.dataFile)

        random.shuffle(data)
        numTrain = int(self.splitTrain/100*self.dataSize)

        dataTrain = data[:numTrain]
        dataValidation = data[numTrain:]

        return [dataTrain, dataValidation]
    

    def generateBatches(self, data):
        num = int(len(data)/self.batchSize)
        batches = []
        for j in range (num+1):
            index = j*self.batchSize
            batches.append(data[index: index+self.batchSize])

        return batches

    def evalTrainValidationSets(self, dataTrainX, dataTrainY, dataValidationX, dataValidationY):
        self.classifier.eval()

        trueTrain = 0
        trueValidation = 0

        with torch.no_grad():
            trainOut = self.classifier(dataTrainX)

            for t in range (len(dataTrainX)):
                if trainOut[t] == dataTrainY[t]:
                    trueTrain+=1

            validationOut = self.classifier(dataValidationX)

            for t in range (len(dataValidationX)):
                if validationOut[t] == dataValidationY[t]:
                    trueValidation+=1

        self.trainingAccuracies.append(trueTrain/len(dataTrainX))
        self.validationAccuracies.append(trueValidation/len(dataValidationX))


    def trainingLoop(self):

        dataTrain, dataValidation = self.retrieveSplitData()
        batches = self.generateBatches(dataTrain)

        for i in range (self.numEpochs):
            self.classifier.train()
            for z in range(len(batches)):
                xData = batches[z][:, :2]
                yData = batches[z][:,2]

                self.optimizer.zero_grad()
                out = self.classifier(xData)
                loss = self.criterion(out, yData)
                loss.backward()
                self.optimizer.step()

            self.evalTrainValidationSets(dataTrain[:, :2], dataTrain[:, 2], dataValidation[:, :2], dataValidation[:, 2])

    
        # exit()

        # self.classifier.train()
        # for i in range(self.numEpochs):


if __name__ == '__main__':

    l = Learner()
    l.trainingLoop()
