import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


class Data:

    def generateData(self, numSamples, fileName):

        data = [["x1", "x2", "class"]]

        for i in range(numSamples):
            # Generate the random numbers
            r0 = random.gauss(0.0, 1.0)
            t0= random.uniform(0, 2*np.pi)
            r1 = random.gauss(0.0, 1.0)
            t1= random.uniform(0, 2*np.pi)

            # X1, class 0
            x1_class0 = r0*np.cos(t0)
            # X2, class 0
            x2_class0 = r0*np.sin(t0)

            # X1, class 1
            x1_class1 = (r1+5)*np.cos(t1)

            # X2, class 1
            x2_class1 = (r1+5)*np.sin(t1)

            # Add datapoints
            data.append([x1_class0, x2_class0, 0.0])
            data.append([x1_class1, x2_class1, 1.0])

        
        with open(fileName, mode="w", newline ="") as file:
            w = csv.writer(file)
            w.writerows(data)

        print("The data was saved to the file: " + fileName)


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

    def __init__(self, learningRate=0.0001, batchSize=32, numEpochs = 500, splitTrain = 80, dataFile ="gen_data.csv", dataSize = 300, save=10):
        
        # Parameters
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.splitTrain = splitTrain
        self.dataSize = dataSize
        self.dataFile = dataFile
        self.save = save

        # Initialize the classifier
        self.classifier = Classifier()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.learningRate)

        # Saving the Accuracies
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

    def evalTrainValidationSets(self, dataTrainX_np, dataTrainY, dataValidationX_np, dataValidationY):
        self.classifier.eval()

        trueTrain = 0
        trueValidation = 0

        dataTrainX = torch.tensor(dataTrainX_np, dtype=torch.float32)
        dataValidationX = torch.tensor(dataValidationX_np, dtype=torch.float32)

        with torch.no_grad():
            trainOut = (self.classifier(dataTrainX)).numpy()

            trainOutNormalized = np.where(trainOut >=0.5,1,0)

            for t in range (len(dataTrainX)):
                if trainOutNormalized[t] == dataTrainY[t]:
                    trueTrain+=1

            validationOut = self.classifier(dataValidationX).numpy()

            validationOutNormalized = np.where(validationOut >=0.5,1,0)

            for t in range (len(dataValidationX)):
                if validationOutNormalized[t] == dataValidationY[t]:
                    trueValidation+=1

        accuracyTrain = (trueTrain/len(dataTrainX))*100
        accuracyValidation = (trueValidation/len(dataValidationX))*100
        self.trainingAccuracies.append(accuracyTrain)
        self.validationAccuracies.append(accuracyValidation)

        print("Training Accuracy: "+ str(int(accuracyTrain)) + "% || Validation Accuracy: "+ str(int(accuracyValidation))+"%")

    
    def plotAccuracies(self):

        epochs = list(range(0,self.numEpochs))
        plt.plot(epochs, self.trainingAccuracies, label="Training Set Accuracies")
        plt.plot(epochs, self.validationAccuracies, label="Validation Set Accuracies")

        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracies in %")
        plt.title("Training and Validation Sets Accuracies in terms of epochs")

        plt.legend()
        plt.show()

    def plotData(self, dataX1, dataX2, dataY):
        
        # divide the data into classes

        dataX1label0 = []
        dataX2label0 = []
        dataX1label1 = []
        dataX2label1 = []


        for d in range (len(dataX1)):
            if dataY[d] == 0:
                dataX1label0.append(dataX1[d])
                dataX2label0.append(dataX2[d])
            else:
                dataX1label1.append(dataX1[d])
                dataX2label1.append(dataX2[d])

        plt.scatter(dataX1label0, dataX2label0, label="Class 0", color="blue", marker="o")
        plt.scatter(dataX1label1, dataX2label1, label= "Class 1", color="orange", marker="^")
        plt.legend()
        plt.title("Visualization of the Validation Data")

        plt.show()

    def trainingLoop(self):

        dataTrain, dataValidation = self.retrieveSplitData()
        self.plotData(dataValidation[:, 0],dataValidation[:, 1], dataValidation[:, 2])

        batches = self.generateBatches(dataTrain)

        for i in range (self.numEpochs):
            self.classifier.train()
            for z in range(len(batches)):
                xData = torch.tensor(batches[z][:, :2], dtype=torch.float32)
                yData = torch.tensor(batches[z][:,2], dtype=torch.float32).view(-1,1)
                self.optimizer.zero_grad()
                out = self.classifier(xData)
                loss = self.criterion(out, yData)
                loss.backward()
                self.optimizer.step()

            self.evalTrainValidationSets(dataTrain[:, :2], dataTrain[:, 2], dataValidation[:, :2], dataValidation[:, 2])

            if i % self.save ==0 or i==(self.numEpochs-1):
                torch.save(self.classifier.state_dict(), "classifier.pth")
                print("The model has been saved at epoch number "+ str(i))

        self.plotAccuracies()

if __name__ == '__main__':

    l = Learner()
    l.trainingLoop()
