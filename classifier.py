import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


class Data:

    def generateData(self, numSamplesEachClass, fileName):

        data = [["x1", "x2", "class"]]

        for i in range(numSamplesEachClass):
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

    def __init__(self, train = False, learningRate=0.001, batchSize=64, numEpochs = 100, splitTrain = 70, dataFile ="gen_data.csv", dataSize = 1000, save=10, saveLocation= "classifier.pth"):
        
        # Parameters
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.splitTrain = splitTrain
        self.dataSize = dataSize
        self.dataFile = dataFile
        self.save = save
        self.saveLocation = saveLocation
        self.train = train  # Setting this parameter to true trains a new model, when it is false, it retrieves the existing model
        self.dataTrain = None
        self.dataValidation = None

        # Initialize the classifier
        self.classifier = Classifier()
        if not self.train:
            state_dict = torch.load(self.saveLocation)
            self.classifier.load_state_dict(state_dict)
        else:
            self.criterion = nn.BCELoss()
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.learningRate)

        # Saving the Accuracies
        self.trainingAccuracies = []
        self.validationAccuracies = []

    def retrieveSplitData(self):
        d = Data()
        data = d.loadData(self.dataFile)

        # Shuffle the data and retrive the train and validate splits
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

            trainOutNormalized = np.where(trainOut >=0.5, 1, 0)

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

    def plotDataAndDecisionBoundary(self, smooth= False):
        
        if self.dataValidation is None:
            self.dataTrain, self.dataValidation = self.retrieveSplitData()
        
        dataX1 = self.dataValidation[:, 0]
        dataX2 = self.dataValidation[:, 1]
        dataY = self.dataValidation[:, 2]

        self.classifier.eval()

        # divide the data into classes
        dataX1label0, dataX2label0, dataX1label1, dataX2label1 = [], [], [], []

        for d in range (len(dataX1)):
            if dataY[d] == 0:
                dataX1label0.append(dataX1[d])
                dataX2label0.append(dataX2[d])
            else:
                dataX1label1.append(dataX1[d])
                dataX2label1.append(dataX2[d])

        x1Min, x1Max = dataX1.min(), dataX1.max()
        x2Min, x2Max = dataX2.min(), dataX2.max()
        
        # Smooth decision boundary 
        if smooth:
            xx, yy = np.meshgrid(np.arange(x1Min, x1Max, 0.01), np.arange(x2Min, x2Max, 0.01))
            grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
            with torch.no_grad():
                output = self.classifier(grid).numpy().reshape(xx.shape)
            plt.contour(xx, yy, output, levels=[0.5], colors="black")

        # Draw lines for each neuron decision boundary
        else:
            x1 = np.arange(x1Min, x1Max, 0.01)
            weights = self.classifier.layer1.weight.detach().numpy()
            biases = self.classifier.layer1.bias.detach().numpy()

            for i in range(weights.shape[0]):
                x2 = -(weights[i,0] * x1 + biases[i])/weights[i, 1]
                plt.plot(x1, x2)

        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.scatter(dataX1label0, dataX2label0, label="Class 0", color="blue", marker="o")
        plt.scatter(dataX1label1, dataX2label1, label= "Class 1", color="orange", marker="^")
        plt.legend()
        plt.title("Visualization of the Validation Data")

        plt.show()

    def trainingLoop(self):

        self.dataTrain, self.dataValidation = self.retrieveSplitData()
        batches = self.generateBatches(self.dataTrain)

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

            self.evalTrainValidationSets(self.dataTrain[:, :2], self.dataTrain[:, 2], self.dataValidation[:, :2], self.dataValidation[:, 2])

            if i % self.save == 0 or i==(self.numEpochs-1):
                torch.save(self.classifier.state_dict(), self.saveLocation)
                print("The model has been saved at epoch number "+ str(i))

        self.plotAccuracies()

if __name__ == '__main__':
    
    # To generate new data, uncomment the 2 following lines
    # dataGen = Data ()
    # dataGen.generateData(numSamplesEachClass=500, fileName= "gen_data.csv")

    # To train a new model, change the train param below
    # To use the existing model, use the file save location
    learner = Learner(train=True, saveLocation="classifier.pth")
    
    # Uncomment this line to train 
    learner.trainingLoop()

    # To plot the individual decision boundaries of each neuron, turn smooth to false
    learner.plotDataAndDecisionBoundary(smooth=True)
    learner.plotDataAndDecisionBoundary(smooth=False)
