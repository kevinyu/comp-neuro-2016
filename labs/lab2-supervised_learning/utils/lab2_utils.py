import matplotlib.pyplot as plt
import numpy as np

class HyperPlanePlotter(object):
    def __init__(self, data, apples, oranges, numTrials, halfNumSamples=0):
        self.data = data
        self.apples = apples
        self.oranges = oranges
        self.numTrials = numTrials
        self.halfNumSamples = halfNumSamples

    def sigmoid(self, u):
        return 1.0 / (1.0 + np.exp(-u))

    def plotSigmoid(self, sigmoid):
        plt.figure(figsize=(3,3))
        width = 10
        x = np.arange(-width, width)
        plt.plot(x, sigmoid(x))

    def computeLine(firstWeight, xPoints, bias, secondWeight):
        return -(firstWeight*xPoints + bias) / secondWeight

    def initPlot(self, name, axes, errorHeight=25):
        axes[0].set_title("Apples and Oranges (%s)" % name)
        axes[0].plot(self.apples[0,:], self.apples[1,:], 'b+', self.oranges[0,:], self.oranges[1,:], 'rx')
        axes[0].axis([0, 4, 0, 4])

        axes[1].set_title("Error over time (%s)" % name)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Error")
        axes[1].set_xlim([0, self.numTrials])
        axes[1].set_ylim([0, errorHeight])

    def initPlotProb2(self):
        figure, axes = plt.subplots(2,2)
        self.linearAxes  = axes[0,:]
        self.sigmoidAxes = axes[1,:]

        self.initPlot("Linear", self.linearAxes)
        self.initPlot("Sigmoid", self.sigmoidAxes, 7)
        self.figure = figure
        plt.tight_layout()

    def setupPlotProb2Help(self, weights, bias, axes):
        hyperPlane = np.zeros((2,5))
        hyperPlane[0,:] = np.arange(5)
        hyperPlane[1,:] = -(weights[0]*hyperPlane[0,:]+bias)/weights[1]
        hyperPlanePlot = axes[0].plot(hyperPlane[0,:], hyperPlane[1,:], 'g--')[0]

        self.hyperPlane = hyperPlane
        self.hyperPlanePlot = hyperPlanePlot

    def setupPlotProb2(self, name, weights, bias):
        if name == "Linear":
            return self.setupPlotProb2Help(weights, bias, self.linearAxes)
        return self.setupPlotProb2Help(weights, bias, self.sigmoidAxes)

    def updatePlotProb2(self, weights, bias):
        hyperPlane = self.hyperPlane
        hyperPlanePlot = self.hyperPlanePlot

        hyperPlane[1,:] = -(weights[0]*hyperPlane[0,:]+bias)/weights[1];
        hyperPlanePlot.set_ydata(hyperPlane[1,:])
        self.figure.canvas.draw()

    def plotErrorProb2(self, name, t, errorT):
        if name == "Linear":
            self.linearAxes[1].scatter(t, errorT, s=1)
        else:
            self.sigmoidAxes[1].scatter(t, errorT, s=1)

    def setupPlotProb3(self, weightsOne, biasOne, weightsTwo, biasTwo):
        # intialize plot layout
        figure, axesGrid = plt.subplots(2,2)
        axes = [axesGrid[0,0], axesGrid[1,0], axesGrid[0,1]]
        self.initPlot('Two-Layer', axes, 7)
        axes[0].set_ybound(-3,4); axes[0].set_xbound(0,5)
        plt.tight_layout()

        halfNumSamples = self.halfNumSamples

        # initialize data plot
        hyperPlaneOne = np.zeros((2,2))
        hyperPlaneOne[0,:] = [0,5]
        hyperPlaneOne[1,:] = -(weightsOne[0,0]*hyperPlaneOne[0,:]+biasOne[0])/weightsOne[1,0]
        hyperPlaneOnePlot = axes[0].plot(hyperPlaneOne[0,:], hyperPlaneOne[1,:], 'g--', 2)[0]

        hyperPlaneTwo = np.zeros((2,2))
        hyperPlaneTwo[0,:] = [0,5]
        hyperPlaneTwo[1,:] = -(weightsOne[0,1]*hyperPlaneTwo[0,:]+biasOne[1])/weightsOne[1,1]
        hyperPlaneTwoPlot = axes[0].plot(hyperPlaneTwo[0,:], hyperPlaneTwo[1,:], 'm-.', 2)[0]

        # initialize hidden units plot
        yData = self.sigmoid(np.dot(weightsOne.T, self.data) + biasOne)
        yDataOnePlot = axes[2].plot(yData[0,:halfNumSamples], yData[1,:halfNumSamples], 'b+')[0]
        yDataTwoPlot = axes[2].plot(yData[0,halfNumSamples:], yData[1,halfNumSamples:], 'rx')[0]

        yHyperPlane = np.zeros((2,2))
        yHyperPlane[0,:] = [-2,2]
        yHyperPlane[1,:] = -(weightsTwo[0]*yHyperPlane[0,:]+biasTwo)/weightsTwo[1]
        yHyperPlanePlot = axes[2].plot(yHyperPlane[0,:], yHyperPlane[1,:], 'g--')[0]

        axes[2].set_title("Hidden representation")
        axes[2].set_ybound(-1,2); axes[2].set_xbound(-1,2)
        axes[2].set_xlabel("First hidden unit (y[0])"); axes[2].set_ylabel("Second hidden unit (y[1])")

        # Keep state of parameters
        self.axes, self.figure = axes, figure

        self.hyperPlaneOne = hyperPlaneOne
        self.hyperPlaneTwo = hyperPlaneTwo
        self.hyperPlaneOnePlot = hyperPlaneOnePlot
        self.hyperPlaneTwoPlot = hyperPlaneTwoPlot
        self.yDataOnePlot = yDataOnePlot
        self.yDataTwoPlot = yDataTwoPlot
        self.yHyperPlane = yHyperPlane
        self.yHyperPlanePlot = yHyperPlanePlot

    def updatePlotProb3(self, weightsOne, biasOne, weightsTwo, biasTwo):
        halfNumSamples = self.halfNumSamples
        hyperPlaneOne = self.hyperPlaneOne
        hyperPlaneTwo = self.hyperPlaneTwo
        hyperPlaneOnePlot = self.hyperPlaneOnePlot
        hyperPlaneTwoPlot = self.hyperPlaneTwoPlot
        yDataOnePlot = self.yDataOnePlot
        yDataTwoPlot = self.yDataTwoPlot
        yHyperPlanePlot = self.yHyperPlanePlot
        yHyperPlane = self.yHyperPlane

        hyperPlaneOne[1,:] = -(weightsOne[0,0]*hyperPlaneOne[0,:]+biasOne[0])/weightsOne[1,0]
        hyperPlaneOnePlot.set_ydata(hyperPlaneOne[1,:])

        hyperPlaneTwo[1,:] = -(weightsOne[0,1]*hyperPlaneTwo[0,:]+biasOne[1])/weightsOne[1,1]
        hyperPlaneTwoPlot.set_ydata(hyperPlaneTwo[1,:])

        yData = self.sigmoid(np.dot(weightsOne.T, self.data) + biasOne)
        yDataOnePlot.set_data(yData[0,:halfNumSamples], yData[1,:halfNumSamples])
        yDataTwoPlot.set_data(yData[0,halfNumSamples:], yData[1,halfNumSamples:])

        yHyperPlane[1,:] = -(weightsTwo[0]*yHyperPlane[0,:]+biasTwo)/weightsTwo[1]
        yHyperPlanePlot.set_ydata(yHyperPlane[1,:])

        self.figure.canvas.draw()

    def plotErrorProb3(self, t, errorT):
        self.axes[1].scatter(t, errorT, s=1)

class FilterPlotter(object):
    def __init__(self, numTrials):
        self.numTrials = numTrials

    def setupPlots(self, weightsOne, numHiddenUnits):
        self.errorFigure = plt.figure(figsize=(3,3))
        self.axes = plt.subplot(111)
        plt.title("Error over time")

        self.axes.set_title("Error over time")
        self.axes.set_xlabel("Iteration")
        self.axes.set_ylabel("Error")
        self.axes.set_xlim([0, self.numTrials])
        self.axes.set_ylim([0, 5])
        plt.tight_layout()

        figure = plt.figure(figsize=(12,3))
        filterPlots = [None] * numHiddenUnits
        for i in np.arange(numHiddenUnits):
            axes = plt.subplot(1,numHiddenUnits,i+1)
            filterPlots[i] = plt.imshow(weightsOne[:,i].reshape(3,3),cmap='Greys',interpolation='none')
            axes.xaxis.set_visible(False)
            axes.yaxis.set_visible(False)
        plt.tight_layout()

        self.figure = figure
        self.filterPlots = filterPlots

    def updatePlots(self, weightsOne):
        for i in range(weightsOne.shape[1]):
            self.filterPlots[i].set_data(weightsOne[:,i].reshape(3,3))
        self.figure.canvas.draw()

    def plotError(self, t, errorT):
        self.axes.scatter(t, errorT, s=1)
        self.errorFigure.canvas.draw()
