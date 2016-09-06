import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class PlotterOne(object):
    def __init__(self, height, totalLength):
        self.height = height
        self.totalLength = totalLength

    def plotCurrent(self, I):
        plt.figure(figsize=(6,3))
        plt.title("I(t)")
        plt.plot(I,linewidth=2,color='black')
        plt.ylim([0,self.height*2])
        plt.xlabel('Time (ms)')
        plt.ylabel('picoAmps')

    def plotMembraneVoltage(self, VMembrane, GMin):
        plt.plot(VMembrane[:self.totalLength], label='G_leak=%dnS' % GMin)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Voltage (mV)')
        plt.legend()

class PlotterTwo(object):
    def __init__(self, GMax, VLow, VMax):
        self.GMax = GMax
        self.VLow = VLow
        self.VMax = VMax * 1.4

    def plotEqMembraneVoltage(self, VMemNa, VMemCl):
        plt.figure(figsize=(10,5))
        plt.title("Equilibrium Membrane Voltage")

        plt.plot(VMemNa, color='r', label="Na open")
        plt.plot(VMemCl, color='b', label="Cl open")

        plt.xlabel("G (nS)")
        plt.ylabel("Membrane Voltage (mV)")
        plt.axis([0, self.GMax, self.VLow, self.VMax])
        plt.legend(loc=2, fontsize=12)

    def plotEqMembraneVoltagePrediction(self, VMemNa, VMemNaCl, VMemLinearNaCl):
        plt.figure(figsize=(10,5))
        plt.title("Equilibrium Membrane Voltage")

        plt.plot(VMemNa, color='r', label="Na open")
        plt.plot(VMemNaCl, color='c', label="Na open, Fixed G_Cl")
        plt.plot(VMemLinearNaCl, color='c', linestyle="--", label="Linear Prediction")

        plt.xlabel("G (nS)")
        plt.ylabel("Membrane Voltage (mV)")
        plt.axis([0, self.GMax, self.VLow, self.VMax])
        plt.legend(loc=2, fontsize=12)
