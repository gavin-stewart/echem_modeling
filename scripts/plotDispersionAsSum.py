#!/bin/python
# Plot the current due to dispersion, and represent it as the weighted sum of
# 5 components.
import getTopLevel
from matplotlib import cm
import matplotlib.pyplot as plt
import tools.solutionTools as st
import tools.gridTools as gt
import tools.io as io
import numpy as np
from scipy.ndimage.filters import gaussian_filter


E_0SD = 0.05
fileName = getTopLevel.makePathFromTop('files/simulationParameters.json')
data = io.readParametersFromJSON(fileName, "Martin's experiment")
tEnd = 2 * (data["ERev"] - data["EStart"]) / data["nu"]
numPts = int(np.ceil(200 * data["freq"] * tEnd))
t = np.linspace(0, tEnd, numPts)

E_0ParamGrid = gt.hermgaussParam(5, data["E_0"], E_0SD, False)

colorScale = E_0ParamGrid[0]
print colorScale
#Rescale to [0,1]
colorScale = (colorScale - np.min(colorScale)) / (np.max(colorScale) - np.min(colorScale))

print colorScale

colors = [cm.bwr(x) for x in colorScale]

#Generate currents
IVals = [None] * 5

# Start with capacitive current
k_0 = data["k_0"]
data["k_0"] = 0
ICap, _ = st.solveIFromJSON(t, data)
data["k_0"] = k_0

# Now, compute the component currents.
for i, (color, E_0, weight) in enumerate(zip(colors, *E_0ParamGrid)):
    data["E_0"] = E_0 
    IVals[i],_ = st.solveIFromJSON(t, data)
    IVals[i] -= ICap
    IVals[i] *= weight
    envUpper = st.interpolatedUpperEnvelope(t, IVals[i])
    envLower = st.interpolatedLowerEnvelope(t, IVals[i])
    plt.plot(t, envUpper, color=color)
    plt.plot(t, envLower, color=color)


IDisp = sum(IVals)
plt.fill_between(t, IDisp, -IDisp)
plt.show(block=True)



    
