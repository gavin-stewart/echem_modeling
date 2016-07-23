# Plot and display the harmonics for the dispersion data.

import matplotlib.pyplot as plt
import tools.dataFileIO as dfio
import tools.solutionTools as st

numEvals = [1, 50]

freq = 8.95925020
trim = slice(5000, -5000)

plotsI = plt.figure(1).add_subplot(111)
plotsHarm = plt.figure(2).add_subplot(111)

for runNum, n in enumerate(numEvals):
	t,I = dfio.readTimeCurrentDataBinary("files/dispersion/HermGauss"+str(n)+"pts.npz")
	plotsI.plot(t, I, label=str(n)+" points", alpha=0.4)
	harm10 = st.extractHarmonic(10, freq * t[-1], I)
	plotsHarm.plot(harm10[trim], label=str(n)+" points", alpha = 0.4)

plotsI.legend()
plotsHarm.legend()

plt.show()
