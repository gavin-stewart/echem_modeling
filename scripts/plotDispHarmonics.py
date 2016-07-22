# Plot and display the harmonics for the dispersion data.

import matplotlib.pyplot as plt
import tools.dataFileIO as dfio
import tools.solutionTools as st

numEvals = [3]

freq = 8.95925020
trim = slice(5000, -5000)

plots = []

for runNum, n in enumerate(numEvals):
	t,I = dfio.readTimeCurrentDataBinary("files/dispersion/benchmark.npz")
	plt.plot(t, I)
	plt.show()
	#harm10 = st.extractHarmonic(10, freq * t[-1], I)
	#plt.figure(runNum)
	#print len(harm10)
	#print harm10
	#plotEntry, = plt.plot(harm10[trim], label=str(n)+" points", alpha = 0.4)
