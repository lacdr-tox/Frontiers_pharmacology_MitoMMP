import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from scipy.integrate import odeint
import cPickle as pickle
import os
import argparse
parser = argparse.ArgumentParser(description='Plot simulation of MMP upon a treatment.')
parser.add_argument('--compound',  nargs='+', default= ['Deguelin', 'Azoxystrobin'])
args = parser.parse_args()

currentDir = os.path.dirname(os.path.abspath(__file__))
KEi = 1.0
VU = 1.0
KAi = 1.0
KU = 1.0
Cf = 1.0
KE = 2.0
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

Xtime = np.arange(0, 23.05, 0.1)
tspan = Xtime.copy()
Yconcentration = np.arange(np.log10(0.0000128), np.log10(10.0), (np.log10(10.0)-np.log10(0.0000128))/len(Xtime))
X, Y = np.meshgrid(Xtime, Yconcentration)

def mapEc(contentration, log10appliedConcentration, EcList):
	Ec = np.interp(contentration, log10appliedConcentration, EcList)
	return Ec

def simulationMitoMMP(parameter, compound, modelIndex):
	DEi = 0
	DAi = 0
	DUi = 0

	alpha = 0
	gamma = 0
	ymatrix = np.array([]).reshape([0, len(tspan)])
	if modelIndex == 0:
		if compound == 'Oligomycin':
			nCon = 8
			VA, KA, r, c1, c0, D0, D1, D2, D3, D4, D5, D6, D7 = parameter
			Dlist = [D0, D1, D2, D3, D4, D5, D6, D7]
		else:
			nCon = 10
			VA, KA, r, c1, c0, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 = parameter
			Dlist = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9]
	elif modelIndex == 1:
		if compound == 'Oligomycin':
			nCon = 8
			VA, KA, r, c1, c0, gamma, D0, D1, D2, D3, D4, D5, D6, D7 = parameter
			Dlist = [D0, D1, D2, D3, D4, D5, D6, D7]
		else:
			nCon = 10
			VA, KA, r, c1, c0, gamma, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 = parameter
			Dlist = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9]
	else:
		if compound == 'Oligomycin':
			nCon = 8
			VA, KA, r, c1, c0, gamma, alpha, D0, D1, D2, D3, D4, D5, D6, D7 = parameter
			Dlist = [D0, D1, D2, D3, D4, D5, D6, D7]
		else:
			nCon = 10
			VA, KA, r, c1, c0, gamma, alpha, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 = parameter
			Dlist = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9]

	for indexD, Di in enumerate(Dlist):
		if compound == 'Oligomycin':
			parameterPerCon = np.array([VA, KA, r, c1, c0, gamma, alpha, 0, Di, 0])
		elif compound == 'FCCP':
			parameterPerCon = np.array([VA, KA, r, c1, c0, gamma, alpha, 0, 0, Di])
		else:
			parameterPerCon = np.array([VA, KA, r, c1, c0, gamma, alpha, Di, 0, 0])
		xo1 = 0.6 * KE ** (-1.0)

		try:
			xo2 = 0.6 * Cf * KA / (VA - 0.6 * Cf)
		except:
			pass
		x0 = np.array([xo1, xo2])
		y = integrateODE(rhs, x0, tspan, parameterPerCon)
		MMP = c1 * y[:, 1] + c0
		MMP = MMP.flatten()
		ymatrix = np.vstack([ymatrix, MMP])
	return ymatrix

def rhs(x, t, parameter):
	x1, x2 = x
	VA, KA, r, c1, c0, gamma, alpha, DEi, DAi, DUi = parameter
	try:
		DEt = DEi * np.exp(-gamma * t)
		DAt = DAi * np.exp(-gamma * t)
		DUt = DUi * np.exp(-gamma * t)
	except:
		pass

	f0 = r * (0.6 - x1 * KE * KEi / (KEi + DEt))
	f1 = r * (Cf * x1 * KE * KEi / (KEi + DEt) - (VA * x2 / (KA + x2)) * KAi / (KAi + DAt) - VU * DUt * x2 / (
			KU + DUt)
	          - alpha * (x2 - 0.6 * Cf * KA / (VA - 0.6 * Cf)))
	f = np.array([f0, f1])
	return f

def integrateODE(funrhs, x0, tspan, parameter):
	y = odeint(funrhs, x0, tspan, args=(parameter,), atol=1e-13, rtol=1e-13)
	return y

def readDat(compound):
	if compound == 'Oligomycin':
		nConcentration = 8
	else:
		nConcentration = 10
	filenameDataTemp = './data/DataCorr_' + \
	                   compound + '.dat'
	with open(filenameDataTemp, 'rb') as fp:
		dataMeanS = pickle.load(fp)
		dataRep = pickle.load(fp)

	dataM = np.array([])
	dataS = np.array([])
	for i in range(nConcentration):
		dataM = np.hstack([dataM, dataMeanS[2 * i]])
		dataS = np.hstack([dataS, dataMeanS[2 * i + 1]])
	dataNoiseT = [np.array(dataM).reshape(23 * nConcentration, ), np.array(dataS).reshape(23 * nConcentration, )]
	return dataNoiseT, dataRep
compoundlabelL = args.compound
fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(111, projection='3d')
fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111, projection='3d')

concentrationList = np.array([0.000128, 0.00064, 0.0032, 0.016, 0.08, 0.4, 1, 2, 5, 10])
log10appliedConcentration = np.log10(concentrationList)
pathname = ''
csvMLE_BasicModel = pathname + './estimates/BasicM1ALLETCCompounds.csv'

for (compound, fig, ax) in zip(compoundlabelL, [fig1, fig2], [ax1, ax2]):
	Data = readDat(compound)
	MMPmean = Data[0][0]
	XList = np.tile(np.arange(1, 24, 1.0), [10, 1]).flatten()
	YList = np.transpose(np.tile(np.transpose(log10appliedConcentration), [23, 1])).flatten()
	
	df = pd.read_csv(csvMLE_BasicModel)
	ZSimulation = np.array([]).reshape(0, len(Xtime))
	for Yconcentrationi, contentrationi in enumerate(Yconcentration):
	    modelIndex = 0
	    Listparam = ["VAparam", "KAparam", "rparam", "c1param", "c0param",
	             "D0param", "D1param", "D2param", "D3param", "D4param", "D5param", "D6param", "D7param",
	             "D8param", "D9param"]

	    MLEcompound = df[df['compound'] == compound][Listparam]
	    MLEcompound = np.array(MLEcompound)[0]
	    MLEcompound = [float(i) for i in MLEcompound]
	    c1 = MLEcompound[3]
	    c0 = MLEcompound[4]
	    KA = MLEcompound[1]
	    VA = MLEcompound[0]
	    xo1 = 0.6 * KE ** (-1.0)

	    try:
	        xo2 = 0.6 * Cf * KA / (VA - 0.6 * Cf)
	    except:
	        pass
	    x0 = np.array([xo1, xo2])
	    concentrationList = np.array([0.000128, 0.00064, 0.0032, 0.016, 0.08, 0.4, 1, 2, 5, 10])
	    log10appliedConcentration = np.log10(concentrationList)
	    Ec = mapEc(contentrationi, log10appliedConcentration, MLEcompound[-10:])
	    parameterPerCon  = np.array([MLEcompound[0],
	                                 MLEcompound[1],
	                                 MLEcompound[2],
	                                 0, # c1
	                                 0, # c0
	                                 0,
	                                 0,
	                                 Ec,
	                                 0,
	                                 0])
	    y = integrateODE(rhs, x0, Xtime, parameterPerCon)
	    MMP = c1 * y[:, 1] + c0
	    MMP = MMP.flatten()
	    ZSimulation = np.vstack([ZSimulation, MMP])
	if compound == 'Deguelin':
	    ax.plot_surface(X, Y, ZSimulation, rstride=1, cstride=1, cmap='Reds',
	                           linewidth=0,
	                           antialiased=True,
	                           alpha=0.5, label='simulation')
	else:
	    ax.plot_surface(X, Y, ZSimulation, rstride=1, cstride=1, cmap='Blues',
	                          linewidth=0,
	                          antialiased=True,
	                          alpha=0.5)
	ax.xaxis._axinfo['label']['space_factor'] = 28
	ax.yaxis._axinfo['label']['space_factor'] = 28
	ax.set_xlabel('time [hr]', fontsize=20, labelpad=10)
	ax.set_ylabel(r'$\log_{10}$'+' concentration \n' + r'[$\mu$M]', fontsize=20, labelpad=15)
	ax.set_zlabel('normalized MMP', fontsize=20, labelpad=15)
	plt.subplots_adjust(top=0.95)

	ax.set_title(compound, fontsize=20)
	ax.grid(False)
	ax.invert_yaxis()
        ax.scatter(XList, YList, MMPmean, s = 40, color= 'k')
	for i in range(10):
	    ax.plot(XList[0:23], YList[i*23:(i+1)*23], MMPmean[i*23:(i+1)*23], 'k-')
	ax.set_zlim([-0.02,1.3])

	fake2Dline_data = mpl.lines.Line2D([0],[0], linestyle="-", c='k', marker = 'o')
        if compound == 'Deguelin':
            fake2Dline_simu = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o',alpha=0.5)
        else:

            fake2Dline_simu = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o',alpha=0.5)
        ax.legend([fake2Dline_data, fake2Dline_simu], ['data','simulation'], numpoints = 1,fontsize=12)
        fig.tight_layout()
plt.show()
