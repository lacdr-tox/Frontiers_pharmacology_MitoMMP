import pandas as pd
import numpy as np
import glob
import cPickle as pickle
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import argparse
parser = argparse.ArgumentParser(description='Plot simulation of MMP upon a treatment.')
parser.add_argument('--compound', nargs='+', default= 'Antimycin A')
args = parser.parse_args()
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
# paraemeters with fixed values
KE = 2.0
Cf = 1.0
F = 3.75
k = 1.0
KAi = 1.0
KEi = 1.0
VU = 1.0
KU = 1.0
lambdaATP = 1.0
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
    elif modelIndex ==1:
        if compound == 'Oligomycin':
            nCon = 8
            VA, KA, r, c1, c0, gamma, D0, D1, D2, D3, D4, D5, D6, D7 = parameter
            Dlist = [D0, D1, D2, D3, D4, D5, D6, D7]
        else:
            nCon = 10
            VA, KA, r, c1, c0, gamma, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 = parameter
            Dlist = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9]
    elif modelIndex ==2:
        if compound == 'Oligomycin':
            nCon = 8
            VA, KA, r, c1, c0, gamma, alpha, D0, D1, D2, D3, D4, D5, D6, D7 = parameter
            Dlist = [D0, D1, D2, D3, D4, D5, D6, D7]
        else:
            nCon = 10
            VA, KA, r, c1, c0, gamma, alpha, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 = parameter
            Dlist = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9]
    elif modelIndex ==3:
        if compound == 'Oligomycin':
            nCon = 8
            K, r, c1, c0, gammaLow, gammaHigh, D1, D2, D3, D4, D5, D6, D7 = parameter
            D0 = 4.125E+15 # fixed at the estimate to overcome identifiability issue
            # D0 = 0.0
            Dlist = [D0, D1, D2, D3, D4, D5, D6, D7]


    for indexD, Di in enumerate(Dlist):
        if compound == 'Oligomycin':
            if modelIndex < 3:
                parameterPerCon = np.array([VA, KA, r, c1, c0, gamma, alpha, 0, Di, 0])
            else:
                if indexD > 3:
                    gamma = gammaHigh
                else:
                    gamma = gammaLow
                kappa = 0 # no leakage
                parameterPerCon = np.array([K, r, c1, c0, kappa, gamma, Dlist[0], 0, Di, 0])
        elif compound == 'FCCP':
            parameterPerCon = np.array([VA, KA, r, c1, c0, gamma, alpha, 0, 0, Di])
        else:
            parameterPerCon = np.array([VA, KA, r, c1, c0, gamma, alpha, Di, 0, 0])
        xo1 = 0.6 * KE ** (-1.0)


        if modelIndex==3:
            xo2 = 0.6 * Cf * K * c1
            x0 = xo2
        else:
            xo2 = 0.6 * Cf * KA / (VA - 0.6 * Cf)
            x0 = np.array([xo1, xo2])
        if modelIndex < 3:
            y = integrateODE(rhs, x0, tspan, parameterPerCon)
            MMP = c1 * y[:,1] +c0
        else:
            y = integrateODE(rhs_kappa, x0, tspan, parameterPerCon)
            MMP = y[:, 0] + c0
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
def rhs_kappa(x, t, parameter):
    x2 = x
    K, r, c1, c0, kappa, gamma, D0, DEi, DAi, DUi = parameter
    try:
        DAt = (DAi) **(-1) * np.exp(gamma * t)
    except:
        pass

    f1 = 0.6 * r * c1 + \
         0.6 * kappa * (D0) ** (-1) * r * c1 \
         - r * x2 / K * DAt \
         - kappa * K ** (-1) * (D0) ** (-1) * r * x2
    return f1
def integrateODE(funrhs, x0, tspan, parameter):
    y = odeint(funrhs, x0, tspan, args = (parameter, ),atol = 1e-13, rtol= 1e-13)

    return y
def readDat(compound):
    if compound == 'Oligomycin':
        nConcentration = 8
    else:
        nConcentration = 10
    filenameDataTemp = './data//DataCorr_' + \
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
    return dataNoiseT,dataRep
tspan = np.arange(0, 23+.0001, .0001)
currentDir = os.path.dirname(os.path.realpath(__file__))
'''
main part
'''
pathname = ''
csvMLE_BasicModel = pathname + './estimates/BasicMSingleCompounds.csv'
csvMLE_PKM        = pathname + './estimates/PKMFCCPOLI.csv'
csvMLE_PKLeakage  = pathname + './estimates/PKLeakageMOLI.csv'
csvMLE_lowhighGamma  = pathname + './estimates/PKLeakageMOLI_noleakage_lowhighGamma.csv' #<-

df_Basic                  = pd.read_csv(csvMLE_BasicModel)
df_PKM                    = pd.read_csv(csvMLE_PKM)
df_PKLeakage              = pd.read_csv(csvMLE_PKLeakage)
df_PKnoLeakage_lowhighGamma = pd.read_csv(csvMLE_lowhighGamma)

colorL = ['c', 'r', 'k', 'b']
labelL = ['basic model',
          'model with compound decay',
          'model with ion leakage and concentration-independent compound decay',
          'model with concentration-dependent compound decay']
compoundList = args.compound
for compound in compoundList:

    dataTemp, dataRep = readDat(compound)
    MMPmean = dataTemp[0]
    MMPstd  = dataTemp[1]
    concentrationList = [0.000128, 0.00064, 0.0032, 0.016, 0.08, 0.4, 1, 2, 5, 10]
    if compound == 'FCCP':
        fig, ax = plt.subplots(2, 5, sharex = True, sharey = True, figsize=(24, 10))
        dfList = [df_Basic,
                  df_PKM]
    elif compound == 'Oligomycin':
        concentrationList = [0.00064, 0.0032, 0.016, 0.08, 0.4, 2, 5, 10]
        fig, ax = plt.subplots(2, 4, sharex = True, sharey = True, figsize=(24, 10))
        dfList = [df_Basic,
                  df_PKM,
                  df_PKLeakage,
                  df_PKnoLeakage_lowhighGamma]
    else:
        fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(16, 8))
        dfList = [df_Basic]
    if compound == 'Picostrobin':
        plt.suptitle('Picoxystrobin', fontsize = 20)
    else:
        plt.suptitle(compound, fontsize = 20)
    for modelIndex, df in enumerate(dfList):
        print(compound)

        if modelIndex == 0:
            if compound == 'Oligomycin':
                Listparam = ["VAparam" ,"KAparam","rparam", "c1param","c0param",
                             "D0param","D1param","D2param","D3param","D4param","D5param","D6param","D7param"]
            else:
                Listparam = ["VAparam" ,"KAparam","rparam", "c1param","c0param",
                         "D0param","D1param","D2param","D3param","D4param","D5param","D6param","D7param","D8param","D9param"]
        elif modelIndex==1:
            if compound == 'Oligomycin':
                Listparam = ["VAparam" ,"KAparam","rparam", "c1param","c0param", "gammaparam",
                             "D0param","D1param","D2param","D3param","D4param","D5param","D6param","D7param"]
            else:
                Listparam = ["VAparam" ,"KAparam","rparam", "c1param","c0param", "gammaparam",
                         "D0param","D1param","D2param","D3param","D4param","D5param","D6param","D7param","D8param","D9param"]
        elif modelIndex == 2:
            if compound == 'Oligomycin':
                Listparam = ["VAparam" ,"KAparam","rparam", "c1param","c0param", "gammaparam", "alphaparam",
                             "D0param","D1param","D2param","D3param","D4param","D5param","D6param","D7param"]
            else:
                Listparam = ["VAparam" ,"KAparam","rparam", "c1param","c0param", "gammaparam", "alphaparam",
                         "D0param","D1param","D2param","D3param","D4param","D5param","D6param","D7param","D8param","D9param"]
        elif modelIndex == 3:
            if compound == 'Oligomycin':
                Listparam = ["Kparam", "rparam", "c1param", "c0param", "gammaLowparam", "gammaHighparam",
                             "D1param", "D2param", "D3param", "D4param", "D5param", "D6param", "D7param"]
        print(modelIndex)
        if modelIndex == 3:
            pass
        MLEcompound = df[df['compound'] == compound][Listparam]
        MLEcompound = np.array(MLEcompound)[0]
        MLEcompound = [float(i) for i in MLEcompound]
        ymatrix = simulationMitoMMP(MLEcompound, compound, modelIndex)
        if (not (compound == 'FCCP')) & (not (compound == 'Oligomycin')):

            f = ((ymatrix[:, 10000::10000]).flatten() - MMPmean) / MMPstd

        elif compound == 'FCCP':
            f = ((ymatrix[:, 10000::10000]).flatten() - MMPmean) / MMPstd
        elif compound == 'Oligomycin':
            f = ((ymatrix[:, 10000::10000]).flatten() - MMPmean) / MMPstd
        nParam = len(Listparam)
        nData = len(MMPmean)
        nCol = ymatrix.shape[0]/2
        for concentrationi in range(ymatrix.shape[0]):
            rowi = concentrationi/nCol
            coli = np.mod(concentrationi, nCol)
            ax[rowi, coli].plot(range(1, 24), dataRep[concentrationi * (nCol - 1)], 'ks-', markersize=3, alpha=0.08)
            ax[rowi, coli].plot(range(1, 24), dataRep[1 + concentrationi * (nCol - 1)], 'ks-', markersize=3, alpha=0.08)
            ax[rowi, coli].plot(range(1, 24), dataRep[2 + concentrationi * (nCol - 1)], 'ks-', markersize=3, alpha=0.08)
            if not compound == 'Oligomycin':
                ax[rowi, coli].plot(range(1, 24), dataRep[3 + concentrationi * (nCol - 1)], 'ks-', markersize=3, alpha=0.08)
            ax[rowi, coli].plot(range(1, 24),
                                dataTemp[0][concentrationi * 23:23 * (concentrationi + 1)],
                                'ko', markersize=6,
                                alpha=0.6)
            
            ax[rowi, coli].fill_between(range(1, 24),
                                    dataTemp[0][concentrationi * 23:23 * (concentrationi + 1)] +
                                    1.98 * dataTemp[1][concentrationi * 23:23 * (concentrationi + 1)],
                                    dataTemp[0][concentrationi * 23:23 * (concentrationi + 1)] -
                                    1.98 * dataTemp[1][concentrationi * 23:23 * (concentrationi + 1)],
                                    color = 'k',
                                    alpha=0.04)
            if concentrationi == 0:
                ax[rowi,coli].plot(tspan[10000::1], ymatrix[concentrationi,10000::1], color = colorL[modelIndex], lw = 3, label= labelL[modelIndex])
            else:
                ax[rowi,coli].plot(tspan[10000::1], ymatrix[concentrationi,10000::1], color = colorL[modelIndex], lw = 3)
    plt.subplots_adjust(top=0.85, bottom=0.1, right=0.99, left=0.08, wspace = 0.05, hspace=0.20)
    # ax[0,0].legend(ncol=1, loc= 'best', fontsize = 14)
    if len(dfList) > 1:
        if compound == 'Oligomycin':
            ax[0,0].legend(ncol=1, loc= 'upper center', fontsize = 10.5, bbox_to_anchor=(1, 1.46))
        else:
            ax[0,0].legend(ncol=1, loc= 'upper center', fontsize = 18, bbox_to_anchor=(1., 1.46))

    for k, axi in enumerate(ax.flat):
        if np.mod(k,5) == 0:
            axi.set(ylabel='normalized MMP [au]')
        elif k>=5:
            axi.set(xlabel='time [hr]')
             
    for concentrationIndex, axi in enumerate(ax.flat):
        axi.label_outer()
        axi.xaxis.label.set_size(20)
        axi.yaxis.label.set_size(20)
        axi.grid('on', axis = 'y')
        axi.set_title( str(concentrationList[concentrationIndex]) + r'$\mu$M', fontsize = 20)
plt.show()

