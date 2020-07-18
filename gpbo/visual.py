import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import glob
import os
from datetime import datetime


def f(x, y):
	return np.log(((4 - 2.1*pow(x*3,2) + pow(x*3,4)/3)*pow(x*3,2) + x*3*y*2 + (-4 + 4*pow(y*2,2))*pow(y*2,2)) - (-1.0316) + 1)
	#return 2*pow(x,2) - 1.05*pow(x,4) + pow(x,6)/6 + x*y + pow(y,2)

def camel3(x1,x2):
    #3hump camel funciton
    z = [x1*1, x2*1]
    y = 2*z[0]**2-1.05*z[0]**4+(z[0]**6)/6. +z[0]*z[1] + z[1]**2
    #noise
    #final = sp.log(y -(0) + 1) * sp.exp(0.01*sp.random.normal(size=(len(x1),len(x2))))
    final = sp.log(y -(0) + 1) + 0.5*sp.random.normal(size=(len(x1),len(x2)))
    return final

def camel6(x1, x2):
    # Scale axes, [-3,3] and [-2,2]
    z = [x1*3, x2*2]
    y = (4 - 2.1*z[0]**2 + z[0]**4/3)*z[0]**2 + z[0]*z[1] + (-4 + 4*z[1]**2)*z[1]**2
    #y = (4 - 2.1*pow(x[0],2) + pow(x[0],4)/3)*pow(x[0],2) + x[0]*x[1] + (-4 + 4*pow(x[1],2))*pow(x[1],2)
    y = sp.log(y - (-1.03162845348987744408920985) + 1)
    return y

def michalewicz(x1, x2):
    # Scale axes, 
    z = [x1*sp.pi/2 + sp.pi/2, x2*sp.pi/2 + sp.pi/2]
    sum = 0
    for i in range(0,2):
        sum += sp.sin(z[i])*(sp.sin((i+1)*(z[i]**2)/sp.pi))**(2*10)
    y = -sum
    #y = sp.log(y -(-1.8013034101) + 1)
    return y

def expf(x1,x2):
    # Scale axes, 
    z = [x1, x2]
    sum = 0
    for i in range(0,2):
        sum += z[i]**2
    y = -sp.exp(-0.5*sum)
    return y

def xinsheyangN4(x1,x2):
    # Scale axes, 
    z = [x1*10,x2*10]
    sum1 = 0
    for i in range(0,2):
        sum1 += np.sin(z[i])**2
    sum2 = 0
    for i in range(0,2):
        sum2 += z[i]**2
    sum3 = 0
    for i in range(0,2):
        sum3 += np.sin(np.sqrt(np.abs(z[i])))**2
    y = (sum1 - np.exp(-sum2))*np.exp(-sum3)
    return y

def schwefel(x1,x2):
    # Scale axes, 
    z = [x1*500,x2*500]
    sum1 = 0
    for i in range(0,2):
        sum1 += z[i] * np.sin(np.sqrt(np.abs(z[i])))
    y = 418.982887272433799807913601398*2 - sum1
    #y = 418.9829*D - sum1
    #y = sp.log(y -(0) + 1)
    return y

def deceptive(x1,x2):
    # Scale axes, 
    print(x1)
    print(x2)
    z = [x1*0.5 + 0.5, x2*0.5+0.5]
    sum1 = 0
    for i in range(0,2):
        alpha = (i+1)/(2+1)
        print(z[i])
        if(0 <= z[i] and z[i] < 4/5*alpha):
            sum1 += -z[i]/alpha + 4/5
        elif(4/5*alpha < z[i] and z[i] <= alpha):
            sum1 += 5*z[i]/alpha - 4
        elif(alpha < z[i] and z[i] <= (1+4*alpha)/5):
            sum1 += 5*(z[i]-alpha)/(alpha-1) + 1
        elif((1+4*alpha)/5 < z[i] and z[i] <= 1):
            sum1 += (z[i]-1)/(1-alpha) + 4/5
    y = -(1/2*sum1)**2
    return y

def ackley(x1,x2):
    # Scale axes, 
    z = [x1*30, x2*30]
    sum1 = 0
    sum2 = 0
    for i in range(0,2):
        sum1 += z[i]**2
        sum2 += sp.cos(2*sp.pi*z[i])
    y = -20*sp.exp(-(1/2)*sp.sqrt(1/2*sum1)) - sp.exp((1/2)*sum2) + 20 + sp.exp(1)
    return y

def rastrigin(x1,x2):
    # Scale axes, 
    z = [x1*1.28, x2*1.28]#[x1*5.12, x2*5.12]#
    sum1 = 0
    for i in range(0,2):
        sum1 += z[i]**2 - 10*sp.cos(2*sp.pi*z[i])
    y = 10*2 + sum1
    # noise
    #s = (80-0)*0.4#1e-2 #variance
    #n = np.abs(sp.random.normal(size=(len(x1),len(x2)))) * sp.sqrt(s)
    #print('y {} + n {}'.format(y,n))
    #final = y + n
    #final = sp.log(final -(0) + 1)
    #print('transformed y {}'.format(final))
    #final = sp.log(y * sp.exp(0.05*sp.random.normal(size=(len(x1),len(x2)))) - (0) + 1)
    final = sp.log(y -(0) + 1)  + 0.05*sp.random.normal(size=(len(x1),len(x2)))
    return -final#y + n

def sphere(x1,x2):
    # Scale axes, 
    z = [x1*5.12, x2*5.12]
    sum1 = 0
    for i in range(0,2):
        sum1 += z[i]**2
    y = sum1
    #y = sp.log(sum1 -(0) + 1)
    # noise
    s = (45-0)*0.1#1e-2 #variance
    n = np.abs(sp.random.normal(size=(len(x1),len(x2))) * sp.sqrt(s))
    print('y {} + n {}'.format(y,n))
    final = y + n
    final = sp.log(final -(0) + 1)
    print('transformed y {}'.format(final))
    return final#y + n

def rosenbrock(x1,x2):
    # Scale axes, 
    z = [x1*2.048, x2*2.048]#[x1*5, x2*5]
    sum1 = 0
    for i in range(0,2-1):
        sum1 += 100*(z[i+1] - z[i]**2)**2 + (1-z[i])**2
    y = sum1
    y = sp.log(y -(0) + 1)
    # noise
    #final = sp.log(y * sp.exp(0.01*sp.random.normal(size=(len(x1),len(x2)))) - (0) + 1)
    return y


def visual3d(f): #TODO
	#temporary
	y = f(x1,x2)

	# for noisy
	y = df['y'].values
	#y = np.exp(y) - 2

	# only when not plotting queried points
	x1Bound = [-5,5]
	x2Bound = [-5,5]
	# TODO: user defined space
	xRange = np.linspace(x1Bound[0], x1Bound[1],300)
	yRange = np.linspace(x2Bound[0], x2Bound[1],300)

	X, Y = np.meshgrid(xRange, yRange)
	Z = f(X, Y)
	#Z, _, _ = camel3([X, Y])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(x1, x2, y, c='red')
	ax.plot_surface(X, Y, Z, cmap='jet', rcount=500, ccount=500)
	#ax.plot_wireframe(X, Y, Z, rcount=100, ccount=100)
	ax.set_xlabel('x0')
	ax.set_ylabel('x1')
	ax.set_zlabel('y')
	print(ax.get_zlim())
	print(ax.get_zlim()[::-1])
	ax.set_zlim(ax.get_zlim()[::-1])
	plt.savefig('results/test3d', dpi=300)
	return

def visualize(f, path):
	df = pd.read_csv(path+'.csv', sep=', ', header=0)

	# Assume 2D functions
	x1 = df['x0'].values
	x2 = df['x1'].values
	y = df['y'].values

	# TODO: user defined space
	xRange = np.linspace(-1,1,50)
	yRange = np.linspace(-1,1,50)

	X, Y = np.meshgrid(xRange, yRange)
	Z = f(X, Y)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x1, x2, y, c='red')
	ax.plot_surface(X, Y, Z, cmap='jet')
	ax.set_xlabel('x0')
	ax.set_ylabel('x1')
	ax.set_zlabel('y')
	plt.savefig(path, dpi=300)

	return 

# Calculate the performance measure tsp: number of function evalutations required to converge
def tspCalc(y, ymin, r):
	tsp = np.full(len(y), -1)

	for xidx, yr in enumerate(y):
		y0 = yr[0]
		for yidx, yi in enumerate(yr):
			if (y0 - yi >= (1-r)*(y0 - ymin)):
				tsp[xidx] = yidx+1
				break
	return tsp

# Calculate how many of the runs converge within alpha function evaluations
def dataProfile(tsps, alpha, d):
	numDone = 0
	for tsp in tsps:
		for tr in tsp:
			if (tr <= alpha and tr > 0):
				numDone += 1

	ds = (1/(tsps.shape[0]*tsps.shape[1]))*numDone
	return ds

def plotDataprofie(numProblems, numRuns, r, numIterations, manyTrueys, globalymin):
	dataProfiles = []
	d = 2
	numIterations = 84
	for trueys in manyTrueys:
		tsps = np.full((numProblems, numRuns),-1)
		for i in range(0,numProblems):
			tsp = tspCalc(trueys[i*numRuns:(i+1)*numRuns], 0, r)#globalymin[i], r)
			tsps[i] = tsp
		dps = [0]
		for alpha in range(0,numIterations):
			dps.append(dataProfile(tsps, (alpha+1)*(d+1),0))
		dataProfiles.append(dps)
		d += 2

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('\u03B1')
	ax1.set_ylabel('Dataprofile d(\u03B1)')
	labels = ['BLOSSOM 1e-2', '4D']
	for idx, dp in enumerate(dataProfiles):
		ax1.plot(dp, label=labels[idx])
	ax1.set_xbound(0,numIterations)
	ax1.set_ybound(0,1)
	ax1.grid()
	ax1.tick_params(axis='y')
	ax1.legend()

	ax1.annotate('\u03C4 = {}'.format(r), xy=(0.01, 0.96), xycoords='axes fraction', fontsize = 12)
	
	fig.tight_layout()
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	plt.savefig('results/dataprofile' + timestamp, dpi=300)
	return

def plotRegret(manyRegrets, manyLengths):
	# only does it for first batch of runs
	manyYs = []
	manyFracleft = []
	for idx, regrets in enumerate(manyRegrets):
		lengths = manyLengths[idx]
		maxlength = max(lengths)

		n = len(regrets)
		
		ys = []
		fracleft = []
		for i in range(0,maxlength):
			sum = 0
			num = 0
			for l in regrets:
				if i < len(l):
					sum += l[i]
					num += 1
			ys.append(sum/num)
			fracleft.append(num/n)	
		
		regretsum = 0
		stepsum = 0
		for l in regrets:
			regretsum += l[-1]
			stepsum += len(l)
		avgregret = regretsum/n
		avgstep = stepsum/n

		print('Avg regret: '+str(avgregret))
		print('Avg steps: '+str(avgstep))
		manyYs.append(ys)
		manyFracleft.append(fracleft)

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('Step')
	ax1.set_ylabel('Regret')
	labels = ['Modified \u03C3 0.05', 'Unmodified \u03C3 0.005', 'Modified \u03C3 0.005', 'Unmodified \u03C3 0.05']
	for idx, ys in enumerate(manyYs):
		ax1.plot(ys, label=labels[idx])#, color='blue')
	ax1.set_yscale('log')
	ax1.tick_params(axis='y')

	
	ax2 = ax1.twinx()
	ax2.set_ylabel('Fraction still running')
	for fracleft in manyFracleft:
		ax2.plot(fracleft, linestyle='dashed')#color='blue',
	ax2.tick_params(axis='y')

	ax1.legend()
	fig.tight_layout()
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	plt.savefig('results/multiregret' + timestamp, dpi=300)
	return

path = sys.argv[1]
try:
	multi = sys.argv[2]
except IndexError:
	multi = False

# Single file
if not multi:
	df = pd.read_csv(path, sep=', ', header=0)

	#bounds
	#x1Bound=[0.395, 0.410]
	#x2Bound=[-0.01, 0.01]
	x1Bound=[-1, 1.]
	x2Bound=[-1, 1.]
	

	# Assume 2D functions
	x1 = df['x0'].values.clip(x1Bound[0], x1Bound[1])
	x2 = df['x1'].values.clip(x2Bound[0], x2Bound[1])
	y = df['y'].values
	sepValues = pd.DataFrame(df['truey at xrecc'].str.split(',').to_list(), columns=['truey at xrecc', 'taq'])
	ymins = sepValues['truey at xrecc'].values.astype(float)

	minyvalue = -1.03162845348987744408920985
	regret = sp.exp(ymins) - 1
	truey = regret + minyvalue

	visual3d(camel3)

	fig = plt.figure()
	print(regret)
	plt.plot(regret)
	plt.yscale('log')
	plt.savefig('results/test', dpi=300)
else: #Multi file
	
	#minyvalue = -1.03162845348987744408920985 #6humpcamel
	manyRegrets = []
	manyLengths = []
	manyTrueys = []
	globalymin = [0, -1, 0, 0, 0, 0] #check order compared to file read order
	numRuns = 4
	numProblems = 1

	for entry in os.listdir(path):
		fullpath = os.path.join(path,entry)
		print('path {}, entry {}, fullpath {}'.format(path,entry,fullpath))
		if os.path.isdir(fullpath):
			regrets = []
			lengths = []
			trueys = []

			count = 0
			for f in sorted(glob.glob(fullpath+'/*.csv')):
				df = pd.read_csv(f, sep=', ')#, usecols=range(0,16))
				lengths.append(len(df.index))
				sepValues = pd.DataFrame(df['truey at xrecc'].str.split(',').to_list(), columns=['truey at xrecc', 'taq'])
				ymins = sepValues['truey at xrecc'].values.astype(float)
				regret = sp.exp(ymins) - 1 # check against functionfiles if transform is used
				#regret = ymins
				truey = regret #+ globalymin[count//numRuns]
				regrets.append(regret)
				trueys.append(truey)
				count += 1
			manyRegrets.append(regrets)
			manyLengths.append(lengths)
			manyTrueys.append(trueys)

	### Regret plotting
	plotRegret(manyRegrets, manyLengths)

	### Dataprofile plotting

	r = 1e-1
	numIterations = 250
	plotDataprofie(numProblems, numRuns, r, numIterations, manyTrueys, globalymin)
	r = 1e-2
	plotDataprofie(numProblems, numRuns, r, numIterations, manyTrueys, globalymin)
	