import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import glob


# TODO: Add way for different functions
def f(x, y):
	return np.log(((4 - 2.1*pow(x*3,2) + pow(x*3,4)/3)*pow(x*3,2) + x*3*y*2 + (-4 + 4*pow(y*2,2))*pow(y*2,2)) - (-1.0316) + 1)
	#return 2*pow(x,2) - 1.05*pow(x,4) + pow(x,6)/6 + x*y + pow(y,2)

def camel3(x,**ev):
    #3hump camel funciton
    z = [5*xi for xi in x]
    f = 2*z[0]**2-1.05*z[0]**4+(z[0]**6)/6. +z[0]*z[1] + z[1]**2
    return f

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



def visual3d(f): #TODO
	#temporary
	y = f(x1,x2)

	# for noisy
	y = df['y'].values
	#y = np.exp(y) - 2

	# TODO: user defined space
	xRange = np.linspace(x1Bound[0], x1Bound[1],100)
	yRange = np.linspace(x2Bound[0], x2Bound[1],100)

	X, Y = np.meshgrid(xRange, yRange)
	Z = f(X, Y)
	#Z, _, _ = camel3([X, Y])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x1, x2, y, c='red')
	ax.plot_surface(X, Y, Z, cmap='jet', rcount=100, ccount=100)
	#ax.plot_wireframe(X, Y, Z, rcount=100, ccount=100)
	ax.set_xlabel('x0')
	ax.set_ylabel('x1')
	ax.set_zlabel('y')
	plt.savefig('results/test3d', dpi=600)
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
	plt.savefig(path, dpi=600)

	return 

def tspCalc(y, ymin, r):
	tsp = np.full((y.shape[0], y.shape[1]), -1)

	for xidx, yp in enumerate(y):
		for yidx, yr in enumerate(yp):
			y0 = yr[0]
			for zidx, yi in enumerate(yr):
				if (y0 - yi >= (1-r)*(y0 - ymin)):
					tsp[xidx][yidx] = zidx+1
					break
	return tsp

def dataProfile(tsp, alpha, d):
	numDone = 0
	for tp in tsp:
		for tr in tp:
			if (tr/(d+1) <= alpha and tr > 0):
				numDone += 1

	ds = 1/tsp.shape[0]*numDone 
	return ds

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

	visual3d(ackley)

	fig = plt.figure()
	print(regret)
	plt.plot(regret)
	plt.yscale('log')
	plt.savefig('results/test', dpi=600)
else: #Multi file
	#df = pd.concat([pd.read_csv(f, sep=', ', usecols=range(0,16)) for f in glob.glob(path+'*.csv')], ignore_index=True)
	
	minyvalue = -1.03162845348987744408920985 #6humpcamel
	
	#regrets = [pd.read_csv(f, sep=', ', usecols=range(0,16)) for f in glob.glob(path+'*.csv')]

	regrets = []
	lengths = []
	trueys = []
	for f in glob.glob(path+'*.csv'):
		df = pd.read_csv(f, sep=', ', usecols=range(0,16))
		lengths.append(len(df.index))
		sepValues = pd.DataFrame(df['truey at xrecc'].str.split(',').to_list(), columns=['truey at xrecc', 'taq'])
		ymins = sepValues['truey at xrecc'].values.astype(float)
		regret = sp.exp(ymins) - 1
		#truey = regret + minyvalue
		regrets.append(regret)
		trueys.append(ymins)

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

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('Step')
	ax1.set_ylabel('Regret')
	ax1.plot(ys, color='blue')
	ax1.set_yscale('log')
	ax1.tick_params(axis='y')
	
	ax2 = ax1.twinx()
	ax2.set_ylabel('Fraction still running')
	ax2.plot(fracleft, color='blue', linestyle='dashed')
	ax2.tick_params(axis='y')

	fig.tight_layout()
	plt.savefig('results/multitest', dpi=600)

	ymin = 0
	r = 0.1
	tsp = tspCalc(trueys, ymin, r)
	dps = []
	for alpha in range(0,250):
		dps.append(dataProfile(tsp, alpha+1,2))

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('Alpha')
	ax1.set_ylabel('ds(alpha)')
	ax1.plot(ys, color='blue')
	#ax1.set_yscale('log')
	ax1.tick_params(axis='y')
	
	fig.tight_layout()
	plt.savefig('results/dataprofiletest', dpi=600)