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


def visual3d(f): #TODO
	#temporary
	y = f(x1,x2)

	# TODO: user defined space
	xRange = np.linspace(x1Bound[0], x1Bound[1],100)
	yRange = np.linspace(x2Bound[0], x2Bound[1],100)

	X, Y = np.meshgrid(xRange, yRange)
	Z = f(X, Y)
	#Z, _, _ = camel3([X, Y])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x1, x2, y, c='red')
	#ax.plot_surface(X, Y, Z, cmap='jet')
	ax.plot_wireframe(X, Y, Z, rcount=20, ccount=20)
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

	visual3d(michalewicz)

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
	for f in glob.glob(path+'*.csv'):
		df = pd.read_csv(f, sep=', ', usecols=range(0,16))
		lengths.append(len(df.index))
		sepValues = pd.DataFrame(df['truey at xrecc'].str.split(',').to_list(), columns=['truey at xrecc', 'taq'])
		ymins = sepValues['truey at xrecc'].values.astype(float)
		regret = sp.exp(ymins) - 1
		#truey = regret + minyvalue
		regrets.append(regret)

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
