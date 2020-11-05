import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import scipy as sp
import glob
import os
from datetime import datetime
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from examples.util import *
#import gpbo


def f(x, y):
        return np.log(((4 - 2.1*pow(x*3,2) + pow(x*3,4)/3)*pow(x*3,2) + x*3*y*2 + (-4 + 4*pow(y*2,2))*pow(y*2,2)) - (-1.0316) + 1)
        #return 2*pow(x,2) - 1.05*pow(x,4) + pow(x,6)/6 + x*y + pow(y,2)

def camel3(x1,x2):
    #3hump camel funciton
    z = [x1*5, x2*5]
    y = 2*z[0]**2-1.05*z[0]**4+(z[0]**6)/6. +z[0]*z[1] + z[1]**2
    #noise
    #final = sp.log(y -(0) + 1) * sp.exp(0.01*sp.random.normal(size=(len(x1),len(x2))))
    final = sp.log(y -(0) + 1) + 0.5*sp.random.normal(size=(len(x1),len(x2)))
    return final

def camel3Value(x):
    #3hump camel funciton
    z = [xi*5 for xi in x]
    y = 2*z[0]**2-1.05*z[0]**4+(z[0]**6)/6. +z[0]*z[1] + z[1]**2
    final = sp.log(y + 1)
    #noise
    #final = sp.log(y -(0) + 1) * sp.exp(0.01*sp.random.normal(size=(len(x1),len(x2))))
    #final = sp.log(y -(0) + 1) + 0.5*sp.random.normal(size=(len(x1),len(x2)))
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
    z = [x1*5.12, x2*5.12]#[x1*1.28, x2*1.28]#
    sum1 = 0
    for i in range(0,2):
        sum1 += z[i]**2 - 10*sp.cos(2*sp.pi*z[i])
    y = 10*2 + sum1
    # noise
    final = sp.log(y -(0) + 1)  + 0.005*sp.random.normal(size=(len(x1),len(x2)))
    return final

def rastriginValue(x):
    # Scale axes, 
    z = [xi*5.12 for xi in x]#[x1*1.28, x2*1.28]#
    sum1 = 0
    for i in range(0,2):
        sum1 += z[i]**2 - 10*sp.cos(2*sp.pi*z[i])
    y = 10*2 + sum1
    final = sp.log(y + 1)
    # noise
    #final = sp.log(y -(0) + 1)  + 0.005*sp.random.normal(size=(len(x1),len(x2)))
    return final

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

def rosenbrockPlot(x1,x2):
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

def rosenbrockValue(x):
    # Scale axes, 
    z = [xi*2.048 for xi in x]#[x1*5, x2*5]
    sum1 = 0
    for i in range(0,len(x)-1):
        sum1 += 100*(z[i+1] - z[i]**2)**2 + (1-z[i])**2
    y = sum1
    y = sp.log(y -(0) + 1)
    # noise
    #final = sp.log(y * sp.exp(0.01*sp.random.normal(size=(len(x1),len(x2)))) - (0) + 1)
    return y

def qaoaValue(beta_gamma_angles):

	# read data
	# constraint matrix
	file_location = 'data_instances/'
	file_name = 'instance_8_0.mps'


	[c, A, nbr_of_flights, nbr_of_routes] = read_data_from_mps(file_location, file_name)

	nbr_of_qubits = nbr_of_routes
	nbr_states = 2**nbr_of_qubits



	# compute the ising spin glass hamiltonian
	[linear_cost_hamiltonian,
	quadratic_cost_hamiltonian,
	cost_hamiltonian] =  create_ising_hamiltonian(c,
	                         A,
	                         nbr_of_qubits,
	                         nbr_of_flights)

	# Exact_cover:
	H = quadratic_cost_hamiltonian

	# mixer operator
	sigmax = np.array([[0, 1],
	               [1, 0]])

	initial_state = 1.0/np.sqrt(nbr_states)*np.ones((nbr_states, 1))

	# compute the state for a variational angle for a certain level of p
	# the dimension of the expectation value function is 2*p
	p = int(len(beta_gamma_angles)/2)

	gamma = [b*2*np.pi for b in beta_gamma_angles[:p]]
	beta = [b*np.pi for b in beta_gamma_angles[p:2*p]]

	E = expectation_value((gamma+beta), H, p, nbr_of_qubits, initial_state,sigmax)
	return E

def qaoaSuccessProb(beta_gamma_angles):
	# read data
	# constraint matrix
	file_location = 'data_instances/'
	file_name = 'instance_8_0.mps'


	[c, A, nbr_of_flights, nbr_of_routes] = read_data_from_mps(file_location, file_name)

	nbr_of_qubits = nbr_of_routes
	nbr_states = 2**nbr_of_qubits



    # compute the ising spin glass hamiltonian
	[linear_cost_hamiltonian,
	quadratic_cost_hamiltonian,
	cost_hamiltonian] =  create_ising_hamiltonian(c, A, nbr_of_qubits, nbr_of_flights)

	# Exact_cover:
	H = quadratic_cost_hamiltonian

	# mixer operator
	sigmax = np.array([[0, 1], [1, 0]])

	initial_state = 1.0/np.sqrt(nbr_states)*np.ones((nbr_states, 1))

	# compute the state for a variational angle for a certain level of p
	# the dimension of the expectation value function is 2*p
	p = int(len(beta_gamma_angles)/2)

	# final state
	#gamma_opt= beta_gamma_angles[:p]
	#beta_opt = beta_gamma_angles[p:2*p]
	gamma_opt = [b*2*np.pi for b in beta_gamma_angles[:p]]#(b+1)/2
	beta_opt = [b*np.pi for b in beta_gamma_angles[p:2*p]]

	qaoa_state = state_fast(H, p, nbr_of_qubits, initial_state, sigmax, gamma_opt, beta_opt)
	# determine what the optimal value is
	opt_val_idx = np.argmin(H)#3 # this must be known for the classical optimization problem
	prob_of_obtaining_correct_answer_1_shot = np.abs(qaoa_state[opt_val_idx])**2
	return prob_of_obtaining_correct_answer_1_shot

def visual3d(f): #TODO
        #temporary
        y = f(x1,x2)

        # for noisy
        y = df['y'].values
        #y = np.exp(y) - 2

        # only when not plotting queried points
        x1Bound = [-1, 1]
        x2Bound = [-1, 1]
        # TODO: user defined space
        xRange = np.linspace(x1Bound[0], x1Bound[1],300)
        yRange = np.linspace(x2Bound[0], x2Bound[1],300)

        X, Y = np.meshgrid(xRange, yRange)
        Z = f(X, Y)
        #Z, _, _ = camel3([X, Y])

        fig = plt.figure(figsize=(12.8, 4.8)) #default (6.4, 4.8)
        # First subplot
        ax = fig.add_subplot(121, projection='3d')
        #ax.scatter(x1, x2, y, c='red')
        ax.plot_surface(X, Y, Z, cmap='jet', rcount=500, ccount=500)
        #ax.plot_wireframe(X, Y, Z, rcount=100, ccount=100)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
        #print(ax.get_zlim())
        #print(ax.get_zlim()[::-1])
        ax.set_zlim(ax.get_zlim()[::-1])

        # Second subplot
        x1Bound = [-0.2,0.2]
        x2Bound = [-0.2,0.2]
        xRange = np.linspace(x1Bound[0], x1Bound[1],300)
        yRange = np.linspace(x2Bound[0], x2Bound[1],300)
        X, Y = np.meshgrid(xRange, yRange)
        Z = f(X, Y)

        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='jet', rcount=500, ccount=500)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
        ax.set_zlim(ax.get_zlim()[::-1])

        plt.tight_layout()
        plt.savefig('results/test3d', dpi=200)
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
        plt.savefig(path, dpi=200)

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
        plt.savefig('results/dataprofile' + timestamp, dpi=200)
        return

def plotRegret(manyRegrets, manyLengths, plotOrder):
	# only does it for first batch of runs
	manyYs = []
	manyFracleft = []
	for idx, regrets in enumerate(manyRegrets):
		lengths = manyLengths[idx]
		maxlength = max(lengths)
		print(lengths)

		n = len(regrets)
		
		ys = []
		fracleft = []
		for i in range(0,min(maxlength,250)):
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
			print("last regret {}".format(l[-1]))
			regretsum += l[-1]
			stepsum += len(l)
		avgregret = regretsum/n
		avgstep = stepsum/n

		print('Plotorder: {}'.format(plotOrder[idx]))
		print('Avg regret: '+str(avgregret))
		print('Avg steps: '+str(avgstep))
		manyYs.append(ys)
		manyFracleft.append(fracleft)

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('Step')
	ax1.set_ylabel('Regret')
	#labels = ['BLOSSOM \u03C3=0.005', 'BLOSSOM \u03C3=0.05', 'BLOSSOM \u03C3=0', 'PES \u03C3=0.5', 'PES \u03C3=0.05', 'PES \u03C3=0', 'BLOSSOM \u03C3=0.5', 'PES \u03C3=0.005']
	#labels = ['PES \u03C3=0.005', 'BLOSSOM \u03C3=0.05', 'BLOSSOM \u03C3=0', 'PES \u03C3=0.5', 'PES \u03C3=0.05', 'BLOSSOM \u03C3=0.5', 'BLOSSOM \u03C3=0.005', 'PES \u03C3=0']
	labels = ['BLOSSOM \u03C3=0', 'BLOSSOM \u03C3=0.005', 'BLOSSOM \u03C3=0.05', 'BLOSSOM \u03C3=0.5']
	#labels = ['BLOSSOM 2D', 'BLOSSOM 8D', 'BLOSSOM 4D']
	#labels = ['unmodified \u03C3=0.005', 'modified \u03C3=0.005', 'unmodified \u03C3=0.05', 'modified \u03C3=0.05', 'unmodified \u03C3=0',]
	for idx, ys in enumerate(manyYs):
		linestyle = 'solid'
		if(plotOrder[idx].find('pes') >=0):
			linestyle = 'dashed'
		elif(plotOrder[idx].find('ei') >=0):
			linestyle = 'dotted'
		color = 'green'
		if(plotOrder[idx].find('low') >= 0):
			color='blue'
		elif(plotOrder[idx].find('med') >= 0):
			color='purple'
		elif(plotOrder[idx].find('high') >= 0):
			color='red'
		elif(plotOrder[idx].find('biggest') >= 0):
			color='brown'
		ax1.plot(ys, linestyle=linestyle, color=color)#, label=labels[idx])
	#ax1.set_ylim(0.6*10e-1, 3.2*10e2)
	#ax1.set_yscale('log')
	ax1.tick_params(axis='y')

	
	# ax2 = ax1.twinx()
	# ax2.set_ylabel('Fraction still running')
	# for idx, fracleft in enumerate(manyFracleft):
	# 	linestyle = 'solid'
	# 	if(plotOrder[idx].find('pes') >=0):
	# 		linestyle = 'dashed'
	# 	elif(plotOrder[idx].find('ei') >=0):
	# 		linestyle = 'dotted'
	# 	color = 'green'
	# 	if(plotOrder[idx].find('low') >= 0):
	# 		color='blue'
	# 	elif(plotOrder[idx].find('med') >= 0):
	# 		color='purple'
	# 	elif(plotOrder[idx].find('high') >= 0):
	# 		color='red'
	# 	ax2.plot(fracleft, linestyle='dotted', color=color)#color='blue',
	# ax2.set_ylim(-0.05, 1.05)
	# ax2.tick_params(axis='y')

	brown_patch = mpatches.Patch(color='brown', label='Very high noise')
	red_patch = mpatches.Patch(color='red', label='High noise')
	purple_patch = mpatches.Patch(color='purple', label='Medium noise')
	blue_patch = mpatches.Patch(color='blue', label='Low noise')
	green_patch = mpatches.Patch(color='green', label='No noise')
	blossom_line = mlines.Line2D([], [], color='black', label='Blossom')
	pes_line = mlines.Line2D([], [], color='black', linestyle='dashed', label='PES')
	ei_line = mlines.Line2D([], [], color='black', linestyle='dotted', label='EI')
	ax1.legend(handles=[brown_patch, red_patch, purple_patch,  green_patch, blossom_line, ei_line])
	#ax1.legend()
	fig.tight_layout()
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	plt.savefig('results/multiregret' + timestamp, dpi=200)
	return

def plotSuccessProb(manyXreccs, manyLengths, plotOrder):

	manySuccessProbs = []
	manyFracleft = []
	for idx, xreccss in enumerate(manyXreccs):
		lengths = manyLengths[idx]
		maxlength = max(lengths)
		print(lengths)

		n = len(xreccss)
		
		successProbs = []
		fracleft = []
		for i in range(0,min(maxlength,250)):
			sum = 0
			num = 0
			for l in xreccss:
				if i < len(l):
					sum += qaoaSuccessProb([(x+1)/2 for x in l[i]]) #*2*np.pi etc
					num += 1
				else:
					sum += qaoaSuccessProb([(x+1)/2 for x in l[-1]])
					num += 1
			successProbs.append(sum/num)
			fracleft.append(num/n)	
		
		successprobsum = 0
		stepsum = 0
		for l in xreccss:
			successprobsum += qaoaSuccessProb([(x+1)/2 for x in l[-1]])
			stepsum += len(l)
		avgsuccessprob = successprobsum/n
		avgstep = stepsum/n

		print('Plotorder: {}'.format(plotOrder[idx]))
		print('Avg success prob: '+str(avgsuccessprob))
		print('Avg steps: '+str(avgstep))
		manySuccessProbs.append(successProbs)
		manyFracleft.append(fracleft)

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('Step')
	ax1.set_ylabel('Success probability')
	#labels = ['BLOSSOM \u03C3=0.005', 'BLOSSOM \u03C3=0.05', 'BLOSSOM \u03C3=0', 'PES \u03C3=0.5', 'PES \u03C3=0.05', 'PES \u03C3=0', 'BLOSSOM \u03C3=0.5', 'PES \u03C3=0.005']
	#labels = ['PES \u03C3=0.005', 'BLOSSOM \u03C3=0.05', 'BLOSSOM \u03C3=0', 'PES \u03C3=0.5', 'PES \u03C3=0.05', 'BLOSSOM \u03C3=0.5', 'BLOSSOM \u03C3=0.005', 'PES \u03C3=0']
	labels = ['BLOSSOM \u03C3=0', 'BLOSSOM \u03C3=0.005', 'BLOSSOM \u03C3=0.05', 'BLOSSOM \u03C3=0.5']
	#labels = ['BLOSSOM 2D', 'BLOSSOM 8D', 'BLOSSOM 4D']
	#labels = ['unmodified \u03C3=0.005', 'modified \u03C3=0.005', 'unmodified \u03C3=0.05', 'modified \u03C3=0.05', 'unmodified \u03C3=0',]
	for idx, successprobs in enumerate(manySuccessProbs):
		linestyle = 'solid'
		if(plotOrder[idx].find('pes') >=0):
			linestyle = 'dashed'
		elif(plotOrder[idx].find('ei') >=0):
			linestyle = 'dotted'
		color = 'green'
		if(plotOrder[idx].find('low') >= 0):
			color='blue'
		elif(plotOrder[idx].find('med') >= 0):
			color='purple'
		elif(plotOrder[idx].find('high') >= 0):
			color='red'
		elif(plotOrder[idx].find('biggest') >= 0):
			color='brown'
		ax1.plot(successprobs, linestyle=linestyle, color=color)#, label=labels[idx])
	#ax1.set_ylim(0.6*10e-1, 3.2*10e2)
	#ax1.set_yscale('log')
	ax1.tick_params(axis='y')


	brown_patch = mpatches.Patch(color='brown', label='Very high noise')
	red_patch = mpatches.Patch(color='red', label='High noise')
	purple_patch = mpatches.Patch(color='purple', label='Medium noise')
	blue_patch = mpatches.Patch(color='blue', label='Low noise')
	green_patch = mpatches.Patch(color='green', label='No noise')
	blossom_line = mlines.Line2D([], [], color='black', label='Blossom')
	pes_line = mlines.Line2D([], [], color='black', linestyle='dashed', label='PES')
	ei_line = mlines.Line2D([], [], color='black', linestyle='dotted', label='EI')
	ax1.legend(handles=[brown_patch, red_patch, purple_patch,  green_patch, blossom_line, ei_line])
	#ax1.legend()
	fig.tight_layout()
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	plt.savefig('results/successprobs' + timestamp, dpi=200)
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
        plt.savefig('results/test', dpi=200)
else: #Multi file
	
	#minyvalue = -1.03162845348987744408920985 #6humpcamel
	manyRegrets = []
	manyLengths = []
	manyTrueys = []
	manyXreccs = []
	globalymin = [0, -1, 0, 0, 0, 0] #check order compared to file read order
	numRuns = 4
	numProblems = 1
	plotOrder = []

	for entry in sorted(os.listdir(path)):
		fullpath = os.path.join(path,entry)
		print('path {}, entry {}, fullpath {}'.format(path,entry,fullpath))
		if os.path.isdir(fullpath):
			if(entry.find('pes') >= 0): #match find to namingscheme
				plotEntry = 'pes'
			elif(entry.find('ei') >= 0):
				plotEntry = 'ei'
			else:
				plotEntry = 'blossom'
			if(entry.find('e0005') >= 0):
				plotEntry += 'low'
				print("low")
			elif(entry.find('e005') >= 0):
				plotEntry += 'med'
				print("med")
			elif(entry.find('e05') >= 0):
				plotEntry += 'high'
				print("high")
			elif(entry.find('e25') >= 0):
				plotEntry += 'biggest'
				print("biggest")
			else:
				plotEntry += 'none'
				print("none")
			plotOrder.append(plotEntry)

			regrets = []
			lengths = []
			trueys = []
			xreccss = []

			count = 0
			func = qaoaValue#rastriginValue#camel3Value#rosenbrockValue#
			if(entry.find('ei') >= 0):
				for f in sorted(glob.glob(fullpath+'/*evals*.txt')):
					df = pd.read_csv(f, sep='\t')
					bestys = []
					besty = func(df.iloc[0, 2:].values)
					bestx = np.copy(df.iloc[0, 2:].values)
					xreccs = df.iloc[:, 2:].values
					for idx, var in enumerate(df.iloc[:, 2:].values):
						y = func(var) #is this correct for qaoa?
						if(y < besty):
							besty = y
							bestx = np.copy(var)
							#print("bestx {}".format(bestx))
							#print("updating bestx {}".format(bestx*2-1))
							#print(xreccs[idx])
						bestys.append(besty)
						#print("bestx 2nd {}".format(bestx))
						xreccs[idx] = bestx*2-1
						#print("putting in {}, after {}".format(bestx*2-1, xreccs[idx]))
					print("last x {}".format(xreccs[-1]))
					xreccss.append(xreccs)
					lengths.append(len(df.index))
					#regret = sp.exp(bestys) - 1 # check against functionfiles if transform is used. (for QAOA probably)
					regret = bestys
					truey = regret #+ globalymin[count//numRuns]
					regrets.append(regret)
					trueys.append(truey)
					count += 1
			else:
				for f in sorted(glob.glob(fullpath+'/*.csv')):
					df = pd.read_csv(f, sep=', ')#, usecols=range(0,16))
					lengths.append(len(df.index))

					xreccs = df.loc[:, 'rx0':'truey at xrecc'].iloc[:, 0:-1].values
					
					ymins = []
					for i in range(0,250):
						if (i >= len(xreccs)):
							ymins.append(func([(x+1)/2 for x in xreccs[-1]])) #for qaoa
							#ymins.append(func(xreccs[-1]))
						else:
							ymins.append(func([(x+1)/2 for x in xreccs[i]])) #for qaoa
							#ymins.append(func(xreccs[i]))
					#sepValues = pd.DataFrame(df['truey at xrecc'].str.split(',').to_list(), columns=['truey at xrecc', 'taq'])
					#ymins = sepValues['truey at xrecc'].values.astype(float)
					#regret = sp.exp(ymins) - 1 # check against functionfiles if transform is used. (for QAOA probably)
					regret = ymins
					truey = regret #+ globalymin[count//numRuns]
					regrets.append(regret)
					trueys.append(truey)
					xreccss.append(xreccs)
					count += 1
			
			manyRegrets.append(regrets)
			manyLengths.append(lengths)
			manyTrueys.append(trueys)
			manyXreccs.append(xreccss)
	print(plotOrder)
	### Regret plotting
	plotRegret(manyRegrets, manyLengths, plotOrder)
	plotSuccessProb(manyXreccs, manyLengths, plotOrder) # for qaoa only

	### Dataprofile plotting

	# r = 1e-1
	# numIterations = 250
	# plotDataprofie(numProblems, numRuns, r, numIterations, manyTrueys, globalymin)
	# r = 1e-2
	# plotDataprofie(numProblems, numRuns, r, numIterations, manyTrueys, globalymin)

