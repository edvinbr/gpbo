import gpbo
import scipy as sp
import numpy as np
import os
from datetime import datetime
import sys
#from GPyOpt.methods import BayesianOptimization
import GPyOpt

try:
    suffix = sys.argv[1]
except IndexError:
    suffix = ''
gpbo.core.debugoutput['pathsuffix'] = suffix

#gpbo.core.debugoutput['path']='dbout/schwefel1'

#gpbo.core.debugoutput['adaptive'] = True
#gpbo.core.debugoutput['acqfn2d'] = True
#gpbo.core.debugoutput['support'] = True
#gpbo.core.debugoutput['drawlap'] = True
#gpbo.core.debugoutput['tmp'] = True

# dimensionality
D = 2
# noise variance
s = 0.0
# number of step to take
n = 250


#define a simple 2d objective in x which also varies with respect to the environmental variable
def f(x,**ev):
    # Scale axes, 
    z = [xi*2.048 for xi in x]
    z = z[0]
    sum1 = 0
    for i in range(0,D-1):
        sum1 += 100*(z[i+1] - z[i]**2)**2 + (1-z[i])**2
    y = sum1
    y = sp.log(y -(0) + 1)
    # fixed cost
    c = 1.
    # noise
    n = sp.random.normal() * s
    # we want to check the noiseless value when evaluating performance
    if 'cheattrue' in ev.keys():
        if ev['cheattrue']:
            n = 0
    print('f inputs x:{} ev:{} outputs y:{} (n:{}) c:{}'.format(x, ev, y + n, n, c))
    return y + n

bounds2d = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (-1,1)}]

optimizer = GPyOpt.methods.BayesianOptimization(f, domain=bounds2d)

optimizer.run_optimization(max_iter=250,verbosity = True, report_file = '2Drosenbrock-test-report2.txt',evaluations_file='2DRosenbrock-test-evals2.txt',models_file='2Drosenbrock-test-models2.txt')

result = optimizer.x_opt

print(result)


#timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#arguments to generate default config are objective function, dimensionality,number of initialization points, number of steps, noise variance, result directory and result filename
#C=gpbo.core.config.switchdefault(f,D,10,n,s,'results', '2DRosenbrock-e2-nonoise-pes-run3.csv')

# set the target global regret
#C.choosepara['regretswitch'] = 1e-2
#C.choosepara['pvetol'] = 1e-2
#C.aqpara[1]['tol']=None#1e-6


#print("before search")
# Add namesuffix as argument to use different savefiles
#initdata = False
#if initdata:
#    C.choosepara = (np.load(os.path.join(gpbo.core.debugoutput['path'], "choosepara"+gpbo.core.debugoutput['pathsuffix']+".npy"),allow_pickle=True)).tolist()
    #C = (np.load(os.path.join(gpbo.core.debugoutput['path'], "optconfig.npy"),allow_pickle=True)).tolist()
#out = gpbo.search(C, initdata)
#print(out)

