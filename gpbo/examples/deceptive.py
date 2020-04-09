import gpbo
import scipy as sp
import numpy as np
import os
from datetime import datetime
import sys

try:
    suffix = sys.argv[1]
except IndexError:
    suffix = ''
gpbo.core.debugoutput['pathsuffix'] = suffix

#gpbo.core.debugoutput['path']='dbout/deceptive'

#gpbo.core.debugoutput['adaptive'] = True
#gpbo.core.debugoutput['acqfn2d'] = True
#gpbo.core.debugoutput['support'] = True
#gpbo.core.debugoutput['drawlap'] = True
#gpbo.core.debugoutput['tmp'] = True

# dimensionality
D = 2
# noise variance
s = 0.
# number of step to take
n = 250*(D+1)


#define a simple 2d objective in x which also varies with respect to the environmental variable
def f(x,**ev):
    # Scale axes, 
    z = [xi*0.5 + 0.5 for xi in x]
    sum1 = 0
    for i in range(0,D):
        alpha = (i+1)/(D+1)
        if(0 <= z[i] and z[i] < 4/5*alpha):
            sum1 += -z[i]/alpha + 4/5
        elif(4/5*alpha < z[i] and z[i] <= alpha):
            sum1 += 5*z[i]/alpha - 4
        elif(alpha < z[i] and z[i] <= (1+4*alpha)/5):
            sum1 += 5*(z[i]-alpha)/(alpha-1) + 1
        elif((1+4*alpha)/5 < z[i] and z[i] <= 1):
            sum1 += (z[i]-1)/(1-alpha) + 4/5
    y = -(1/D*sum1)**2
    #y = sp.log(y -(-1) + 1)
    # fixed cost
    c = 1.
    # noise
    n = sp.random.normal() * s
    # we want to check the noiseless value when evaluating performance
    if 'cheattrue' in ev.keys():
        if ev['cheattrue']:
            n = 0
    print('f inputs x:{} ev:{} outputs y:{} (n:{}) c:{}'.format(x, ev, y + n, n, c))
    return y + n, c, dict()


timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#arguments to generate default config are objective function, dimensionality,number of initialization points, number of steps, noise variance, result directory and result filename
C=gpbo.core.config.switchdefault(f,D,10,n,s,'results',str(D)+'Ddeceptive'+timestamp+'.csv')
#C = gpbo.core.config.switchetest(f, D, 10, n, s, 'results', 'deceptive.csv')

# set the target global regret
C.choosepara['regretswitch'] = 1e-2
C.choosepara['pvetol'] = 1e-2
C.aqpara[1]['tol']=None#1e-6


print("before search")
# Add namesuffix as argument to use different savefiles
initdata = False
if initdata:
    C.choosepara = (np.load(os.path.join(gpbo.core.debugoutput['path'], "choosepara"+gpbo.core.debugoutput['pathsuffix']+".npy"),allow_pickle=True)).tolist()
    #C = (np.load(os.path.join(gpbo.core.debugoutput['path'], "optconfig.npy"),allow_pickle=True)).tolist()
out = gpbo.search(C, initdata)
print(out)

