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

#gpbo.core.debugoutput['path']='dbout/exponential3'

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
n = 250


#define a simple 2d objective in x which also varies with respect to the environmental variable
def f(x,**ev):
    # Scale axes, 
    z = [xi for xi in x]
    sum = 0
    for i in range(0,D):
        sum += z[i]**2
    y = -sp.exp(-0.5*sum)
    #y = sp.log(y -(-1) + 1)
    # fixed cost
    c = 1.
    # noise
    n = sp.random.normal() * sp.sqrt(s)
    # we want to check the noiseless value when evaluating performance
    if 'cheattrue' in ev.keys():
        if ev['cheattrue']:
            n = 0
    print('f inputs x:{} ev:{} outputs y:{} (n:{}) c:{}'.format(x, ev, y + n, n, c))
    return y + n, c, dict()


timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#arguments to generate default config are objective function, dimensionality,number of initialization points, number of steps, noise variance, result directory and result filename
C = gpbo.core.config.switchdefault(f, D, 10, n, s, 'results', str(D)+'Dexponential'+timestamp+'.csv')

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
print('True value for noisy functions')
f(out[0],cheattrue=True)[0]

