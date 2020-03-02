import gpbo
import scipy as sp
import numpy as np
import os
from datetime import datetime


# dimensionality
D = 2
# noise variance
s = 0.
# number of step to take
n = 100


#define a simple 2d objective in x which also varies with respect to the environmental variable
def f(x,**ev):
    # Scale axes, [-5,5]^2, [-0.2, 0.2]^2 in objectives.py
    z = [xi*0.2 for xi in x]
    y = 2*pow(z[0],2) - 1.05*pow(z[0],4) + pow(z[0],6)/6 + z[0]*z[1] + pow(z[1],2)
    y = sp.log(y + 1)
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
#C=gpbo.core.config.switchdefault(f,D,10,n,s,'results','3humpcamel'+timestamp+'.csv')
C = gpbo.core.config.switchdefault(f, D, 10, n, s, 'results', '3humpcamel.csv')
print(C)
# set the target global regret
C.choosepara['regretswitch'] = 1e-2
C.choosepara['pvetol'] = 1e-2

print("before search")
initdata = True
if initdata:
    C.choosepara = (np.load(os.path.join(gpbo.core.debugoutput['path'], "choosepara.npy"),allow_pickle=True)).tolist()
    #C = (np.load(os.path.join(gpbo.core.debugoutput['path'], "optconfig.npy"),allow_pickle=True)).tolist()
out = gpbo.search(C, initdata)
print(out)
