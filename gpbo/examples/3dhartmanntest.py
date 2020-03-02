import gpbo
import scipy as sp
import numpy as np
from datetime import datetime

#dimensionality
D=3
#noise variance
s=0.
#number of step to take
n=200

#define a simple 2d objective in x which also varies with respect to the environmental variable
def f(x,**ev):
    # Scale axes, [-5,10]^3 in thesis, [0,1]^3 online, [0,1]^3 in objectives.py
    z = [xi*0.5+0.5 for xi in x]
    alpha = [1., 1.2, 3., 3.2]
    A = [[3, 10, 30],
        [0.1, 10, 35],
        [3, 10, 35],
        [0.1, 10, 35]]
    P = [[0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0381, 0.5743, 0.8828]]
    y = 0
    for i in range(0,4):
        sum = 0
        for j in range(0,3):
            #y = y - (alpha[i] * -(A[i][j]*pow((x[j]*15/2+5/2)-P[i][j],2)))
            #y = y - (alpha[i] * -(A[i][j]*pow((x[j]*1/2+1/2)-P[i][j],2)))
            sum = sum -(A[i][j]*pow((z[j])-P[i][j],2))
        y = y - alpha[i] * sp.exp(sum)
    y = sp.log(y - (-3.862779787332660444) + 1)
    #fixed cost
    c=1.
    #noise
    n = sp.random.normal()*s
    #we want to check the noiseless value when evaluating performance
    if 'cheattrue' in ev.keys():
        if ev['cheattrue']:
            n=0
    print('f inputs x:{} ev:{} outputs y:{} (n:{}) c:{}'.format(x,ev,y+n,n,c))
    return y+n,c,dict()

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#arguments to generate default config are objective function, dimensionality,number of initialization points, number of steps, noise variance, result directory and result filename
C=gpbo.core.config.switchdefault(f,D,10,n,s,'results','3dhartmann'+timestamp+'.csv')
#set the target global regret
C.choosepara['regretswitch']=1e-2
out = gpbo.search(C)
print (out)