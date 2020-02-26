import gpbo
import scipy as sp
from datetime import datetime

#dimensionality
D=6
#noise variance
s=0.
#number of step to take
n=400

#define a simple 2d objective in x which also varies with respect to the environmental variable
def f(x,**ev):
    # Scale axes, [0,1]^6
    alpha = [1, 1.2, 3., 3.2]

    A = [[10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]]

    P = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]

    y = 0
    for i in range(0,4):
        sum = 0
        for j in range(0,6):
            sum = sum -(A[i][j]*pow((x[j]*1/2+1/2)-P[i][j],2))
        y = y - alpha[i] * sp.exp(sum)
    y = sp.log(y - (-3.3223680114155138112) + 1)
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
C=gpbo.core.config.switchdefault(f,D,10,n,s,'results','6dhartmann'+timestamp+'.csv')
#set the target global regret
C.choosepara['regretswitch']=1e-2
out = gpbo.search(C)
print (out)