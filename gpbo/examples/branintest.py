import gpbo
import scipy as sp
from datetime import datetime

#dimensionality
D=2
#noise variance
s=0.
#number of step to take
n=200

#define a simple 2d objective in x which also varies with respect to the environmental variable
def f(x,**ev):
    # Scale axes, [0,1]^2 in paper, [-5,10]^2 in objectives.py, [-5,10]x[0,15] online
    z = [xi*7.5+2.5 for xi in x]
    y = pow((-1.275*pow((z[0])/sp.pi,2) + 5*(z[0])/sp.pi + (z[1]) - 6), 2) + (10 - 5/(4*sp.pi))*sp.cos((z[0])) + 10
    y = sp.log(y - 0.39788735772973816 + 1)
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
C=gpbo.core.config.switchdefault(f,D,10,n,s,'results','branin'+timestamp+'.csv')
#set the target global regret
C.choosepara['regretswitch']=1e-2
out = gpbo.search(C)
print (out)