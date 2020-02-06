import gpbo
import scipy as sp

#dimensionality
D=2
#noise variance
s=0.
#number of step to take
n=100

#define a simple 2d objective in x which also varies with respect to the environmental variable
def f(x,**ev):
    #y=-sp.cos(x[0])-sp.cos(x[1])+2
    # Scale axes
    y = pow((-1.275*pow(x[0]/sp.pi,2) + 5*x[0]/sp.pi + x[1] - 6), 2) + (10 - 5/(4*sp.pi))*sp.cos(x[0]) + 10
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

#arguments to generate default config are objective function, dimensionality,number of initialization points, number of steps, noise variance, result directory and result filename
C=gpbo.core.config.switchdefault(f,D,10,n,s,'results','branin.csv')
#set the target global regret
C.choosepara['regretswitch']=1e-2
out = gpbo.search(C)
print (out)