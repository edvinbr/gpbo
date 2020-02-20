import gpbo
import scipy as sp
from datetime import datetime

#dimensionality
D=2
#noise variance
s=0.
#number of step to take
n=400

#define a simple 2d objective in x which also varies with respect to the environmental variable
def f(x,**ev):
    # Scale axes, [-3,3] and [-2,2]
    y = (4 - 2.1*pow(x[0]*3,2) + pow(x[0]*3,4)/3)*pow(x[0]*3,2) + x[0]*3*x[1]*2 + (-4 + 4*pow(x[1]*2,2))*pow(x[1]*2,2)
    #y = (4 - 2.1*pow(x[0],2) + pow(x[0],4)/3)*pow(x[0],2) + x[0]*x[1] + (-4 + 4*pow(x[1],2))*pow(x[1],2)
    y = sp.log(y - (-1.0316) + 1)
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
C=gpbo.core.config.switchdefault(f,D,10,n,s,'results','6humpcamel'+timestamp+'.csv')
#set the target global regret
C.choosepara['regretswitch']=1e-2
out = gpbo.search(C)
print (out)