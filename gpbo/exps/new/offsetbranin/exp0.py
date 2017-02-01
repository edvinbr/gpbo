import gpbo
import numpy as np
import scipy as sp
#mode='run'

mode=['run','plot'][0]
nreps=4
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--offset', dest='offset', action='store', default=0,type=int)

args = parser.parse_args()

vers=[2,3][1]
D=2

s=1e-6
lb = sp.array([-1.,-1.])
ub = sp.array([1.,1.])

from objective import f

from objective import truemin
all2confs=[]
all3confs=[]
rpath='results0'
#-----------------------
#eimle
C=gpbo.core.config.eimledefault(f,D,12,s,rpath,'null.csv')
C.aqpara['nrandinit']=10
C.stoppara = {'tmax': 60*60*2}
C.stopfn = gpbo.core.optimize.totaltstopfn
C.reccfn = gpbo.core.reccomenders.argminrecc
all2confs.append(['eimle',C])

#pesbs----------------------------
C=gpbo.core.config.pesbsdefault(f,D,50,s,rpath,'null.csv')
C.stoppara = {'tmax': 60 * 60 * 2}
C.stopfn = gpbo.core.optimize.totaltstopfn
C.aqpara['overhead']='last'
C.aqpara['nrandinit']=20
C.reccfn=gpbo.core.reccomenders.gphinasargminrecc

all2confs.append(['pesbs_argmin',C])

#pesbs----------------------------
C=gpbo.core.config.pesbsdefault(f,D,50,s,rpath,'null.csv')
C.stoppara = {'tmax': 60 * 60 * 2}
C.stopfn = gpbo.core.optimize.totaltstopfn
C.aqpara['overhead']='last'
C.aqpara['nrandinit']=20
C.reccfn=gpbo.core.reccomenders.gphinasrecc

all2confs.append(['pesbs_postmin',C])

#-----------------
#mtbo
C={'lowtask':4,
   'ninit':20,
   'nsteps':150}

#all3confs.append(['mtbo2',C])

#-----------------
#mtbo
C={'lowtask':16,
   'ninit':20,
   'nsteps':150}

#all3confs.append(['mtbo4',C])

#-----------------
#mtbo
C={'lowtask':64,
   'ninit':20,
   'nsteps':150}

all3confs.append(['mtbo8',C])
#---------------
#fabolas
C={'ninit':30,
   'nsteps':200}
#all3confs.append(['fabmod',C])
#---------------
#fabolas
C={'ninit':30,
   'nsteps':150}
all3confs.append(['fabolas',C])
if mode=='run':
    if vers==2:
        gpbo.runexp(f,lb,ub,rpath,nreps,all2confs,indexoffset=args.offset*nreps)
    else:
        gpbo.runexp(f,lb,ub,rpath,nreps,all3confs,indexoffset=args.offset*nreps)
elif mode=='plot':
    gpbo.plotall(all2confs+all3confs,1,rpath,trueopt=truemin)
else:
    pass
