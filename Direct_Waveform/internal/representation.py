"""Direct 900-sample E(t)/Ec genotype; no amplitude/slew/endpoint constraints.

Waveform-level genetic operators: Gaussian spikes, step mutations, segment
replacement, one- and two-point hard crossover, and blended crossover.
"""
import numpy as np
from scipy.signal import find_peaks
DIM=DRIVE_SAMPLES=900

def fields(x,n=900):
    x=np.asarray(x,float)
    single=x.ndim==1
    if single:x=x[None,:]
    if x.ndim!=2 or x.shape[1]!=900:raise ValueError("Requires 900 native samples")
    if int(n)<2:raise ValueError("n must be >=2")
    if int(n)==900:out=x
    else:
        t=np.linspace(0,1,900);u=np.linspace(0,1,int(n))
        out=np.array([np.interp(u,t,row) for row in x])
    return out[0] if single else out

def complexity(x,n=4097):
    w=np.atleast_2d(fields(x,n))
    z={key:np.zeros(len(w),int) for key in ("turns","raw_turns","turns_1pct","turns_5pct")}
    for i,row in enumerate(w):
        span=float(np.ptp(row))
        if span>0 and np.isfinite(span):
            for key,p in [("raw_turns",0),("turns_1pct",.01),("turns",.02),("turns_5pct",.05)]:
                z[key][i]=len(find_peaks(row,prominence=p*span)[0])+len(find_peaks(-row,prominence=p*span)[0])
    z["peak_ec"]=np.max(np.abs(w),axis=1)
    z["rms_ec"]=np.sqrt(np.mean(w*w,axis=1))
    return z

def _amplitude(rng,parent=None):
    if parent is None:scale=np.exp(rng.normal(np.log(25.),1.))
    else:scale=max(1.,float(np.sqrt(np.mean(np.asarray(parent)**2))))*np.exp(rng.normal(0.,1.))
    return float(rng.choice([-1.,1.])*scale)

def initial_design(seed,n=64):
    if not isinstance(n,(int,np.integer)) or n<2:raise ValueError("Need >=2 initial waveforms")
    rng=np.random.default_rng(int(seed));t=np.linspace(0,1,900)
    gauss=lambda center,width:np.exp(-.5*((t-center)/width)**2)
    waves=[np.zeros(900),30*gauss(.5,.11),-30*gauss(.35,.07),
           np.where((t>=.2)&(t<.7),20.,-10.),
           40*t-20.,45*np.sin(6*np.pi*t),
           np.interp(t,[0,.13,.29,.53,.76,1],[0,25,-30,0,40,-12]),
           np.where(t<.5,35.,-35.),
           25*np.sin(4*np.pi*t)+16*gauss(.76,.04)]
    while len(waves)<n:
        w=np.zeros(900)
        for _ in range(int(rng.integers(1,6))):
            op=int(rng.integers(4));amp=_amplitude(rng)
            if op==0:
                w+=amp*gauss(rng.uniform(0,1),np.exp(rng.uniform(np.log(.003),np.log(.22))))
            elif op==1:
                a,b=sorted(rng.choice(900,2,replace=False));w[int(a):int(b)+1]+=amp
            elif op==2:
                k=int(rng.integers(2,9))
                knots=np.r_[0.,np.sort(rng.uniform(0,1,k-1)),1.]
                w+=np.interp(t,knots,rng.normal(0,abs(amp),size=k+1))
            else:w+=amp*np.sin(2*np.pi*rng.integers(1,17)*t+rng.uniform(0,2*np.pi))
        waves.append(w)
    return np.asarray(waves[:n],float)

def gaussian_spike(parent,rng):
    child=np.array(parent,float,copy=True)
    center=rng.uniform(0,899);width=np.exp(rng.uniform(np.log(1.),np.log(250.)))
    child+=_amplitude(rng,parent)*np.exp(-.5*((np.arange(900)-center)/width)**2)
    return child,"gaussian_spike"

def step_mutation(parent,rng):
    child=np.array(parent,float,copy=True)
    a,b=sorted(rng.choice(900,2,replace=False))
    child[int(a):int(b)+1]+=_amplitude(rng,parent)
    return child,"step_mutation"

def segment_replacement(parent,rng):
    child=np.array(parent,float,copy=True)
    a,b=sorted(rng.choice(900,2,replace=False));a=int(a);b=int(b)+1
    u=np.linspace(0,1,b-a);amp=_amplitude(rng,parent)
    choice=int(rng.integers(4))
    if choice==0:piece=np.full(b-a,amp)
    elif choice==1:piece=np.linspace(amp,-amp,b-a)
    elif choice==2:piece=amp*np.sin(2*np.pi*rng.integers(1,7)*u+rng.uniform(0,2*np.pi))
    else:piece=amp*np.exp(-.5*((u-rng.uniform())/rng.uniform(.02,.35))**2)
    child[a:b]=piece
    return child,"segment_replacement"

def mutate(parent,rng):
    return (gaussian_spike,step_mutation,segment_replacement)[int(rng.integers(3))](parent,rng)

def one_point_crossover(a,b,rng):
    k=int(rng.integers(1,900))
    return np.r_[a[:k],b[k:]],"one_point_crossover"

def two_point_crossover(a,b,rng):
    l,r=sorted(rng.choice(np.arange(1,900),2,replace=False))
    child=np.array(a,copy=True);child[int(l):int(r)]=b[int(l):int(r)]
    return child,"two_point_crossover"

def smooth_crossover(a,b,rng):
    k=int(rng.integers(1,900));w=int(rng.integers(2,180))
    l=max(0,k-w//2);r=min(899,k+w//2)
    child=np.array(a,copy=True)
    alpha=np.linspace(0.,1.,r-l+1)
    child[l:r+1]=(1-alpha)*a[l:r+1]+alpha*b[l:r+1]
    child[r+1:]=b[r+1:]
    return child,"smooth_crossover"

def proposed_child(method,a,b,rng):
    if method=="Mutation-Only-GA":return mutate(a,rng)
    if rng.random()<.7:
        if method=="Hard-Crossover-GA":
            result,op=(one_point_crossover if rng.random()<.5 else two_point_crossover)(a,b,rng)
        elif method=="Smooth-Crossover-GA":result,op=smooth_crossover(a,b,rng)
        else:raise ValueError("Unknown GA variant")
        if rng.random()<.3:
            result,mutation=mutate(result,rng);op=op+"+"+mutation
        return result,op
    return mutate(a,rng)
