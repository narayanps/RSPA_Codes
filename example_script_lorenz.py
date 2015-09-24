import sys
import numpy
from pyunicorn.timeseries import RecurrenceNetwork
from pyunicorn.timeseries import RecurrencePlot
import scipy.io as sio
from multiprocessing import Pool
import random

#Author : Narayan Subramaniyam / ELT / BioMediTech / TUT. August 2015
#run the script as python example_script_lorenz.py arg1 arg2
#arg1 = length of data - 200,500,1000,5000 or 10000 or whatever
#arg2 recurrence rate - 0.01,0.02,0.03,0.04 or 0.05 or whatever
# this prog requires pyunicorn to be installed

#function to generate iAAFT surrogates
def UnivariateSurrogates(data_f,MaxIter):
    
    xs=data_f.copy()
    xs.sort() #sorted amplitude stored
    pwx=numpy.abs(numpy.fft.fft(data_f)) # amplitude of fourier transform of orig
    
    data_f.shape = (-1,1)
    xsur = numpy.random.permutation(data_f) #random permutation as starting point
    xsur.shape = (1,-1)
    
    for i in range(MaxIter):
        fftsurx = pwx*numpy.exp(1j*numpy.angle(numpy.fft.fft(xsur)))
        xoutb = numpy.real(numpy.fft.ifft(fftsurx))
        ranks = xoutb.argsort(axis=1)
        xsur[:,ranks] = xs
    return(xsur) 

#run main program to compute RN measures
def run_main_prog(params):
    #start = time.clock()
    (m,numSig) = params # m is index for noise level, numSig is idex for realization
    numSurr = 99
    SignalAndSurr = numpy.zeros((numSurr+1,N),dtype='float32')
    noise = NL[m]*numpy.std(SignalX[numSig,:].copy())*numpy.random.normal(0,1,N)
    SignalAndSurr[0,:] = SignalX[numSig,:].copy()+noise
    for j in range(1,numSurr+1,1):
         SignalAndSurr[j,:] = UnivariateSurrogates(SignalAndSurr[0,:].copy(),120)
    
    T = numpy.zeros((numSurr+1,1),dtype='float32')          
    
    for k in range(0,numSurr+1,1):
        ts = SignalAndSurr[k,:]
        ts.shape = (-1,1)
        psv = RecurrencePlot.embed_time_series(ts,dim=3,tau=3)
        randomVertices = random.sample(xrange(psv.shape[0]), int(sys.argv[1]))
        R = RecurrenceNetwork(psv[randomVertices,:],recurrence_rate=float(sys.argv[2]),silence_level=2)
        T[k] = R.transitivity() # compute network measure for hypothesis testing

    if T[0] > max(T[1:]): #non-parametric testing
         H0 = 1 #null-hypothesis rejected
    else:
         H0 = 0
    #elapsed = (time.clock() - start)
    #print elapsed
    return (H0)
    

#initialize constants and variables

numItrs = 100 #number of simulations
NL=[0,0.1,0.2,0.4,0.6,1] #different noise level

Y =range(6) #variable for different noise level
Z=range(numItrs) #variable for different realizations

fname = 'LorenzNL0.npy' #load data 100 X 100000 = 100 realizations (x-component) with 100000 points
SignalX = numpy.load(fname) #load data into SignalX

N = SignalX.shape[1] #no. of data points
     
if __name__ == '__main__':
    pool = Pool(processes=8)              # start 24 worker processes
    params = [(y,z)  for y in Y for z in Z]
    get_results = pool.map(run_main_prog,params)
    pool.close()
     
#collect results     
outfile = 'H0_N%s'%int(sys.argv[1])+'_RR00%s.mat'%int(float(sys.argv[2])*100)
outname = 'H0_N%s'%int(sys.argv[1])+'_RR00%s'%int(float(sys.argv[2])*100)

sio.savemat(outfile, {outname:get_results})
