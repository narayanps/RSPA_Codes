import numpy
import pylab
import random
import scipy.io as sio

#Author : Narayan Subramaniyam / ELT / BioMediTech / TUT. August 2015
# use the script below to generate realizations from a Lorenz system
def generate(data_length, odes, state, parameters):
    data = numpy.zeros([state.shape[0], data_length],dtype='float32')

    for i in xrange(50000):
        state = rk4(odes, state, parameters)

    for i in xrange(data_length):
        state = rk4(odes, state, parameters)
        data[:, i] = state

    return data


def rk4(odes, state, parameters, dt=0.05):
    k1 = dt * odes(state, parameters)
    k2 = dt * odes(state + 0.5 * k1, parameters)
    k3 = dt * odes(state + 0.5 * k2, parameters)
    k4 = dt * odes(state + k3, parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def lorenz_odes((x1, y1, z1), (sigma,r,beta)):

    x1_dot = sigma*(y1 - x1)
    y1_dot = x1*r - y1 - x1*z1
    z1_dot = x1*y1 - beta*z1
    
    return numpy.array([x1_dot,y1_dot,z1_dot])
    
def lorenz_generate(data_length):
    return generate(data_length, lorenz_odes, \
        numpy.array([random.random(),random.random(),random.random()]), numpy.array([10, 28, 8.0/3.0]))
        
random.seed
numItrs = 100
N=100000
SignalX = numpy.zeros((numItrs,N),dtype='float32')
for i in range(numItrs):   
     data=lorenz_generate(N)
     SignalX[i,:] = data[0,:].copy()
outfile = "Lorenz_NL0.npy"  
numpy.save(outfile,SignalX)
outfile = "Lorenz_NL0.mat"  
matfile = "Lorenz_NL0" 
sio.savemat(outfile, {matfile:SignalX})
