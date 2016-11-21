#! /usr/bin/env python3

"""
Goal of the attributes of this module cumulatively is to numerically 
approximate the coupled ODE's that describe the motion of a ball in a 
"sombrero" potential with both a driving and damping force. 
"""

import numpy as np
import matplotlib.pyplot as plt

# yPrime and xPrime together make up the problem specified by CW12
def yPrime(t,x,y,d=.25,m=1,w=1,F=.18):
    """
    Returns yPrime at a specified time
    x = just value
    y = just value
    """
    yPrime = float(-d*y + x - x**3 + F*np.cos(w*t))/float(m)
    return(yPrime)

def xPrime(t,x,y):
    """
    Returns xPrime at a specified time, for specified value of y
    """
    return(y)

def coupledStep(wP,w,t,dt):
    """
    Carries out one step of the Runga-Kutta 4th order approximation 
    for the coupled set of ODE's given by w(t,x,y)
    Args:
    - wP: the function to be approximated (wP = dw/dt, 
        w = [x(t,x,y), y(t,x,y)])
    - w: initial values x_i, y_i
    - t: initial value of t_i
    - dt: the time step
    """
    w[0] = float(w[0])
    w[1] = float(w[1])

    def K1(t,w):
        return(dt*wP(t,w[0],w[1]))

    def K2(t,w):
        return(dt*wP(t+.5*dt, w[0]+.5*K1(t,w)[0], w[1]+.5*K1(t,w)[1]))

    def K3(t,w):
        return(dt*wP(t+.5*dt, w[0]+.5*K2(t,w)[0], w[1]+.5*K2(t,w)[1]))

    def K4(t,w):
        return(dt*wP(t+dt, w[0]+K3(t,w)[0], w[1]+K3(t,w)[1]))

    wNext = w + (K1(t,w) + 2*K2(t,w) + 2*K3(t,w) + K4(t,w))/6.

    return(wNext)

def test_coupledStep():
    """
    Tests a single step of RK4 for the coupled equations:
    u'(t) = v(t)
    v'(t) = -u(t)
    u(0) = 1, v(0) = 0
    against the analytic solution:
    u(t) = cos(t)

    """
    def one(t,x,y):
        return(y)
    def two(t,x,y):
        return(-x)
    def wP1(t,x,y):
        return(np.array([one(t,x,y),two(t,x,y)]))
    w = np.array([1,0])
    maybe = coupledStep(wP1,w,0,.01)
    assert(abs(maybe[0]-np.cos(.01))<=.001 and abs(maybe[1]+np.sin(.01))<=.001)

def coupled(xP,yP,x0,y0,tmin,tmax,dt=.001):
    """
    Iterates the function coupledStep over the range given.
    - xP: x'(t,x,y) = f(t,x,y)
    - yP: y'(t,x,y) = g(t,x,y)
    - x0, y0: initial values of x() and y()
    - tmin, tmax: range over which to evaluate
    - dt: time step, initialized to .001
    """
    def wP(t,x,y):
        return(np.array([xP(t,x,y),yP(t,x,y)]))

    N = int(float(tmax-tmin)/dt + 1)
    ti = np.linspace(tmin,tmax,N)
    xi = np.zeros_like(ti)
    yi = np.zeros_like(ti)
    wi = np.transpose(np.array([xi,yi]))
    wi[0][0] = x0
    wi[0][1] = y0
    for k in range(N-1):
        wi[k+1] = coupledStep(wP,wi[k],ti[k],dt)
    return(ti,wi)

def test_coupled():
    """
    Tests using the functions:
    u'(t) = v(t)
    v'(t) = -u(t)
    u(0) = 1, v(0) = 0
    against the analytic solution:
    u(t) = cos(t)
    """
    def one(t,x,y):
        return(y)
    def two(t,x,y):
        return(-x)
    ti = np.linspace(0,5,5./.001 + 1)
    xi = np.cos(ti)
    maybe = coupled(one,two,1,0,0,5)[1]
    print(maybe)
    #print(maybe,xi)
    for i in [1,200,1000,2000]:
        if abs(maybe[i][0]-xi[i])>=.001:
            assert(False)
    assert(True)

def plotProb1(x0,y0,tmin,tmax,F=.18,d=.25):
    # yPrime and xPrime together make up the problem specified by CW12
    def yPrime(t,x,y,d=.25,m=1,w=1):
        """
        Returns yPrime at a specified time
        x = just value
        y = just value
        """
        yPrime = float(-d*y + x - x**3 + F*np.cos(w*t))/float(m)
        return(yPrime)

    def xPrime(t,x,y):
        """
        Returns xPrime at a specified time, for specified value of y
        """
        return(y)

    l = coupled(xPrime,yPrime,x0,y0,tmin,tmax,dt=.005)
    ti = l[0]
    xiyi = np.transpose(l[1])
    plt.plot(ti,xiyi[0])
    plt.show()

    plt.plot(xiyi[0],xiyi[1])
    plt.show()

def plotPoincare(x0,y0,tmin,tmax,F=.18):
    l2 = coupled(xPrime,yPrime,x0,y0,tmin,tmax,np.pi/100)
    ti = l2[0]
    xiyi = np.transpose(l2[1])
    xi = []
    yi = []
    for i in range(int(tmax/(2*np.pi))):
        xi.append(xiyi[0][200*i])
        yi.append(xiyi[1][200*i])
    plt.plot(xi,yi,'.')
    plt.show()

