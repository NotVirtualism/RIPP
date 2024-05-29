import math
from math import sqrt as sqrt
import numpy as np
import matplotlib.pyplot as plt

nop=1 #number of particles
q=1     #particle charge
m=1    #particle mass
B=[0,0,1] #magnetic field
E=[0,0,0] #electric field
dt=0.01 #time step
nt=1000 #number of time steps -> actual time = nt*dt=10
x0=[0,0,0] #initial particle position
v0=[1,0,0] #initial particle velocity
c=299792458 #speed of light in a vacuum

pos = [x0]
vel = [v0]
xhalf = x0 + (np.asanyarray(v0) * (dt / 2))
pos.append(xhalf)
for n in range(1000):
    T = (dt/2)*(q/(m*c))*np.asanyarray(B)
    S = T/(1 + (np.linalg.norm(T) ** 2))
    vmin = vel[n] + ((dt/2) * q/m * np.asanyarray(E))
    vprime = vmin + np.cross(vmin, T)
    vplus = vmin + np.cross(vprime, S)
    vnext = vplus + ((dt/2)*(q/m * np.asanyarray(E)))
    vel.append(vnext)
    xnext = pos[n] + (dt * vnext)
    pos.append(xnext)

xpos = [p[0] for p in pos]
ypos = [p[1] for p in pos]
plt.style.use('_mpl-gallery')

plt.plot(xpos, ypos)
plt.show()