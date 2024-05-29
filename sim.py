import math
from math import sqrt as sqrt
import numpy as np
import matplotlib.pyplot as plt

nop = 1                                 #number of particles
q = 1.0                                 #particle charge
m = 1.0                                 #particle mass
c = 1.0                                 #speed of light in a vacuum for normalized
E = np.array([0.0, 0.0, 0.0])
B = np.array([0.0, 0.0, 1.0])
dt = 0.01                               #time step
nt = 1000                               #number of time steps -> actual time = nt*dt=10
x = np.array([0.0, 0.0, 0.0])           #initial particle position
v = np.array([1.0, 0.0, 0.0])           #initial particle velocity


pos = np.zeros((nt, 3))
vel = np.zeros((nt, 3))

for step in range(1000):
    pos[step] = x
    vel[step] = v

    v_minus = v + (q / m) * E * (dt / 2)

    T = (dt / 2) * (q * B / (m * c))
    S = 2 * T / (1 + np.dot(T, T))
    v_prime = v_minus + np.cross(v_minus, T)
    v_plus = v_minus + np.cross(v_prime, S)

    v = v_plus + (q * E / m) * (dt / 2)
    x = x + v * dt

xpos = [p[0] for p in pos]
ypos = [p[1] for p in pos]

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(xpos, ypos)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_box_aspect(1)
ax.set_title("Particle Trajectory")
plt.show()