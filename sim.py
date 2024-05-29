import math
from math import sqrt as sqrt
import numpy as np
import matplotlib.pyplot as plt

nop = 1                                 # number of particles
q = 1.0                                 # particle charge
m = 1.0                                 # particle mass
c = 1.0                                 # speed of light in a vacuum (1.0 for normalized)
E = np.array([0.0, 0.0, 0.0])           # electric field
B = np.array([0.0, 0.0, 1.0])           # magnetic field
dt = 0.01                               # time step
nt = 1000                               # number of time steps
x = np.array([0.0, 0.0, 0.0])           # initial particle position
v = np.array([1.0, 0.0, 0.0])           # initial particle velocity

pos = np.zeros((nt, 3))                 # array of positions
vel = np.zeros((nt, 3))                 # array of velocities

for step in range(nt):
    # Store current position and velocity
    pos[step] = x
    vel[step] = v

    # Half-step for velocity.
    v_minus = v + (q / m) * E * (dt / 2)

    # Accounting for magnetic field.
    T = (dt / 2) * (q * B / (m * c))
    S = 2 * T / (1 + np.dot(T, T))
    v_prime = v_minus + np.cross(v_minus, T)
    v_plus = v_minus + np.cross(v_prime, S)

    # Full-step for velocity.
    v = v_plus + (q * E / m) * (dt / 2)
    
    # Full-step for position.
    x = x + v * dt

# Graph only concerned with X and Y values.
xpos = [p[0] for p in pos]
ypos = [p[1] for p in pos]

# Plotting
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(xpos, ypos)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_box_aspect(1) # Graph is 1:1
ax.set_title("Particle Trajectory")
plt.show()