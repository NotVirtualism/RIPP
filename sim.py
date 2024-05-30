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
v = np.array([1.0, 0.0, 1.0])           # initial particle velocity

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

# Plotting
fig = plt.figure(figsize=(10, 8))

# 3D Plot
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(pos[:, 0], pos[:, 1], pos[:,2])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Trajectory')

# XY Plot
ax2 = fig.add_subplot(222)
ax2.plot(pos[:, 0], pos[:, 1])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('XY')
ax2.set_box_aspect(1)

# XZ Plot
ax3 = fig.add_subplot(223)
ax3.plot(pos[:, 0], pos[:, 2])
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.set_title('XZ')
ax3.set_box_aspect(1)

# YZ Plot
ax4 = fig.add_subplot(224)
ax4.plot(pos[:, 1], pos[:, 2])
ax4.set_xlabel('Y')
ax4.set_ylabel('Z')
ax4.set_title('YZ')
ax4.set_box_aspect(1)

plt.tight_layout()
plt.show()
