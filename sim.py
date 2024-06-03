"""
TO DO:
- Calculate and plot the gyrofrequency
- Calculate and plot particle energy over time
- Play with Euler scheme (1st order) to try it out
- Reformat particles to be objects to prepare when particles are more than 1
"""

import math
from math import sqrt as sqrt
import numpy as np
import matplotlib.pyplot as plt

nop = 1                                 # number of particles
q = 1.0                                 # particle charge
m = 1.0                                 # particle mass
c = 1.0                                 # speed of light in a vacuum (1.0 for normalized)
dt = 0.01                               # time step
nt = 1000                               # number of time steps

E = np.array([1.0, 1.0, 0.0])           # electric field
B = np.array([0.0, 1.0, 1.0])           # magnetic field
x = np.array([0.0, 0.0, 0.0])           # initial particle position
v = np.array([1.0, 0.0, 0.0])           # initial particle velocity

pos = np.zeros((nt, 3))                 # array of positions
vel = np.zeros((nt, 3))                 # array of velocities
en = np.zeros(nt)                       # array of particle energies

"""
Methods:
"boris" - boris leapfrog scheme
"euler" - euler / first order scheme
"""
method = "boris"                        

for step in range(nt):
    # Store current position and velocity
    pos[step] = x
    vel[step] = v
    en[step] = 0.5 * m * np.dot(v, v)
    
    if method == "boris":
        # Half-step for velocity.
        v_minus = v + (q / m) * E * (dt / 2)

        # Accounting for magnetic field.
        T = (dt / 2) * (q * B / (m * c))
        S = 2 * T / (1 + np.dot(T, T))
        v_prime = v_minus + np.cross(v_minus, T)
        v_plus = v_minus + np.cross(v_prime, S)

        # Full-step for velocity.
        v = v_plus + (q * E / m) * (dt / 2)
    
    if method == "euler":
        v += dt * (q / m) * (E + np.cross(v / c, B))

    # Posiiton change
    x = x + v * dt

# Calculating cyclotron frequency and gyroradius if magnetic field is static uniform.
B_mag = np.linalg.norm(B)
if B[0] == B_mag or B[1] == B_mag or B[2] == B_mag:
    cyc_f = q * B_mag / m
    v_perp = 0
    if(B[0] != 0): # gets the perpendicular values of v to B
        v_perp = sqrt(v[1]**2 + v[2]**2)
    elif(B[1] != 0):
        v_perp = sqrt(v[0]**2 + v[2]**2)
    else:
        v_perp = sqrt(v[0]**2 + v[1]**2)
    gyroradius = v_perp / cyc_f
    gyrofreq = cyc_f / (2*np.pi)
    print("Cyclotron frequency: {}\nGyroradius: {}\nGyrofrequency: {}".format(cyc_f, gyroradius, gyrofreq))


# Plotting
fig = plt.figure(figsize=(10, 8))

# 3D Plot
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2])
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

fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(121)
ax.plot(np.arange(0, dt * nt, dt), en)
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.set_title('Particle Energy')
ax.set_box_aspect(1)

ax2 = fig2.add_subplot(122)
ax2.plot(np.arange(0, dt * nt, dt), pos[:, 0])
ax2.set_xlabel('Time')
ax2.set_ylabel('X position')
ax2.set_title('Position over Time')
ax2.set_box_aspect(1)
plt.tight_layout()
plt.show()
