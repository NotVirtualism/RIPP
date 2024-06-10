"""
TO DO:
- Simulate particles in cusp magnetic field situations (DONE)
- Animate the particles over time (DONE)
- Apply machine learning algorithms to segments of the timesteps and particles to group them
"""

import math
from math import sqrt as sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython import display
import matplotlib as mpl

nop = 1                                 # number of particles
q = 1.0                                 # particle charge
m = 1.0                                 # particle mass
c = 1.0                                 # speed of light in a vacuum (1.0 for normalized)
dt = 0.01                               # time step
nt = 10000                              # number of time steps

E = np.array([0.0, 0.0, 0.0])           # electric field
B = np.array([0.0, 0.0, 0.0])           # magnetic field
x = np.array([0.0, 0.0, 0.0])           # initial particle position
v = np.array([1.0, 3.0, 0.0])           # initial particle velocity

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
    B[0] = -x[1]     ;B[1] = -x[0] ;B[2] = 0.0
    
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
if (B[0] == B_mag or B[1] == B_mag or B[2] == B_mag) and np.linalg.norm(E) == 0:
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

# Energy and Axis plots
fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(121)
ax.plot(np.arange(0, dt * nt, dt), en)
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.set_title('Particle Energy')
ax.set_box_aspect(1)

ax2 = fig2.add_subplot(122)
ax2.plot(np.arange(0, dt * nt, dt), pos[:, 0], label = 'x position')
ax2.plot(np.arange(0, dt * nt, dt), pos[:, 1], label = 'y position', color = 'red')
ax2.plot(np.arange(0, dt * nt, dt), pos[:, 2], label = 'z position', color = 'green')
ax2.set_xlabel('Time')
ax2.set_ylabel('Position')
ax2.set_title('Position over Time')
ax2.set_box_aspect(1)
plt.legend()
plt.tight_layout()
plt.show()

# Animation
anim_bool = False
if anim_bool:
    fig3 = plt.figure(figsize=(6, 6))
    ax = fig3.add_subplot(111)
    def update(it):
        ax.cla()
        #fig.clf() #clear the figure
        #ax = fig3.add_subplot(111)

        ax.plot(pos[0:it, 0], pos[0:it,1])
        ax.plot(pos[it,0],pos[it,1],'ro')
        n = 10
        x, y = np.mgrid[-n:n, -n:n]
        u, v = -y, -x
        ax.quiver(x, y, u, v, 1, alpha=1.)
        ax.set_xlim(-n, n)
        ax.set_ylim(-n, n)

    ani = animation.FuncAnimation(fig3, update, interval=1, frames = nt)
    ani.save('sample.mp4', writer="Pillow") #save the animation as a gif file
# Machine learning

"""
Apply machine learning algorithms to segments of the timesteps and particles to group them
"""
ml_bool = True
if ml_bool:
    #Segmenting the full algorithm to timestep chunks
    tsc = 100 # Time Step Chunk: Segments will be grouped by x tsc (100 position points per segment by default).
    # MAKE SURE NT % 100 = 0
    segments = np.reshape(pos, (nt//tsc, tsc, 3))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for s in segments:
        ax.plot(s[:, 0], s[:, 1])
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()
