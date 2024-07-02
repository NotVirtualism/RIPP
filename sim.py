"""
TODO:
- Split features among various .py files for sake of avoiding bloat. (DONE)
- Export and import data as .cvs tables or .txt files.
- Further research and implement functionality with plasmapy and astropy, which may require rethinking and rewriting a
  lot of base code.
- - - It may be easier to just not use astropy and plasmapy.
"""
from math import sqrt as sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ML
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.fft import fft

# Plasma
from plasmapy.particles import DimensionlessParticle
from plasmapy import formulary
from plasmapy.simulation.particle_integrators import boris_push
import astropy.units as u
import astropy.constants as const


def track_particle_boris(E0, B0, x0, v0, particle, dt, nt):
    # Initialize arrays to save trajectory and velocity
    x = np.zeros((nt+1, 3))*u.m
    v = np.zeros((nt+1, 3))*u.m/u.s

    x[0,:] = x0
    v[0,:] = v0

    for i in range(nt):
        _x = x[i, :].to(u.m).value
        _v = v[i, :].to(u.m/u.s).value
        boris_push(_x,
                   _v,
                   np.array([B0.to(u.T).value,]),
                   np.array([E0.to(u.V/u.m).value,]),
                   particle.charge,
                   particle.mass,
                   dt)
        x[i+1,:] = _x * u.m
        v[i+1,:] = _v*u.m/u.s

    return x,v


nop = 2         # number of particles
particles = []  # particle array
x_s = []        # initial positions
v_s = []        # initial velocities
pos_s = []      # particle positions
vel_s = []      # particle velocities
c = 1.0         # speed of light in a vacuum (1.0 for normalized)
dt = 0.01       # time step
nt = 1000       # number of time steps

E = np.array([0.0, 0.0, 0.0])  # electric field
B = np.array([0.0, 0.0, 5.0])  # magnetic field
for p in range(nop):
    particles.append(DimensionlessParticle(mass=1.0, charge=1.0))

# Particle 1
x_s.append(np.array([0.0, 0.0, 0.0]))  # initial particle position
v_s.append(np.array([1.0, 0.0, 1.0]))  # initial particle velocity

# Particle 2
x_s.append(np.array([0.0, 0.0, 0.0]))   # initial particle position
v_s.append(np.array([1.0, 0.0, -1.0]))  # initial particle velocity

# Various booleans for turning on and off features of the simulation for different tests.
# plot_bool -- Plots various aspects of the simulations results. these being:
#            - 3D motion of the particle
#            - Orthogonal views of the 3D motion
#            - Particle energy over time
#            - Changes in X, Y, and Z axis over time
plot_bool = True

# anim_bool -- Creates an animation of the plotting of the X-Y axis cross-section of the plot.
#              Depending on the number of timesteps both in the simulation and the animation, this
#              can get VERY time- and resource-heavy.
anim_bool = False

# ml_bool  --  Applies machine learning algorithms to segments of the timesteps and particles to group them.
#              Specifically, this is done in the form of:
#            - Fast Fourier Transform (FFT) Preprocessing
#            - Principal Component Analysis (PCA)
#            - K-Means Clustering
ml_bool = False

"""
Methods:
"boris" - boris leapfrog scheme
"euler" - euler / first order scheme
"""
method = "boris"
for i in range(nop):
    pos, vel = track_particle_boris(E, B, x_s[i], v_s[i], particles[i], dt, nt)
    pos_s.append(pos)
    vel_s.append(vel)
'''
for step in range(nt):
    # Store current position and velocity
    pos[step] = x
    vel[step] = v
    en[step] = 0.5 * particle.charge * np.dot(v, v)

    B[0] = -x[1]
    B[1] = -x[0]
    B[2] = 0.0  # Creates the cusp environment for this simulation.

    if method == "boris":
        # Half-step for velocity.
        v_minus = v + (particle.charge / particle.mass) * E * (dt / 2)

        # Accounting for magnetic field.
        T = (dt / 2) * (particle.charge * B / (particle.mass * c))
        S = 2 * T / (1 + np.dot(T, T))
        v_prime = v_minus + np.cross(v_minus, T)
        v_plus = v_minus + np.cross(v_prime, S)

        # Full-step for velocity.
        v = v_plus + (particle.charge * E / particle.mass) * (dt / 2)

    if method == "euler":
        v += dt * (particle.charge / particle.mass) * (E + np.cross(v / c, B))

    # Position change
    x = x + v * dt
'''

'''
# Calculating cyclotron frequency and gyroradius if magnetic field is static uniform.
B_mag = np.linalg.norm(B)
if (B[0] == B_mag or B[1] == B_mag or B[2] == B_mag) and np.linalg.norm(E) == 0:
    cyc_f = particle.charge * B_mag / particle.charge
    v_perp = 0
    if B[0] != 0:  # gets the perpendicular values of v to B
        v_perp = sqrt(v[1] ** 2 + v[2] ** 2)
    elif B[1] != 0:
        v_perp = sqrt(v[0] ** 2 + v[2] ** 2)
    else:
        v_perp = sqrt(v[0] ** 2 + v[1] ** 2)
    gyroradius = v_perp / cyc_f
    gyrofreq = cyc_f / (2 * np.pi)
    print("Cyclotron frequency: {}\nGyroradius: {}\nGyrofrequency: {}".format(cyc_f, gyroradius, gyrofreq))
'''

# Basic Figure Plotting
if plot_bool:
    fig = plt.figure(figsize=(10, 8))

    # 3D Plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')

    # XY Plot
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY')
    ax2.set_box_aspect(1)

    # XZ Plot
    ax3 = fig.add_subplot(223)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ')
    ax3.set_box_aspect(1)

    # YZ Plot
    ax4 = fig.add_subplot(224)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ')
    ax4.set_box_aspect(1)

    for i in range(nop):
        ax1.plot(pos_s[i][:, 0], pos_s[i][:, 1], pos_s[i][:, 2])
        ax2.plot(pos_s[i][:, 0], pos_s[i][:, 1])
        ax3.plot(pos_s[i][:, 0], pos_s[i][:, 2])
        ax4.plot(pos_s[i][:, 1], pos_s[i][:, 2])
    plt.tight_layout()
    plt.show()
'''
    # Energy and Axis plots
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(121)
    ax.plot(np.arange(0, dt * nt, dt), en)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Particle Energy')
    ax.set_box_aspect(1)

    ax2 = fig2.add_subplot(122)
    ax2.plot(np.arange(0, (dt * nt)+1, dt), pos[:, 0], label='x position')
    ax2.plot(np.arange(0, (dt * nt)+1, dt), pos[:, 1], label='y position', color='red')
    ax2.plot(np.arange(0, (dt * nt)+1, dt), pos[:, 2], label='z position', color='green')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position')
    ax2.set_title('Position over Time')
    ax2.set_box_aspect(1)
    plt.legend()
    plt.tight_layout()
    plt.show()
'''

