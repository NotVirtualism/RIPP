"""
this is a template. I am currently just moving respective code chunks to their own files for making work later.
"""
# Animation
if anim_bool:
    fig3 = plt.figure(figsize=(6, 6))
    ax = fig3.add_subplot(111)


    def update(it):
        ax.cla()
        # fig.clf() #clear the figure
        # ax = fig3.add_subplot(111)

        ax.plot(pos[0:it, 0], pos[0:it, 1])
        ax.plot(pos[it, 0], pos[it, 1], 'ro')
        n = 10
        x, y = np.mgrid[-n:n, -n:n]
        u, v = -y, -x
        ax.quiver(x, y, u, v, 1, alpha=1.)
        ax.set_xlim(-n, n)
        ax.set_ylim(-n, n)


    ani = animation.FuncAnimation(fig3, update, interval=1, frames=nt)
    ani.save('sample.mp4', writer="Pillow")  # save the animation as a gif file