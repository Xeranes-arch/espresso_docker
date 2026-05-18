def progress():

    import matplotlib.pyplot as plt
    import numpy as np
    import time

    # --- setup interactive plot ---
    plt.ion()
    fig, ax = plt.subplots()
    scat = ax.scatter([], [])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    fig.show()

    # --- mock simulation loop ---
    for step in range(5000):
        # here: integrate MD step (placeholder)
        # system.integrator.run(1)

        # fake data
        x = np.random.randn(100)
        y = np.random.randn(100)

        if step % 100 == 0:
            # update plot data
            scat.set_offsets(np.c_[x, y])

            # redraw without blocking loop
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # (your loop continues immediately — no plt.pause)
        time.sleep(0.001)  # simulate work per step

    print("done")
