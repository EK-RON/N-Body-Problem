import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---- Parametreler (NBodySimulation.java ile aynı mantık) ----
N = 30
STEPS = 150
G = 6.67430e-2   # gorsellestirme icin buyutulmus G (gercekci hareket icin)
DT = 0.05

rng = np.random.default_rng(42)
pos = rng.uniform(-10, 10, size=(N, 2)).astype(np.float64)
vel = np.zeros((N, 2))
mass = rng.uniform(1, 10, size=N)

history = np.zeros((STEPS, N, 2))

for s in range(STEPS):
    # her parcacik icin diger tum parcaciklarin kuvvetini hesapla (O(N^2))
    diff = pos[None, :, :] - pos[:, None, :]      # (N,N,2)
    distSqr = (diff**2).sum(axis=2) + 1e-1        # softening
    invDist3 = distSqr ** -1.5
    np.fill_diagonal(invDist3, 0)
    f = G * mass[None, :, None] * invDist3[:, :, None] * diff
    force = f.sum(axis=1)

    vel += force * DT
    pos += vel * DT
    history[s] = pos

# ---- Animasyon ----
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_facecolor("black")
ax.set_title("N-Body Simulasyonu (N=30)", color="white")
fig.patch.set_facecolor("black")
ax.tick_params(colors="white")

scat = ax.scatter(pos[:, 0], pos[:, 1], s=mass*3, c="cyan")
trail_lines = [ax.plot([], [], lw=0.5, alpha=0.4, color="cyan")[0] for _ in range(N)]

def update(frame):
    scat.set_offsets(history[frame])
    start = max(0, frame - 20)
    for i, line in enumerate(trail_lines):
        line.set_data(history[start:frame+1, i, 0], history[start:frame+1, i, 1])
    return [scat] + trail_lines

ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=40, blit=True)
ani.save("/home/claude/nbody/nbody_simulation.gif", writer="pillow", fps=25)
print("done")
