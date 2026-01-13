import argparse
import numpy as np
import time
from numba import cuda
import math


@cuda.jit
def compute_forces_cuda(pos, vel, force, dt):
    i = cuda.grid(1)
    n = pos.shape[0]
    if i < n:
        fx = 0.0
        fy = 0.0
        fz = 0.0
        for j in range(n):
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist_sq = dx * dx + dy * dy + dz * dz + 1e-9
                dist = math.sqrt(dist_sq)
                f = 1.0 / (dist_sq * dist)
                fx += f * dx
                fy += f * dy
                fz += f * dz

        force[i, 0] = fx
        force[i, 1] = fy
        force[i, 2] = fz

        # Update position and velocity (simple Euler)
        vel[i, 0] += fx * dt
        vel[i, 1] += fy * dt
        vel[i, 2] += fz * dt

        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--iter", type=int, default=100)
    args = parser.parse_args()

    n = args.n
    pos = np.random.rand(n, 3).astype(np.float32)
    vel = np.random.rand(n, 3).astype(np.float32)
    force = np.zeros((n, 3), dtype=np.float32)

    # Move to device
    d_pos = cuda.to_device(pos)
    d_vel = cuda.to_device(vel)
    d_force = cuda.to_device(force)

    threadsperblock = 128
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock

    start = time.time()
    for i in range(args.iter):
        compute_forces_cuda[blockspergrid, threadsperblock](d_pos, d_vel, d_force, 0.01)
        cuda.synchronize()
        if i % 10 == 0:
            print(f"Iteration {i}")

    end = time.time()
    print(f"Total time: {end - start:.4f}s")


if __name__ == "__main__":
    main()
