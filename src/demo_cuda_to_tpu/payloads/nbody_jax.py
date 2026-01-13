import argparse
import jax
import jax.numpy as jnp
import time


def compute_forces(pos):
    # pos: (N, 3)
    # diff: (N, N, 3)
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1) + 1e-9
    dist = jnp.sqrt(dist_sq)
    f_mag = 1.0 / (dist_sq * dist)
    # Remove self-interaction (where dist is small/identity)
    f_mag = jnp.where(jnp.eye(pos.shape[0], dtype=bool), 0.0, f_mag)

    force = jnp.sum(f_mag[..., None] * diff, axis=1)
    return force


@jax.jit
def step_fn(pos, vel, dt=0.01):
    force = compute_forces(pos)
    vel = vel + force * dt
    pos = pos + vel * dt
    return pos, vel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--iter", type=int, default=100)
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    pos = jax.random.uniform(key, (args.n, 3))
    vel = jax.random.uniform(key, (args.n, 3))

    # Compile
    _ = step_fn(pos, vel)
    print("Compilation complete.")

    start = time.time()
    for i in range(args.iter):
        pos, vel = step_fn(pos, vel)
        # Block until ready to measure timing accurately
        pos.block_until_ready()
        if i % 10 == 0:
            print(f"Iteration {i}")

    end = time.time()
    print(f"Total time: {end - start:.4f}s")


if __name__ == "__main__":
    main()
