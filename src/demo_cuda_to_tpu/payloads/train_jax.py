import time
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

print(f"JAX Devices: {jax.devices()}")

# --- ResNet-50 Implementation in Flax ---
class Bottleneck(nn.Module):
    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x
        out = nn.Conv(self.features, (1, 1), use_bias=False)(x)
        out = nn.BatchNorm(use_running_average=False)(out)
        out = nn.relu(out)
        
        out = nn.Conv(self.features, (3, 3), strides=(self.stride, self.stride), padding=((1, 1), (1, 1)), use_bias=False)(out)
        out = nn.BatchNorm(use_running_average=False)(out)
        out = nn.relu(out)
        
        out = nn.Conv(self.features * 4, (1, 1), use_bias=False)(out)
        out = nn.BatchNorm(use_running_average=False)(out)

        if self.stride != 1 or x.shape[-1] != self.features * 4:
            residual = nn.Conv(self.features * 4, (1, 1), strides=(self.stride, self.stride), use_bias=False)(x)
            residual = nn.BatchNorm(use_running_average=False)(residual)

        return nn.relu(out + residual)

class ResNet50(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding=((3, 3), (3, 3)), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Stages
        for _ in range(3):
            x = Bottleneck(64)(x)
        
        x = Bottleneck(128, stride=2)(x)
        for _ in range(3):
            x = Bottleneck(128)(x)
        
        x = Bottleneck(256, stride=2)(x)
        for _ in range(5):
            x = Bottleneck(256)(x)
        
        x = Bottleneck(512, stride=2)(x)
        for _ in range(2):
            x = Bottleneck(512)(x)

        x = jnp.mean(x, axis=(1, 2)) # Global Average Pool
        x = nn.Dense(1000)(x)
        return x

def create_train_state(rng, input_shape, learning_rate):
    model = ResNet50()
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.sgd(learning_rate, momentum=0.9)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    images, labels = batch
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 1000)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
    # Hyperparameters
    batch_size = 128
    steps = 300
    learning_rate = 0.001
    
    print("Generating synthetic ImageNet data...")
    key = jax.random.PRNGKey(0)
    # Just create one batch and reuse it to stress compute
    inputs = jax.random.normal(key, (batch_size, 224, 224, 3)) # NHWC
    labels = jax.random.randint(key, (batch_size,), 0, 1000)
    
    # Initialize
    print("Initializing ResNet-50...")
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, (1, 224, 224, 3), learning_rate)
    
    print("Compiling JAX graph...")
    start_compile = time.time()
    state, loss = train_step(state, (inputs, labels))
    loss.block_until_ready()
    print(f"Compilation finished in {time.time() - start_compile:.2f}s")

    print("Starting Training Race...")
    start_time = time.time()

    for step in range(1, steps + 1):
        step_start = time.time()
        
        state, loss = train_step(state, (inputs, labels))
        
        # Sync occasionally for display
        if step % 10 == 0:
            loss.block_until_ready()
            elapsed = time.time() - step_start
            # elapsed is mostly just launch time unless blocked, but average will be correct
            img_sec = batch_size / elapsed
            print(f"Step {step}/{steps} | Loss: {loss:.4f} | {img_sec:.1f} img/s")

    # Final sync
    loss.block_until_ready()
    total_time = time.time() - start_time
    throughput = (steps * batch_size) / total_time
    
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Throughput: {throughput:.1f} images/sec")

if __name__ == "__main__":
    main()