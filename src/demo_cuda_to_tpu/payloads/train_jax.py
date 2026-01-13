import time
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

print(f"JAX Devices: {jax.devices()}")

class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def create_train_state(rng, input_shape, learning_rate):
    cnn = SimpleCNN()
    params = cnn.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_images, batch_labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_images)
        one_hot = jax.nn.one_hot(batch_labels, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
    # Hyperparameters
    batch_size = 2048
    learning_rate = 0.001
    epochs = 5
    
    # Synthetic Data
    print("Generating synthetic data...")
    key = jax.random.PRNGKey(0)
    # NHWC format for Flax/JAX default
    train_data = jax.random.normal(key, (60000, 28, 28, 1))
    train_labels = jax.random.randint(key, (60000,), 0, 10)
    
    num_batches = 60000 // batch_size

    # Initialize
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, (1, 28, 28, 1), learning_rate)
    
    print("Starting Training...")
    
    # Compilation Warmup
    print("Compiling JAX graph...")
    dummy_x = train_data[0:batch_size]
    dummy_y = train_labels[0:batch_size]
    _ = train_step(state, dummy_x, dummy_y)
    print("Compilation Complete.")

    start_total = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        
        # Simple loop (in real world use jax.lax.scan or data loader)
        for i in range(num_batches):
            batch_images = train_data[i*batch_size : (i+1)*batch_size]
            batch_labels = train_labels[i*batch_size : (i+1)*batch_size]
            state, loss = train_step(state, batch_images, batch_labels)
            running_loss += loss
            
        # Wait for computation to finish for timing
        loss.block_until_ready()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/num_batches:.4f} | Time: {epoch_time:.4f}s")

    total_time = time.time() - start_total
    print(f"Total Training Time: {total_time:.4f}s")

if __name__ == "__main__":
    main()
