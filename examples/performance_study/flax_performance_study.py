
import os
# xla_gpu_autotune_level ironically makes things much slower 
# on top of removing determinism
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import timeit
import numpy as np
import statistics


class Model(nnx.Module):
  def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dmid, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.dropout = nnx.Dropout(0.2, rngs=rngs)
    self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x):
    x = nnx.relu(self.dropout(self.bn(self.linear(x))))
    return self.linear_out(x)

def train_step(model, optimizer, metrics, x, y):
  def loss_fn(model):
    y_pred = model(x)  # call methods directly
    return ((y_pred - y) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)  # in-place updates
  metrics.update(loss=loss)

  return loss

key = jax.random.PRNGKey(0)
x = jax.random.uniform(key, shape=(32, 2), minval=0.0, maxval=1.0)
y = jnp.zeros((32, 3))
batch_size = 4

## the first approach using Asynchronous dispatch. as in https://flax.readthedocs.io/en/latest/guides/performance.html#asynchronous-dispatch
train_step_1 = nnx.jit(train_step)
def train_function_1(model, optimizer, metrics, num_epochs):
    rng, _ = jax.random.split(jax.random.PRNGKey(0))
    for _ in range(num_epochs):
        rng, sub = jax.random.split(rng)
        batch_orders = jax.random.permutation(sub, len(x)).reshape(-1, batch_size)
        for batch_ids in batch_orders:
            loss = train_step_1(model, optimizer, metrics, x[batch_ids], y[batch_ids])
    return loss

## the second approach using nnx.cached_partial to cache the graph node traversals as in https://flax.readthedocs.io/en/latest/guides/performance.html#caching-graph-node-traversals
train_step_2 = nnx.jit(train_step)
def train_function_2(model, optimizer, metrics, num_epochs):
    cached_train_step = nnx.cached_partial(train_step_2, model, optimizer, metrics)
    rng, _ = jax.random.split(jax.random.PRNGKey(0))
    for _ in range(num_epochs):
        rng, sub = jax.random.split(rng)
        batch_orders = jax.random.permutation(sub, len(x)).reshape(-1, batch_size)
        for batch_ids in batch_orders:
            loss = cached_train_step(x[batch_ids], y[batch_ids])
    return loss

## the third approach use Functional training loop as in https://flax.readthedocs.io/en/latest/guides/performance.html#functional-training-loop
def jax_train_step(graphdef, state, x, y):
  # merge at the beginning of the function
  model, optimizer, metrics = nnx.merge(graphdef, state)

  def loss_fn(model):
    y_pred = model(x)  # call methods directly
    return ((y_pred - y) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)
  metrics.update(loss=loss)

  state = nnx.state((model, optimizer, metrics))
  return loss, state

train_step_3 = jax.jit(jax_train_step)
def train_function_3(model, optimizer, metrics, num_epochs):
    graphdef, state = nnx.split((model, optimizer, metrics))  
    rng, _ = jax.random.split(jax.random.PRNGKey(0))  
    for _ in range(num_epochs):
        rng, sub = jax.random.split(rng)
        batch_orders = jax.random.permutation(sub, len(x)).reshape(-1, batch_size)
        for batch_ids in batch_orders:        
            loss, state = train_step_3(graphdef, state, x[batch_ids], y[batch_ids])
    # update objects after training
    nnx.update((model, optimizer, metrics), state)
    return loss

## the forth approach using nnx.scan to roll out see discussions in https://github.com/google/flax/issues/4045
@nnx.jit
def optimized_nnx_train_step(state, data):
  model, optimizer, metrics = state  
  def loss_fn(model):
    y_pred = model(x)
    return ((y_pred - y) ** 2).mean()
  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)
  metrics.update(loss=loss)
  return loss

train_step_4 = nnx.jit(lambda st, dat, idxs: nnx.scan(lambda st, idxs: (st, optimized_nnx_train_step(st, jax.tree.map(lambda a: a[idxs], dat))))(st, idxs))  
def train_function_4(model, optimizer, metrics, num_epochs): 
    rng, _ = jax.random.split(jax.random.PRNGKey(0))  
    state = (model, optimizer, metrics)     
    for _ in range(num_epochs):
        rng, sub = jax.random.split(rng)
        order = jax.random.permutation(sub, len(x)).reshape(-1, batch_size)
        _, losses = train_step_4(state, (x,y), order)
    return jnp.mean(losses)    


# Benchmark
n_repeat = 5   # How many times to repeat the timing
n_number = 5    # How many times to run the function in each repeat
def test_train_function(function_name,train_function):
    model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'),)    
    jit_time = timeit.repeat(lambda: train_function(model, optimizer, metrics,1).block_until_ready,repeat=1, number=1)    
    times = timeit.repeat(lambda: train_function(model, optimizer, metrics,10).block_until_ready,repeat=n_repeat, number=n_number)
    normalized_times = np.array(times) / n_number
    print(f"{function_name} statistics:")
    print(f"jit time: {jit_time[0]:.6f} sec")
    print(f"Per-run mean time: {statistics.mean(normalized_times):.6f} sec")
    print(f"Per-run std dev  : {statistics.stdev(normalized_times):.6f} sec")       

test_train_function("train_function_1",train_function_1)
test_train_function("train_function_2",train_function_2)
test_train_function("train_function_3",train_function_3)
test_train_function("train_function_4",train_function_4)
 
