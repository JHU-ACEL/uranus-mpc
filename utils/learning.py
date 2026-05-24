import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import optax
import equinox as eqx

import json
import wandb
import pdb

from .models import FFNet
from dynamics.base_dynamics import Dynamics

# =================== Define pure functions for jax operations ==================#

# Define loss functions
def huber_loss(pred_y, y, delta=1.0):
  residual = jnp.abs(pred_y - y)
  loss = jnp.where(residual < delta, 0.5 * residual**2, delta * (residual - 0.5 * delta))
  return loss

def mse_loss(pred_y, y):
  return (pred_y - y)**2

def compute_loss(model, x, y):
  pred_y = eqx.filter_vmap(model)(x)
  #return jnp.mean(huber_loss(pred_y, y))
  return jnp.mean(mse_loss(pred_y, y))

loss_and_grad = eqx.filter_value_and_grad(compute_loss)

def compute_masked_loss(model, x, y, mask):
    pred_y = eqx.filter_vmap(model)(x)
    squared_error = mse_loss(pred_y, y)
    
    masked_error = squared_error * mask[:, None]
    total_loss = jnp.sum(masked_error)   
    total_elems = jnp.sum(mask) * y.shape[1]
    
    # Return the sum and the count
    return total_loss, total_elems
  
compute_loss_eval = eqx.filter_jit(compute_masked_loss)

def evaluate(model, dataloader):
    total_loss = 0.0
    total_elements = 0.0

    for x, y, mask in dataloader:
        batch_loss_sum, batch_elems = compute_loss_eval(model, x, y, mask)
        total_loss += batch_loss_sum
        total_elements += batch_elems
        
    return float(total_loss / total_elements) if total_elements > 0 else 0.0


def eval_dataloader(x, y, mask, batch_size):
    n = x.shape[0]
    for i in range(0, n, batch_size):
        yield x[i:i+batch_size], y[i:i+batch_size], mask[i:i+batch_size]
      
# Infinite dataloader used in training to provide batch on demand
def infinite_dataloader(arrays, batch_size, key):
  n = arrays[0].shape[0]
  assert all(array.shape[0] == n for array in arrays)
  indices = jnp.arange(n)
  full_n = (n // batch_size) * batch_size
  if full_n == 0:
    raise ValueError("batch_size is larger than dataset; reduce batch_size.")

  while True:
    key, subkey = jrandom.split(key)
    perm = jrandom.permutation(subkey, indices)

    for start in range(0, full_n, batch_size):
      batch_idx = perm[start:start + batch_size]
      yield tuple(array[batch_idx] for array in arrays)

def save_model(filename, hyperparams, model):
  with open(filename, "wb") as f:
      hyperparam_str = json.dumps(hyperparams)
      f.write((hyperparam_str + "\n").encode())
      eqx.tree_serialise_leaves(f, model)

def load_model(filename):
  with open(filename, "rb") as f:
      hyperparams = json.loads(f.readline().decode())
      model = FFNet(key=jrandom.key(0), **hyperparams, in_mean= jnp.zeros(hyperparams['input_size']), in_std= jnp.zeros(hyperparams['input_size']), out_mean= jnp.zeros(hyperparams['output_size']), out_std= jnp.zeros(hyperparams['output_size']))
      return eqx.tree_deserialise_leaves(f, model), hyperparams

# @eqx.filter_jit
def adapt_mag_model(model, pcpf_positions, magnetometer_measurements):
  # Based off of (https://docs.kidger.site/equinox/examples/frozen_layer/)
  # hyperparams
  batch_size = 32
  lr = 1e-1
  steps = 200

  # Define dataloader to get random batches
  dataloader = infinite_dataloader((pcpf_positions, magnetometer_measurements), batch_size, jrandom.key(789))

  # Filter to freeze everything except last layer
  filter_spec = jtu.tree_map(lambda _: False, model)
  filter_spec = eqx.tree_at(
      lambda tree: (tree.layers[-1].weight, tree.layers[-1].bias),
      filter_spec,
      replace=(True, True),
  )

  # Optimizaiton setup
  params = eqx.filter(model, eqx.is_array)
  mask_decay = jtu.tree_map(lambda p: eqx.is_array(p) and (p.ndim >= 2), params)
  optim = optax.chain(
          optax.clip_by_global_norm(1.0),
          optax.adamw(
              learning_rate=lr, 
              weight_decay=.001, 
              mask=mask_decay
          ) 
      )
  trainable_params = eqx.filter(model, filter_spec)
  opt_state = optim.init(trainable_params)

  @eqx.filter_jit
  def adapt_step(model, x, y, opt_state):
      @eqx.filter_grad
      def loss(diff_model, static_model, x, y):
          model = eqx.combine(diff_model, static_model)
          pred_y = jax.vmap(model)(x)
          return jnp.mean((y - pred_y) ** 2)
  
      diff_model, static_model = eqx.partition(model, filter_spec)
      grads = loss(diff_model, static_model, x, y)
      params = eqx.filter(diff_model, eqx.is_array)    
      updates, opt_state = optim.update(grads, opt_state, params)
      model = eqx.apply_updates(model, updates)
      return model, opt_state
  
  # Train (fine-tuning)
  for step, (x, y) in zip(range(steps), dataloader):
      model, opt_state = adapt_step(model, x, y, opt_state)
  return model

# =================== Solver class ======================= #
# Class for all learning functions (train, test, etc)
class Trainer():
  """
    Manages the setup, training, and evaluation of a Neural Network for 
    learning system dynamics in a JAX/Equinox environment.

    This class encapsulates data handling (splitting, batching), model 
    initialization (Equinox MLP), optimizer setup (AdamW with warm-up cosine 
    decay scheduler), logging (Weights & Biases), and the training loop 
    itself, primarily aimed at identifying a residual dynamic model.

    Attributes:
        nx (int): Dimension of the state space.
        nu (int): Dimension of the control/input space.
        features (jax.Array): The complete input dataset (e.g., [state, control] pairs).
        targets (jax.Array): The complete target dataset (e.g., next_state - state).
        training_params (dict): Dictionary holding all training hyperparameters (epochs, batch size, learning rate, etc.).
        lr (float): Peak learning rate value.
        weight_decay (float): L2 regularization parameter for AdamW.
        key (jax.Array): The original JAX random key.
        data_key (jax.Array): JAX key used for data randomization (splitting).
        model_key (jax.Array): JAX key used for model initialization.
        train_key (jax.Array): JAX key used for generating training data batches.
        run (Optional[wandb.Run]): Weights & Biases run object for experiment tracking.
        X_train (jax.Array): Training features subset.
        y_train (jax.Array): Training targets subset.
        X_val (jax.Array): Validation features subset.
        y_val (jax.Array): Validation targets subset.
        X_test (jax.Array): Test features subset.
        y_test (jax.Array): Test targets subset.
        lr_scheduler (optax.Schedule): Cosine decay learning rate schedule with warm-up.
        model (Optional[eqx.Module]): The Equinox neural network model (e.g., MLP).
        model_depth (int): Number of hidden layers in the model.
        model_width_size (int): Number of neurons per hidden layer.
    """
  def __init__(self, features: jax.Array, targets: jax.Array, key: jax.Array, n_epochs:int, lr:float):
    self.features = features
    self.targets = targets

    self.training_params = {}
    self.training_params['NUM_EPOCHS'] = n_epochs
    self.training_params['BATCH_SIZE'] = 32
    self.training_params['CHECKPOINT_AFTER'] = int(100)
    self.training_params['SAVEPOINT_AFTER'] = int(100)
    self.training_params['TEST_BATCH_SIZE'] = 32
    self.training_params['FILENAME'] = 'model.eqx'
    self.training_params['RUN_NAME'] = "FFNet-run"
    self.training_params['lr'] = lr
    self.training_params['weight_decay'] = 0.001
    

    # Define learning rate and weight decay
    self.lr = self.training_params['lr']
    self.weight_decay = self.training_params['weight_decay']

    # Generate jax keys for randomization
    self.key, self.data_key, self.model_key, self.train_key = jrandom.split(key, num=4)

    # Set up logging with wandb
    self.run = None

    # Split data into train, test, val and separate into batches
    self._split_and_pad_data()

    # Define number of training steps
    self.training_params['NUM_STEPS'] = (self.X_train.shape[0] // self.training_params['BATCH_SIZE']) * self.training_params['NUM_EPOCHS']

    # Learning rate scheduler
    warmup_epochs = n_epochs // 5
    warmup_steps = warmup_epochs * (self.training_params['NUM_STEPS'] // self.training_params['NUM_EPOCHS']) # 5 epochs of warm-up
    decay_steps = self.training_params['NUM_STEPS'] - warmup_steps
    self.lr_scheduler = optax.schedules.warmup_cosine_decay_schedule(
      init_value = 0.0,
      peak_value = self.lr,
      warmup_steps = warmup_steps,
      decay_steps = decay_steps,
      end_value=0.0,
    )

  def _split_and_pad_data(self, train_ratio=0.8, val_ratio=0.1, key=None):
    if key is None:
      key = self.data_key
      
    # Split data into train, test, val and separate into batches
    n_samples = self.features.shape[0]
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)

    # Create random permutation for shuffling
    indices = jrandom.permutation(key, n_samples) 

    # Split indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Function to pad/mask val and test datasets for equal batch sizes
    def pad_to_batch_size(x, y, batch_size):
      n = x.shape[0]
      remainder = n % batch_size
      if remainder == 0:
          return x, y, jnp.ones(n)
      
      padding_size = batch_size - remainder
      
      # Pad with zeros
      x_padded = jnp.pad(x, ((0, padding_size), (0, 0)))
      y_padded = jnp.pad(y, ((0, padding_size), (0, 0)))
      
      # Create a mask: 1.0 for data, 0.0 for padding
      mask = jnp.concatenate([jnp.ones(n), jnp.zeros(padding_size)])
      return x_padded, y_padded, mask

    # Split data using indices
    self.X_train = self.features[train_idx]
    self.y_train = self.targets[train_idx]
    val_features = self.features[val_idx]
    val_targets = self.targets[val_idx]
    test_features  = self.features[test_idx]
    test_targets = self.targets[test_idx]

    # Use mask and padding for equal batch sizes for val and test
    self.X_val, self.y_val, self.val_mask = pad_to_batch_size(val_features, val_targets, self.training_params['TEST_BATCH_SIZE'])

    self.X_test, self.y_test, self.test_mask = pad_to_batch_size(test_features, test_targets, self.training_params['TEST_BATCH_SIZE'])
    
  def setup_wandb(self, project_name="magnetic-field-learning-project", run_name=None):
    if run_name is None:
      run_name = self.training_params["RUN_NAME"]
      
    self.run = wandb.init(
      project=project_name,
      name=run_name,
      entity="acel",
      job_type="simple-train-loop",
      config=self.training_params)
    
    self.run.config.update({"model_depth":self.model_depth, "model_width": self.model_width_size})

  def setup_network(self, model=FFNet, depth=3, neurons=32, key=None):
    if key is None:
      key = self.model_key

    self.model_depth = depth
    self.model_width_size = neurons

    input_size = self.features.shape[1]
    output_size = self.targets.shape[1]
    hidden_sizes = [neurons] * depth
    ff_shape = [input_size, *hidden_sizes, output_size]

    # Get data for normalization
    in_mean = jnp.mean(self.X_train, axis=0)
    in_std = jnp.std(self.X_train, axis=0) + 1e-6
    out_mean = jnp.mean(self.y_train, axis=0)
    out_std = jnp.std(self.y_train, axis=0) + 1e-6

    # Set hyperparams
    self.hyperparameters = {"input_size":input_size, "output_size":output_size, "width": neurons, "depth": depth}

    # Use custom model (includes normalization)
    self.model = model(key=key, **self.hyperparameters, in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std)

  def train(self, verbose=True):
    # get training params
    NUM_EPOCHS = self.training_params['NUM_EPOCHS']
    NUM_STEPS = self.training_params['NUM_STEPS']
    BATCH_SIZE = self.training_params['BATCH_SIZE']
    TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']
    CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
    SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
    FILENAME  = self.training_params['FILENAME']

    print("Peak LR: ",self.lr)
    print("NUM_STEPS: ", NUM_STEPS)
    print("NUM_EPOCHS: ", NUM_EPOCHS)

    model = self.model
    params = eqx.filter(model, eqx.is_array)

    # Initialize wandb logging
    self.setup_wandb()
    
    # mask true for weights (ndim >= 2) false for biases (ndim ==1)
    mask_decay = jax.tree_util.tree_map(lambda p: eqx.is_array(p) and (p.ndim >= 2), params)

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=self.lr_scheduler, # change to self.lr for no lr schedule
            weight_decay=self.weight_decay, 
            mask=mask_decay
        ) 
    )

    opt_state = opt.init(params)

    train_loader = infinite_dataloader((self.X_train, self.y_train), BATCH_SIZE, key=self.train_key)

    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = loss_and_grad(model, x, y)
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = opt.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    train_losses = []
    val_losses = []
    checkpoint_train_loss = 0.0

    # Training loop
    # TODO: tqdm for progress bar 
    for step, (x,y) in zip(range(NUM_STEPS), train_loader):
      model, opt_state, train_loss = make_step(model, x, y, opt_state)
      checkpoint_train_loss += train_loss

      if ((step+1) % CHECKPOINT_AFTER) == 0 or (step == NUM_STEPS - 1):
        #val_loader = eval_dataloader((self.X_val, self.y_val), TEST_BATCH_SIZE)
        val_loader = eval_dataloader(self.X_val, self.y_val, self.val_mask, TEST_BATCH_SIZE)
        val_loss = evaluate(model, val_loader)
        val_losses.append(float(val_loss))
        avg_train_loss = float(checkpoint_train_loss) / CHECKPOINT_AFTER
        train_losses.append(avg_train_loss)

        # Logging with wandb
        if self.run is not None:
          self.run.log({
            "train/loss": avg_train_loss,
            "val/loss": float(val_loss),
            "params/learning_rate": float(self.lr_scheduler(step)),
            "epoch": step // (NUM_STEPS // NUM_EPOCHS),
            "step": step + 1,
          })

        if verbose:
          print(f"Step {step + 1}/{NUM_STEPS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {float(val_loss):.6f} ")

        checkpoint_train_loss = 0

      if ((step+1) % SAVEPOINT_AFTER) == 0 or (step == NUM_STEPS -1):
        self.model = model
        save_model(FILENAME,self.hyperparameters, model)

    if self.run is not None:
      self.run.finish()
    
    return train_losses, val_losses

  def test(self, model):
    test_loader = eval_dataloader(self.X_test, self.y_test, self.test_mask, self.training_params['TEST_BATCH_SIZE'])
    test_loss = evaluate(model, test_loader)
    # if self.run is not None:
    #     self.run.summary["test_loss"] = float(test_loss)
    return test_loss

  def get_model_output(self, inputs, model=None, input_scalar=None, target_scalar=None):
    if model is None:
        model = self.model
    if input_scalar is None:
        input_scalar = self.input_scalar
    if target_scalar is None:
      target_scalar = self.target_scalar
    def single_input(input):
      norm_input = input_scalar.normalize(input)
      norm_output = model(norm_input)
      return target_scalar.unnormalize(norm_output)
      
    return jax.vmap(single_input)(inputs)

    

  def get_nn_dynamics(self, dt):
    input_scalar = self.input_scalar
    target_scalar = self.target_scalar
    model = self.model
    class NNDynamics(Dynamics):
      """ Class for storing state_dot of learned dynamics """
      def state_dot(self, state, control, t = 0.0, external_param=None):
        input = jnp.concatenate([state, control])
        norm_input = input_scalar.normalize(input)
        
        norm_residual = model(norm_input)
        dx = target_scalar.unnormalize(norm_residual)
  
        state_dot = dx/dt
        return state_dot
    return NNDynamics()
