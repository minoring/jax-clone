"""A basoc MNIST example using Numpy and JAX.

The primary aim here is simplicity and mininal dependencies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from numpy import random as npr
from absl import app

from jax.api import jit, grad
from jax.examples import datasets
from jax.scipy.misc import logsumexp
import jax.numpy as np


def main(unused_argv):
  layer_sizes = [784, 1024, 1024, 10] # TODO: Revise to standard arch
  param_scale = 0.1
  step_size = 0.001
  num_epochs = 10
  batch_size = 32

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()

  @jit
  def update(params, batch):
    grads = grad(loss)(params, batch)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]
  
  params = init_random_params(param_scale, layer_sizes)
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      params = update(params, next(batches))
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))


if __name__ == "__main__":
  app.run(main)
