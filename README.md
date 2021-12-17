# Loox
[![PyPI version](https://badge.fury.io/py/loox.svg)](https://pypi.org/project/loox/) [![Documentation Status](https://readthedocs.org/projects/loox/badge/?version=latest)](https://loox.readthedocs.io/en/latest/?badge=latest)

Loox is an image data augmentation framework supporting efficiently-composable transformations
with support for JAX function transformations.
Its strengths are efficient execution of complex augmentation pipelines and batched data augmentation on the GPU/TPU via the use of [`jax.jit`](jax:jax-jit) and `jax.vmap`.

In existing data augmentation frameworks,
each transformation is executed separately,
leading to many unnecessary memory reads and writes that could be avoided.
In contrast, Loox tries its best to fuse transformations together,
so that these data-intensive operations are be minimized.

## Getting Started

Loox aims to implement an API similar to that of [Albumentations](https://albumentations.ai).
An augmentation pipeline is defined as a sequence of transformations,
which are then randomly applied to the input images.

```python
import jax
import loox

transform = loox.Chain(
  loox.RandomCrop(256, 256),
  loox.HorizontalFlip(),
  loox.Rotate(),
)

image = ...

rng = jax.random.PRNGKey(27)

transformed_image = transform(rng, image)
```

## Batch-wise Augmentation on the GPU

Leveraging the JAX infrastructure,
it is possible to greatly accelerate data augmentation by using Just-in-Time compilation (`jax.jit`),
which can execute the code on the GPU, as well as batched augmentation (`jax.vmap`).

### Augmenting a single image on the GPU
```python
transformed_image = jax.jit(transform)(rng, image)
```

### Augmenting an entire batch of images on the GPU
```python
sub_rngs = jax.random.split(rng, images.shape[0])
transformed_images = jax.jit(jax.vmap(transform))(sub_rng, images)
```

