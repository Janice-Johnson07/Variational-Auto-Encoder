# Variational Autoencoder (VAE) — Fashion-MNIST
Variational Autoencoder (VAE) built with PyTorch, trained on Fashion-MNIST. Implements a convolutional encoder-decoder architecture with KL divergence regularisation. Experiments with multiple hyperparameter configurations.

## Overview

Implementation of a Variational Autoencoder (VAE) trained on the Fashion-MNIST dataset using PyTorch.
The notebook experiments with multiple model configurations and evaluates them across four dimensions:
train/test loss history, latent space distribution, generated image quality, and reconstruction quality.

## What's in the Notebook

- Convolutional encoder and decoder built as reusable PyTorch classes
- Reparameterisation trick for sampling from the latent space
- KL divergence + binary cross-entropy loss (ELBO)
- Experiments across three model configurations (different hyperparameters)
- Latent space visualisation and image generation over a 2D grid
- Automated test suite — best model passes 15/15 tests

## Tech Stack

- Python 3.11
- PyTorch
- torchvision (Fashion-MNIST)
- NumPy, Matplotlib
- Google Colab (GPU: T4)

## Results

Best model achieved:
- Reconstruction loss: 232.8 (baseline untrained ≈ 543)
- KL divergence: 10.34 per image (no posterior collapse)
- Latent space mean std: 0.624

## How to Run

1. Open the notebook in Google Colab or Jupyter
2. Run all cells in order
3. The notebook auto-detects GPU/MPS/CPU — no manual setup needed
