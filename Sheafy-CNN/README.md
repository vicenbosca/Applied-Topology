# Sheafy CNN

A PyTorch implementation of Convolutional Neural Networks using Sheaf Theory principles. This is a work in progress.

## Overview

This library provides an experimental implementation of CNN layers based on sheaf theory concepts. The main components are:
- `NodestoEdges`: Transforms pixel information to edge information
- `EdgestoInter`: Transforms edge information to intersection information
- `SheafyConvBlock`: Combines both transformations
- `SheafyNet`: Complete neural network using these blocks

## Requirements
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PIL

## Current Status

This is an early implementation being tested on MNIST data. The code is under active development and testing.

## Files
- `SCNN.py`: Main library implementation
- `SCNNnp.ipynb`: Development and testing notebook

## Contact

Vicente Gonzalez Bosca  
Email: vicenteg@sas.upenn.edu