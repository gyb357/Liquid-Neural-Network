# 📑 Introduction

## Liquid Neural Network Overview
This repository implements Liquid Neural Network (LNN) variants, including LTC (Liquid Time-Constant Network) and CfC (Closed-form Continuous-time Network), using PyTorch. These models are designed to capture dynamic, continuous-time relationships in time-series data, inspired by the computational principles of biological neurons.

Unlike traditional RNNs or LSTMs, LNNs offer strong dynamic adaptability with fewer parameters and enhanced interpretability through differential equation-based formulations.

## Purpose of the Project
This project aims to apply LNN models to financial time series forecasting, particularly the prediction of KOSPI and KOSDAQ indices. The goal is to evaluate LNNs against traditional RNN-based methods in terms of both predictive accuracy and their ability to generalize across time scales with minimal overfitting.


*****


# 🔍 Architecture Overview

## LTC (Liquid Time-Constant Network)
The LTC model simulates the internal dynamics of a neuron using an input-dependent time constant. The dynamics are governed by a first-order differential equation:

$$
\frac{d\mathbf{x}(t)}{dt} = -\left[ \frac{1}{\tau} + f(\mathbf{x}(t), \mathbf{I}(t), t, \theta) \right] \mathbf{x}(t) + f(\mathbf{x}(t), \mathbf{I}(t), t, \theta)\mathbf{A}.
$$

Where \( \tau \) is the learnable time constant, \( A \) is a bias term, and \( f \) is a small MLP. The output is computed via integration of these dynamics over time, allowing the model to adapt to different temporal resolutions.

## CfC (Closed-form Continuous-time Network)
CfC extends the LTC by solving the ODE analytically. The closed-form solution is used to update the hidden state directly without numerical integration:

$$
\mathbf{x}(t) = \mathbf{B} \odot e^{ - \left[ w_{\tau} + f(\mathbf{x}, \mathbf{I}; \theta) \right] t } \odot f(-\mathbf{x}, -\mathbf{I}; \theta) + \mathbf{A},
$$

Where \( \sigma \) is a parameterized gate function controlling temporal decay. CfC models are much faster and more stable during training while preserving the expressiveness of LTC.

> [!Note]
> For more detailed background, see the original papers:
> - [LTC (2020)](https://arxiv.org/abs/2006.04439)
> - [CfC (2022)](https://arxiv.org/abs/2106.13898)


*****

