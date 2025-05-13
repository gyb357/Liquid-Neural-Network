# 📑 Introduction

## Liquid Neural Network Overview
Liquid Neural Networks (LNN) are a class of models designed to capture dynamic, continuous-time relationships in time-series data, inspired by the computational principles of biological neurons. Unlike traditional RNNs or LSTMs, LNNs offer strong dynamic adaptability with fewer parameters and enhanced interpretability through differential equation-based formulations.

## Purpose of the Project
This project aims to apply LNN models to financial time series forecasting, particularly the prediction of KOSPI and KOSDAQ indices. The goal is to evaluate LNNs against traditional RNN-based methods in terms of both predictive accuracy and their ability to generalize across time scales with minimal overfitting.


*****


# 🔍 Architecture Overview

## LTC (Liquid Time-Constant Network)
The LTC model simulates the internal dynamics of a neuron using an input-dependent time constant. The dynamics are governed by a first-order differential equation:

$$
\frac{d\mathbf{x}(t)}{dt} = -\left[ \frac{1}{\tau} + f(\mathbf{x}(t), \mathbf{I}(t), t, \theta) \right] \mathbf{x}(t) + f(\mathbf{x}(t), \mathbf{I}(t), t, \theta)\mathbf{A}.
$$

where
 - τ is the learnable time constant,
 - f is a small MLP,
 - A is a bias term.

 The output is computed via integration of these dynamics over time, allowing the model to adapt to different temporal resolutions.


*****

