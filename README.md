# astro-ml-simulations

Machine learning applied to astrophysical problems:  
1. Predicting orbital stability in binary star systems  
2. Estimating the Hubble constant from simulated data  

---

## Overview  

This repository demonstrates how machine learning can be combined with astrophysical modeling.  
It contains two independent scripts:  

1. **Binary Star Orbital Stability** – Simulates randomly generated planets orbiting a binary star system and uses **logistic regression** to classify stable vs. unstable orbits.  
2. **Hubble Constant Estimation** – Generates synthetic galaxy velocity–distance data and applies a **neural network regressor** to recover the Hubble constant.  

These projects serve both as demonstrations of astrophysical principles and as examples of applied machine learning in scientific research.  

---

## Features  

- **Binary Star Orbital Stability**  
  - N-body gravitational dynamics simulation of a binary star system  
  - Random generation of planet masses, orbital distances, and velocities  
  - Stability testing with numerical integration  
  - Machine learning classification (logistic regression) of stable vs. unstable planets  
  - 3D visualization of orbits  

- **Hubble Constant Estimation**  
  - Synthetic galaxy data generation with added noise  
  - Neural network regression (TensorFlow/Keras) to fit Hubble’s law  
  - Estimation of the Hubble constant from model predictions  
  - Visualization of galaxy velocity–distance relationship  

---

## Technology Stack  

- **Python**: NumPy, Matplotlib, scikit-learn, TensorFlow/Keras  
- **Numerical Methods**: N-body dynamics, stability analysis  
- **Machine Learning**: Logistic regression, neural networks  
- **Visualization**: 3D Matplotlib animations and scatter plots  

---

## Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/BenMontalbano/astro-ml-simulations.git
cd astro-ml-simulations
pip install numpy matplotlib scikit-learn tensorflow
