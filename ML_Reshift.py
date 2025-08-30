import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Author: Ben Montalbano
# Purpose: The purpose of this script is to calculate the hubble constant using data and 
# basic machine learning. More personally, the purpose is to gain experience using machine 
# learning to solve physics problems.

H0_true = 69.8  #Hubble constant in Mpc

#Create Fake Data
def generate_hubble_data(num_samples=500):
    distances = np.random.uniform(10, 1000, num_samples)  # In Mpc
    velocities = H0_true * distances + np.random.normal(0, 1000, num_samples)  
    return distances.reshape(-1, 1), velocities.reshape(-1, 1)

# Build and train regression model
X, y = generate_hubble_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu", input_shape=(1,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)  
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=200, verbose=0)

# Run Model
d_test = np.linspace(10, 1000, 100).reshape(-1, 1)  
v_pred = model.predict(d_test) 

# Compute Hubble Constant
H0_pred, _ = np.polyfit(d_test.flatten(), v_pred.flatten(), 1)
Error=H0_true-H0_pred
print(f"Estimated Hubble Constant: {H0_pred:.2f} km/s/Mpc\nError: {H0_true-H0_pred} km/s/Mpc")


# Plot
plt.scatter(X, y, alpha=0.5, label="Data")
plt.plot(d_test, v_pred, color='red', label="Model Fit")
plt.xlabel("Distance (Mpc)")
plt.ylabel("Velocity (km/s)")
plt.title("Estimating the Hubble Constant with ML")
plt.legend()
plt.show()