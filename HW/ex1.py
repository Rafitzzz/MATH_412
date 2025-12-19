import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---- PINBALL LOSS FUNCTION EXERCISE ----

def pinball_loss(z, tau):
    """
    Calculates the pinball loss for a given z and tau.
    
    Args:
        z (np.ndarray): The input value (error term a-y).
        tau (float): The quantile parameter (between 0 and 1).
        
    Returns:
        np.ndarray: The calculated pinball loss.
    """
    # This function expects 'tau' to be a single number.
    return np.where(z > 0, (1 - tau) * z, -tau * z)

# 1. Generate data for the x-axis (z)
z = np.linspace(-10, 10, 400)

# 2. Choose the different tau values to plot
taus = [0.2, 0.5, 0.8]

# 3. Create the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))

# 4. Plot the function for each tau using a loop 
for tau in taus:
    loss = pinball_loss(z, tau)
    ax.plot(z, loss, label=f'τ = {tau}', linewidth=2.5)

# 5. Customize the plot for clarity
ax.set_title('Pinball Loss Function $h_τ(z)$', fontsize=16)
ax.set_xlabel('z  (Prediction Error)', fontsize=12)
ax.set_ylabel('Loss Value', fontsize=12)
ax.legend(title="Quantile (τ)", fontsize=10)

# Center the axes at (0,0) for a cleaner look
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.show()


# ---- POLYNOMIAL REGRESSION EXERCISE ----

# Import data
train_data = pd.read_csv("ex1_simreg1_train.csv")
train_data[["X", "Y"]] = train_data['X;"Y"'].str.split(";", expand=True)

test_data = pd.read_csv("ex1_simreg1_test.csv")
test_data[["X", "Y"]] = test_data['X;"Y"'].str.split(";", expand=True)

# --- Convert new columns to a numeric type ---
train_data[["X", "Y"]] = train_data[["X", "Y"]].astype(float)
test_data[["X", "Y"]] = test_data[["X", "Y"]].astype(float)

# Split data into (x,y)
x_train = train_data["X"].values
y_train = train_data["Y"].values

x_test = test_data["X"].values
y_test = test_data["Y"].values

# Modify x_train matrix by adding y-intercept term (column of ones)
x_train_intercept_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_test_intercept_test = np.c_[np.ones(x_test.shape[0]), x_test]

# Calculate estimators (using train data)
beta_hat = np.linalg.inv(x_train_intercept_train.T @ x_train_intercept_train) @ x_train_intercept_train.T @ y_train

# Calculate risk estimation (train data)
fitted_values_train = x_train_intercept_train @ beta_hat
risk_train = (1/len(y_train)) * np.sum((y_train - fitted_values_train) ** 2)

# Calculate risk for test data
fitted_values_test = x_test_intercept_test @ beta_hat
risk_test =  (1/len(y_test)) * np.sum((y_test - fitted_values_test) ** 2)

print(risk_test<risk_train)

# Add x^2 into our feature matrix
