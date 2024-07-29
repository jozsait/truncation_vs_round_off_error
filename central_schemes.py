import numpy as np
import matplotlib.pyplot as plt

# Define the function and its analytical derivative
def func(x):
    return np.cos(3.2 * (x - 0.1))

def true_derivative(x):
    return -3.2 * np.sin(3.2 * (x - 0.1))

# Second-order central difference scheme
def second_order_central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Fourth-order central difference scheme
def fourth_order_central(f, x, h):
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)

# Compute the error
def compute_error(approx, exact,err_type):
    err_type = err_type.lower()
    if err_type == 'l2':
        return np.sqrt(np.sum((approx - exact) ** 2))
    elif err_type == 'l2_mod':
        return np.sqrt(np.sum((approx - exact) ** 2))/len(approx)
    elif err_type == 'rms':
        return np.std(approx - exact)
    elif err_type == 'mean_abs_rel':
        return np.mean(abs( 100*(approx - exact)/exact ))
    else:
        raise ValueError("ERROR TYPE OF -- " + str(err_type).upper() + " -- IS NOT RECOGNISED!!! \n select L2 or L2_MOD or RMS\n")

# Main code
def evaluate_error(dtype):
    h_values = pow(2, np.arange(0,-22,-1,dtype=np.float32) )
    errors_2nd = []
    errors_4th = []

    for h in h_values:
        x = np.arange(-1, 1+h, h, dtype=dtype)
        f = func(x)
        exact_derivative = true_derivative(x)
        
        approx_2nd = np.array([second_order_central(func, xi, h) for xi in x], dtype=dtype)
        approx_4th = np.array([fourth_order_central(func, xi, h) for xi in x], dtype=dtype)
        
        errors_2nd.append(compute_error(approx_2nd, exact_derivative,'mean_abs_rel'))
        errors_4th.append(compute_error(approx_4th, exact_derivative,'mean_abs_rel'))
    
    return h_values, errors_2nd, errors_4th

# Plotting the results
def plot_errors(ax, h_values, errors_2nd, errors_4th, precision_label):
    ax.loglog(h_values, errors_2nd, '-ko', label='Second Order Central Scheme')
    ax.loglog(h_values, errors_4th, '--rs', label='Fourth Order Central Scheme')
    ax.set_xlabel('Grid Spacing (h)')
    ax.set_ylabel('Error')
    ax.set_title(f'{precision_label}')
    ax.legend()
    ax.grid(True)

# Evaluate and plot for different precisions
precisions = {
    'Half Precision (float16)': np.float16,
    'Single Precision (float32)': np.float32,
    'Double Precision (float64)': np.float64
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (precision_label, dtype) in zip(axes, precisions.items()):
    h_values, errors_2nd, errors_4th = evaluate_error(dtype)
    plot_errors(ax, h_values, errors_2nd, errors_4th, precision_label)

plt.tight_layout()
plt.show()
