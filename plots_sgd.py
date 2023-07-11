import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function to optimize
def function(x, y):
    return x ** 2 + y ** 2 + 5 * np.sin(x) + 5 * np.sin(y)

# Define the partial derivatives of the function
def partial_derivative_x(x, y):
    return 2 * x + 5 * np.cos(x)

def partial_derivative_y(x, y):
    return 2 * y + 5 * np.cos(y)

# Define the gradient descent function
def gradient_descent_3d(learning_rate, max_iterations):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Gradient Descent 3D')

    current_x = 9  # Initial guess for x
    current_y = 9  # Initial guess for y

    # Draw the scatter points above the surface
    ax.scatter(current_x, current_y, function(current_x, current_y), color='red', label='Initial Guess', s=50, zorder=10)

    x_vals = [current_x]
    y_vals = [current_y]
    z_vals = [function(current_x, current_y)]

    for i in range(max_iterations):
        gradient_x = partial_derivative_x(current_x, current_y)
        gradient_y = partial_derivative_y(current_x, current_y)
        current_x -= learning_rate * gradient_x
        current_y -= learning_rate * gradient_y

        # Draw the scatter points above the surface
        ax.scatter(current_x, current_y, function(current_x, current_y), color='red', alpha=(i + 1) / max_iterations, s=50, zorder=10)

        x_vals.append(current_x)
        y_vals.append(current_y)
        z_vals.append(function(current_x, current_y))

    ax.scatter(current_x, current_y, function(current_x, current_y), color='green', label='Final Result', s=50, zorder=10)
    ax.legend()

    # Draw lines between scatter points
    ax.plot(x_vals, y_vals, z_vals, color='blue', linewidth=2)

    plt.show()

# Run the gradient descent algorithm for 3D plot
gradient_descent_3d(0.1, 50)
