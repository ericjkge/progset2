import matplotlib.pyplot as plt
import math
import numpy as np

def binom(n, k):
    return math.comb(n, k)

def plot_triangle_estimates():
    n = 1024
    probabilities = [0.01, 0.02, 0.03, 0.04, 0.05]

    # Replace with your observed triangle counts
    experimental_values = [168, 1418, 4773, 11912, 22829]

    # Compute expected values
    C = binom(n, 3)
    expected_values = [C * (p ** 3) for p in probabilities]

    # Generate smooth curve for theoretical line
    x_smooth = np.linspace(0.005, 0.06, 300)
    y_smooth = [C * (p ** 3) for p in x_smooth]

    # Plot
    plt.plot(x_smooth, y_smooth, label=r"Theoretical $\binom{1024}{3} p^3$", linestyle='-')
    plt.scatter(probabilities, expected_values, label="Expected", marker='o')
    plt.scatter(probabilities, experimental_values, label="Experimental", marker='o')

    plt.xlabel("Edge Probability (p)")
    plt.ylabel("Number of Triangles")
    plt.title("Theoretical vs Experimental Triangle Counts (n=1024)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_triangle_estimates()
