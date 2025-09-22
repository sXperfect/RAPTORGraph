import numpy as np
import matplotlib.pyplot as plt
import ot  # Install with: pip install ot

# Generate two 2D circular distributions (different sizes)
np.random.seed(42)
def generate_circle(n_points, center, radius, noise=0.1):
    """Generate points on a circle with random noise"""
    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = radius + noise * np.random.randn(n_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack((x, y))

# Smaller circle (100 points) at (0, 0)
circle1 = generate_circle(100, (0, 0), 1.0, noise=0.15)
# Larger circle (150 points) at (3, 3)
circle2 = generate_circle(150, (3, 3), 1.5, noise=0.2)

# Compute optimal transport
M = ot.dist(circle1, circle2, metric='euclidean')
a = np.ones(circle1.shape[0]) / circle1.shape[0]
b = np.ones(circle2.shape[0]) / circle2.shape[0]
gamma = ot.emd(a, b, M)

# Extract mapping
mapping = []
for i in range(circle1.shape[0]):
    j = np.argmax(gamma[i])
    mapping.append((i, j))

# Create the visualization
plt.figure(figsize=(10, 8))

# Plot both circles
plt.scatter(circle1[:, 0], circle1[:, 1], c='red', s=50, alpha=0.6, label='Circle 1 (100 points)')
plt.scatter(circle2[:, 0], circle2[:, 1], c='blue', s=30, alpha=0.3, label='Circle 2 (150 points)')

# Plot mapping lines
for i, j in mapping:
    plt.plot([circle1[i, 0], circle2[j, 0]], 
             [circle1[i, 1], circle2[j, 1]], 
             'k-', alpha=0.2, linewidth=0.5)

# Add circle outlines for clarity
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.3, label='Circle 1 outline')
plt.plot(3 + 1.5*np.cos(theta), 3 + 1.5*np.sin(theta), 'b--', alpha=0.3, label='Circle 2 outline')

plt.title('Optimal Transport Mapping Between 2D Circular Distributions', fontsize=14)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.legend(loc='upper right')
plt.axis('equal')
plt.grid(True, alpha=0.3)

# Save the figure as 'ot.png' (high resolution)
plt.savefig('ot.png', bbox_inches='tight', dpi=300)

# Optional: Show the plot (comment out in headless environments)
plt.show()

# Print first 5 mappings
print("First 5 mappings (source index → target index):")
for i in range(5):
    print(f"Point {i} in Circle 1 → Point {mapping[i][1]} in Circle 2")