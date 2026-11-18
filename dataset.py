import numpy as np
import matplotlib.pyplot as plt

def generate_moons(n_samples=300, noise=0.15, seed=42):
    np.random.seed(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    
    # Moon 1
    theta_out = np.linspace(0, np.pi, n_out)
    X_out = np.stack([np.cos(theta_out), np.sin(theta_out)], axis=1) + np.random.normal(0, noise, (n_out, 2))
    y_out = np.zeros((n_out, 1))
    
    # Moon 2
    theta_in = np.linspace(0, np.pi, n_in)
    X_in = np.stack([1 - np.cos(theta_in), 0.5 - np.sin(theta_in)], axis=1) + np.random.normal(0, noise, (n_in, 2))
    y_in = np.ones((n_in, 1))
    
    X = np.vstack([X_out, X_in])
    y = np.vstack([y_out, y_in])
    return X, y

if __name__ == "__main__":
    X, Y = generate_moons()
    print(f"Dataset generated successfully! Shape of X: {X.shape}, Shape of Y: {Y.shape}")
    
    # Visualize Raw Dataset
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=Y.squeeze(), cmap=plt.cm.coolwarm, s=40, edgecolors='k')
    plt.title("Raw Two Moons Dataset (Linearly Inseparable)")
    plt.xlabel("Feature X1")
    plt.ylabel("Feature X2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()