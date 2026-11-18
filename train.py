import numpy as np
import matplotlib.pyplot as plt
from nn_engine import Linear, Sigmoid, MSELoss, Network, SGD
from dataset import generate_moons

def gradient_check(net, loss_fn, x, y_true, h=1e-5):
    print("--- RUNNING NUMERICAL GRADIENT CHECK ---")
    y_pred = net.forward(x)
    loss = loss_fn.forward(y_pred, y_true)
    loss_grad = loss_fn.backward()
    net.backward(loss_grad)

    for layer_idx, layer in enumerate(net.layers):
        if not hasattr(layer, 'W'):
            continue
        for param_name in ['W', 'b']:
            param = getattr(layer, param_name)
            grad_bp = getattr(layer, f'd{param_name}')
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            
            while not it.finished:
                idx = it.multi_index
                orig = param[idx]
                
                param[idx] = orig + h
                loss_high = loss_fn.forward(net.forward(x), y_true)
                
                param[idx] = orig - h
                loss_low = loss_fn.forward(net.forward(x), y_true)
                
                param[idx] = orig
                grad_num = (loss_high - loss_low) / (2 * h)
                gbp = grad_bp[idx]
                
                rel_error = abs(gbp - grad_num) / (max(abs(gbp), abs(grad_num)) + 1e-8)
                assert rel_error < 1e-6, f"Gradient check failed at layer {layer_idx} {param_name} {idx}: rel_error={rel_error}"
                it.iternext()
    print("-> Gradient Check PASSED! All relative errors < 1e-6.\n")

if __name__ == "__main__":
    # 1. Load Dataset
    X, Y = generate_moons(n_samples=300, noise=0.15)
    
    # 2. Build Network Architecture (2 -> 8 -> 1)
    net = Network([
        Linear(input_dim=2, output_dim=8),
        Sigmoid(),
        Linear(input_dim=8, output_dim=1),
        Sigmoid()
    ])
    loss_fn = MSELoss()
    optimizer = SGD(net.layers, lr=0.8)

    # 3. Sanity check gradient on a single sample before training
    gradient_check(net, loss_fn, X[0], Y[0])

    # 4. Training Loop
    epochs = 8000
    print(f"--- STARTING TRAINING FOR {epochs} EPOCHS ---")
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(len(X)):
            y_pred = net.forward(X[i])
            loss = loss_fn.forward(y_pred, Y[i])
            total_loss += loss.item()
            
            net.backward(loss_fn.backward())
            optimizer.step()
            
        avg_loss = total_loss / len(X)
        epoch_losses.append(avg_loss)
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.6f}")

    # 5. Evaluate Accuracy
    correct = 0
    for i in range(len(X)):
        pred = net.forward(X[i]).item()
        pred_label = 1 if pred >= 0.5 else 0
        if pred_label == int(Y[i][0]):
            correct += 1
    accuracy = (correct / len(X)) * 100
    print(f"\nTraining Complete! Final Accuracy: {accuracy:.2f}%")

    # 6. VISUALIZATIONS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot A: Training Loss Curve
    axes[0].plot(range(1, epochs + 1), epoch_losses, color='purple', linewidth=2)
    axes[0].set_title("Training Loss Curve over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Squared Error (MSE)")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot B: Decision Boundary & Final Accuracy
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([net.forward(pt).item() for pt in grid_points]).reshape(xx.shape)

    contour = axes[1].contourf(xx, yy, Z, levels=50, cmap=plt.cm.Spectral, alpha=0.8)
    fig.colorbar(contour, ax=axes[1], label='Network Confidence')
    axes[1].scatter(X[:, 0], X[:, 1], c=Y.squeeze(), cmap=plt.cm.coolwarm, edgecolors='k', s=40)
    axes[1].set_title(f"Decision Boundary (Accuracy: {accuracy:.1f}%)")
    axes[1].set_xlabel("Feature X1")
    axes[1].set_ylabel("Feature X2")

    plt.tight_layout()
    plt.show()