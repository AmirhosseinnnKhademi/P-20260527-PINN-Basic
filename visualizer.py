import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.semilogy(history)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

def plot_solution(model):
    x_plot = np.linspace(-1, 1, 256).astype(np.float32)
    t_plot = np.linspace(-1, 1, 100).astype(np.float32)
    X, T = np.meshgrid(x_plot, t_plot)
    x_flat = X.flatten().reshape(-1,1)
    t_flat = T.flatten().reshape(-1,1)
    u_pred = model(tf.concat([x_flat, t_flat], axis=1)).numpy().reshape(100, 256)

    plt.pcolormesh(t_plot, x_plot, u_pred, cmap="RdBu_r", shading='auto')
    plt.colorbar(label='u(x, t)')
    plt.ylabel('x'); plt.xlabel('t')
    plt.title('Burgers PINN Solution')
    plt.show()

def plot_slices(model, t_slices=[0.0, 0.25, 0.5, 0.75, 1.0]):
    x_arr = np.linspace(-1, 1, 256)
    fig, axes = plt.subplots(1, len(t_slices), figsize=(4*len(t_slices), 4), sharey=True)

    for ax, t_val in zip(axes, t_slices):
        x_in = x_arr.reshape(-1, 1).astype(np.float32)
        t_in = np.full_like(x_in, t_val)
        u_pred = model(tf.concat([x_in, t_in], axis=1)).numpy().flatten()

        ax.plot(x_arr, u_pred, 'm--', lw=1.5, label='Prediction')
        ax.set_title(f't = {t_val:.2f}')
        ax.set_xlabel('x')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('u(t, x)')
    axes[0].legend()
    plt.tight_layout()
    plt.show()
