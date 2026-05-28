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

    plt.pcolormesh(x_plot, t_plot, u_pred, cmap="RdBu_r", shading='auto')
    plt.colorbar(label='u(x, t)')
    plt.xlabel('x'); plt.ylabel('t')
    plt.title('Burgers PINN Solution')
    plt.show()
