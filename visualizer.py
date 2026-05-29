import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.semilogy(history)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

def _model_input(x_in, t_in, nu_val=None):
    inp = tf.concat([x_in, t_in], axis=1)
    if nu_val is not None:
        log_nu = tf.fill(tf.shape(x_in), float(np.log10(nu_val)))
        inp = tf.concat([inp, log_nu], axis=1)
    return inp


def plot_solution(model, nu_val=None):
    x_plot = np.linspace(-1, 1, 256).astype(np.float32)
    t_plot = np.linspace(0, 1, 100).astype(np.float32)
    X, T = np.meshgrid(x_plot, t_plot)
    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)
    u_pred = model(_model_input(x_flat, t_flat, nu_val)).numpy().reshape(100, 256)

    plt.pcolormesh(t_plot, x_plot, u_pred.T, cmap="RdBu_r", shading='auto')
    plt.colorbar(label='u(x, t)')
    plt.ylabel('x'); plt.xlabel('t')
    title = 'Burgers PINO Solution' if nu_val is not None else 'Burgers PINN Solution'
    plt.title(title + (f'  (nu={nu_val:.4f})' if nu_val is not None else ''))
    plt.show()

def plot_slices(model, t_slices=[0.0, 0.25, 0.5, 0.75, 1.0], ref_path="reference_data.npz", nu_val=None):
    x_arr = np.linspace(-1, 1, 256)
    fig, axes = plt.subplots(1, len(t_slices), figsize=(4*len(t_slices), 4), sharey=True)

    # Load reference data, generating it on the fly if needed
    if not os.path.exists(ref_path):
        print(f"Reference file '{ref_path}' not found — generating now (takes ~1 min)…")
        from reference_solution import generate_reference_solution
        generate_reference_solution(save_path=ref_path)
    ref = np.load(ref_path)

    for ax, t_val in zip(axes, t_slices):
        x_in = x_arr.reshape(-1, 1).astype(np.float32)
        t_in = np.full_like(x_in, t_val)
        u_pred = model(_model_input(x_in, t_in, nu_val)).numpy().flatten()

        t_idx = np.argmin(np.abs(ref["t"] - t_val))
        ax.plot(ref["x"], ref["u_ref"][t_idx], 'k-', lw=1.5, label='Reference')
        ax.plot(x_arr, u_pred, 'm--', lw=1.5, label='Prediction')
        ax.set_title(f't = {t_val:.2f}')
        ax.set_xlabel('x')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('u(t, x)')
    axes[0].legend()
    plt.tight_layout()
    plt.show()


# ── Multi-model comparison ─────────────────────────────────────────────────────

_STYLE = {
    'Solver':    ('black',      '-',   2.0),
    'PINN':      ('tab:blue',   '--',  1.5),
    'PINO':      ('tab:green',  '-.',  1.5),
    'PI-FNO':    ('tab:purple', ':',   1.8),
}


def plot_loss_comparison(histories, labels=None):
    """Overlay training loss curves of multiple models on a single log plot."""
    if labels is None:
        labels = [f'Model {i}' for i in range(len(histories))]
    colors = ['tab:blue', 'tab:green', 'tab:purple']
    for h, lbl, c in zip(histories, labels, colors):
        plt.semilogy(h, label=lbl, color=c)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training Loss — all models')
    plt.legend(); plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout(); plt.show()


def plot_eval(preds, x_arr, t_arr, nu_val, scenario,
              t_slices=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Two-row comparison figure for one evaluation scenario.

    Parameters
    ----------
    preds    : OrderedDict  name -> u array (nt, nx).
               Keys expected: 'Reference', 'PINN', 'PINO', 'PI-FNO'.
    x_arr    : (nx,) float32
    t_arr    : (nt,) float32
    nu_val   : float  viscosity of this scenario
    scenario : str    label for suptitle
    t_slices : time values at which to draw slice panels
    """
    n = len(preds)
    ref_u = preds.get('Reference', next(iter(preds.values())))
    vmin, vmax = float(ref_u.min()), float(ref_u.max())

    # ── x-t velocity contours ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5))
    for ax, (name, u) in zip(np.atleast_1d(axes), preds.items()):
        im = ax.pcolormesh(t_arr, x_arr, u.T, cmap='RdBu_r', shading='auto',
                           vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label='u')
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('t'); ax.set_ylabel('x')
    plt.suptitle(f'{scenario}  |  nu = {nu_val:.5f}', fontsize=12, y=1.01)
    plt.tight_layout(); plt.show()

    # ── u(x) time-slice comparison ───────────────────────────────────────────
    fig, axes = plt.subplots(1, len(t_slices),
                             figsize=(4 * len(t_slices), 3.5), sharey=True)
    for ax, t_val in zip(axes, t_slices):
        ti = int(np.argmin(np.abs(t_arr - t_val)))
        for name, u in preds.items():
            c, ls, lw = _STYLE.get(name, ('gray', '-', 1.5))
            ax.plot(x_arr, u[ti], color=c, ls=ls, lw=lw, label=name)
        ax.set_title(f't = {t_val:.2f}')
        ax.set_xlabel('x'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('u(x, t)')
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle(f'Time slices  |  {scenario}  |  nu = {nu_val:.5f}', fontsize=12, y=1.01)
    plt.tight_layout(); plt.show()


# ── PI-FNO visualisation ───────────────────────────────────────────────────────

def _fno_predict(model, x_grid, t_grid):
    """Run the FNO and return u as a (Nx, Nt) numpy array."""
    u0    = -np.sin(np.pi * x_grid).astype(np.float32)
    fno_in = np.stack([u0, x_grid], axis=-1)[np.newaxis]  # (1, Nx, 2)
    u_pred = model(fno_in.astype(np.float32)).numpy()[0]   # (Nx, Nt)
    return u_pred


def plot_fno_solution(model, nx=128, nt=50):
    x_grid = np.linspace(-1, 1, nx).astype(np.float32)
    t_grid = np.linspace(0,  1, nt).astype(np.float32)
    u_pred = _fno_predict(model, x_grid, t_grid)           # (Nx, Nt)

    plt.pcolormesh(t_grid, x_grid, u_pred, cmap="RdBu_r", shading='auto')
    plt.colorbar(label='u(x, t)')
    plt.ylabel('x'); plt.xlabel('t')
    plt.title('Burgers PI-FNO Solution')
    plt.show()


def plot_fno_slices(model, nx=128, nt=50,
                    t_slices=(0.0, 0.25, 0.5, 0.75, 1.0),
                    ref_path="reference_data.npz"):
    x_grid = np.linspace(-1, 1, nx).astype(np.float32)
    t_grid = np.linspace(0,  1, nt).astype(np.float32)
    u_pred = _fno_predict(model, x_grid, t_grid)           # (Nx, Nt)

    if not os.path.exists(ref_path):
        print(f"Reference file '{ref_path}' not found — generating now (takes ~1 min)…")
        from reference_solution import generate_reference_solution
        generate_reference_solution(save_path=ref_path)
    ref = np.load(ref_path)

    fig, axes = plt.subplots(1, len(t_slices), figsize=(4 * len(t_slices), 4), sharey=True)
    for ax, t_val in zip(axes, t_slices):
        t_idx_pred = np.argmin(np.abs(t_grid - t_val))
        t_idx_ref  = np.argmin(np.abs(ref["t"] - t_val))
        ax.plot(ref["x"], ref["u_ref"][t_idx_ref], 'k-',  lw=1.5, label='Reference')
        ax.plot(x_grid,   u_pred[:, t_idx_pred],   'm--', lw=1.5, label='PI-FNO')
        ax.set_title(f't = {t_val:.2f}')
        ax.set_xlabel('x')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('u(x, t)')
    axes[0].legend()
    plt.tight_layout()
    plt.show()
