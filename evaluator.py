"""
Model evaluation utilities for 1-D Burgers PINN / PINO / PI-FNO comparison.

All predict_* functions return u with shape (nt, nx): rows = time, cols = space.
"""
import os
import time
import tracemalloc
import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, RegularGridInterpolator


# ── Reference solution ─────────────────────────────────────────────────────────

def _ref_cache_path(nu_val, cache_dir, nx, nt):
    fname = f"ref_nu{nu_val:.10e}_nx{nx}_nt{nt}.npz"
    return os.path.join(cache_dir, fname)


def load_or_generate_reference(nu_val, cache_dir="ref_cache", nx=256, nt=201):
    """Return (x_arr, t_arr, u_ref), loading from disk if cached, else computing and saving.

    The .npz file is stored under `cache_dir/` and reused on every subsequent call,
    so the expensive Radau integration only runs once per (nu_val, nx, nt) combination.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _ref_cache_path(nu_val, cache_dir, nx, nt)

    if os.path.exists(path):
        data = np.load(path)
        return data["x"].astype(np.float32), data["t"].astype(np.float32), data["u_ref"].astype(np.float32)

    x_arr, t_arr, u_ref = generate_reference(nu_val, nx=nx, nt=nt)
    np.savez(path, x=x_arr, t=t_arr, u_ref=u_ref, nu=np.float64(nu_val))
    return x_arr, t_arr, u_ref


def generate_reference(nu_val, nx=256, nt=201):
    """High-fidelity MOL/Radau reference for 1-D Burgers at a given nu.

    Returns
    -------
    x_arr : (nx,)  float32
    t_arr : (nt,)  float32
    u_ref : (nt, nx)  float32
    """
    N = 1024
    x_fine = np.linspace(-1, 1, N + 2)
    dx = x_fine[1] - x_fine[0]
    u0_fine = -np.sin(np.pi * x_fine[1:-1])
    t_eval = np.linspace(0, 1, nt)

    def rhs(_, u):
        up = np.r_[0.0, u, 0.0]
        return -u * (up[2:] - up[:-2]) / (2 * dx) + nu_val * (up[2:] - 2 * u + up[:-2]) / dx**2

    sol = solve_ivp(rhs, [0, 1], u0_fine, method='Radau',
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)

    n_out = sol.y.shape[1]
    if n_out < nt:
        # Solver stopped early (shock / extreme stiffness); trim to computed range
        t_eval = sol.t
        nt = n_out

    x_out = np.linspace(-1, 1, nx)
    u_ref = np.zeros((nt, nx), dtype=np.float32)
    for i in range(nt):
        u_full = np.r_[0.0, sol.y[:, i], 0.0]
        u_ref[i] = interp1d(x_fine, u_full, kind='cubic')(x_out)

    return x_out.astype(np.float32), t_eval.astype(np.float32), u_ref


# ── Model prediction helpers ───────────────────────────────────────────────────

def predict_pinn(model, x_arr, t_arr):
    """Evaluate PINN (fixed-nu MLP) on a grid. Nu is baked into the model."""
    X, T = np.meshgrid(x_arr, t_arr)
    x_in = X.flatten().reshape(-1, 1).astype(np.float32)
    t_in = T.flatten().reshape(-1, 1).astype(np.float32)
    u = model(tf.concat([x_in, t_in], axis=1)).numpy()
    return u.reshape(len(t_arr), len(x_arr))


def predict_pino(model, x_arr, t_arr, nu_val):
    """Evaluate PINO with log10(nu) as the 3rd input feature."""
    log_nu = np.float32(np.log10(nu_val))
    X, T = np.meshgrid(x_arr, t_arr)
    x_in = X.flatten().reshape(-1, 1).astype(np.float32)
    t_in = T.flatten().reshape(-1, 1).astype(np.float32)
    ln   = np.full_like(x_in, log_nu)
    u = model(tf.concat([x_in, t_in, ln], axis=1)).numpy()
    return u.reshape(len(t_arr), len(x_arr))


def predict_solver(nu_val, nx=256, nt=201):
    """Numerical solver competitor: fresh Radau solve (no cache) for fair timing.

    L2 error vs. the reference is ~0 by construction; the value of including it
    in the comparison is its wall-clock time and memory footprint.
    """
    _, _, u = generate_reference(nu_val, nx=nx, nt=nt)
    return u


def predict_fno(model, x_arr, t_arr, nx_fno, nt_fno):
    """Evaluate PI-FNO on its native grid and bilinearly interpolate to (x_arr, t_arr)."""
    x_fno = np.linspace(-1, 1, nx_fno).astype(np.float32)
    t_fno = np.linspace(0,  1, nt_fno).astype(np.float32)
    u0    = -np.sin(np.pi * x_fno)
    fno_in = np.stack([u0, x_fno], axis=-1)[np.newaxis].astype(np.float32)
    u_fno  = model(fno_in).numpy()[0]                          # (nx_fno, nt_fno)

    interp = RegularGridInterpolator(
        (x_fno, t_fno), u_fno, method='linear',
        bounds_error=False, fill_value=None)
    X, T = np.meshgrid(x_arr, t_arr)
    pts  = np.stack([X.flatten(), T.flatten()], axis=-1)
    return interp(pts).reshape(len(t_arr), len(x_arr))


# ── Metrics ────────────────────────────────────────────────────────────────────

def rel_l2(u_pred, u_ref):
    """Relative L2 error: ||u_pred - u_ref|| / ||u_ref||."""
    return float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))


def time_predict(predict_fn, n_runs=5, warmup=True):
    """Mean wall-clock inference time (seconds).

    warmup=True  — one extra call before timing (lets TF/JIT compile; use for neural models).
    warmup=False — skip warmup (use for the numerical solver: deterministic, no JIT).
    """
    if warmup:
        predict_fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        predict_fn()
    return (time.perf_counter() - t0) / n_runs


def peak_memory_mb(predict_fn):
    """Peak Python-heap allocation (MB) during one inference call via tracemalloc."""
    tracemalloc.start()
    predict_fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024**2


def param_mb(model):
    """Model weight footprint in MB, assuming float32 (4 bytes per parameter)."""
    n = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    return n * 4 / 1024**2
