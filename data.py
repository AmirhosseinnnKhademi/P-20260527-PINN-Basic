import numpy as np


def get_data(N_f, N_ic, N_bc):
    #collocation
    x_f = np.random.uniform(-1, 1, (N_f, 1)).astype(np.float32)
    t_f = np.random.uniform(0, 1, (N_f, 1)).astype(np.float32)

    #IC
    x_ic = np.linspace(-1, 1, N_ic).reshape(-1,1).astype(np.float32)
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(np.pi * x_ic).astype(np.float32)

    #BC
    t_left = np.random.uniform(0, 1, (N_bc, 1)).astype(np.float32)
    t_right = np.random.uniform(0, 1, (N_bc, 1)).astype(np.float32)
    x_left = -np.ones_like(t_left)
    x_right = np.ones_like(t_right)
    t_bc = np.vstack([t_left, t_right])
    x_bc = np.vstack([x_left, x_right])
    u_bc = np.zeros_like(x_bc)

    return (x_f, t_f), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc)


def get_fno_data(nx, nt):
    """Grid data for PI-FNO: uniform x/t grids + Burgers IC u0(x) = -sin(πx).

    Returns:
        u0      (nx,)  float32 — initial condition values
        x_grid  (nx,)  float32 — x ∈ [-1, 1]
        t_grid  (nt,)  float32 — t ∈ [0, 1]
    """
    x_grid = np.linspace(-1, 1, nx).astype(np.float32)
    t_grid = np.linspace(0,  1, nt).astype(np.float32)
    u0     = -np.sin(np.pi * x_grid)
    return u0, x_grid, t_grid