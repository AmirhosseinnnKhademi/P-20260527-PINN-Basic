import numpy as np

def get_data(N_f, N_bc, N_ic):
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