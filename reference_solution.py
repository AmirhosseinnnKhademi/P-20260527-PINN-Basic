"""
Reference solution for 1-D viscous Burgers equation:
  u_t + u*u_x = nu*u_xx,  x in [-1,1], t in [0,1]
  IC: u(x,0) = -sin(pi*x),  BC: u(-1,t) = u(1,t) = 0

Method: method of lines (2nd-order central FD + Radau).
Output: reference_data.npz  with keys  x (256,), t (201,), u_ref (201,256).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from config import nu


def generate_reference_solution(save_path="reference_data.npz"):
    N = 1024
    x_fine = np.linspace(-1, 1, N + 2)   # includes boundary nodes
    dx = x_fine[1] - x_fine[0]
    u0 = -np.sin(np.pi * x_fine[1:-1])   # IC on interior points
    t_eval = np.linspace(0, 1, 201)

    def rhs(t, u):
        u_pad = np.r_[0.0, u, 0.0]       # Dirichlet BCs
        u_x   = (u_pad[2:] - u_pad[:-2]) / (2 * dx)
        u_xx  = (u_pad[2:] - 2 * u + u_pad[:-2]) / dx**2
        return -u * u_x + nu * u_xx

    sol = solve_ivp(rhs, [0, 1], u0, method="Radau",
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)

    x_out = np.linspace(-1, 1, 256)
    u_ref = np.zeros((201, 256))
    for i in range(201):
        u_full  = np.r_[0.0, sol.y[:, i], 0.0]
        u_ref[i] = interp1d(x_fine, u_full, kind="cubic")(x_out)

    np.savez(save_path, x=x_out, t=t_eval, u_ref=u_ref)
    print(f"Saved {save_path}  —  u_ref shape: {u_ref.shape}")
    return x_out, t_eval, u_ref


if __name__ == "__main__":
    generate_reference_solution()
