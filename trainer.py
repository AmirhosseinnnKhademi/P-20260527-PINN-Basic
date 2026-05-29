import numpy as np
import tensorflow as tf
from config import nu_range


class _BaseTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def _loss(self, data):
        raise NotImplementedError

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._loss(data)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class PINNTrainer(_BaseTrainer):
    def __init__(self, model, optimizer, nu):
        super().__init__(model, optimizer)
        self.nu = float(nu)

    def _pde_residuals(self, x, t):
        x = tf.convert_to_tensor(x)
        t = tf.convert_to_tensor(t)
        with tf.GradientTape(persistent=True) as t2:
            t2.watch([x, t])
            with tf.GradientTape(persistent=True) as t1:
                t1.watch([x, t])
                u = self.model(tf.concat([x, t], axis=1))
            u_x = t1.gradient(u, x)
            u_t = t1.gradient(u, t)
        u_xx = t2.gradient(u_x, x)
        return u_t + u * u_x - self.nu * u_xx

    def _loss(self, data):
        (x_f, t_f), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = data
        r    = self._pde_residuals(x_f, t_f)
        l_f  = tf.reduce_mean(tf.square(r))
        l_ic = tf.reduce_mean(tf.square(self.model(tf.concat([x_ic, t_ic], axis=1)) - u_ic))
        l_bc = tf.reduce_mean(tf.square(self.model(tf.concat([x_bc, t_bc], axis=1)) - u_bc))
        return l_f + 10 * l_ic + 10 * l_bc


class PIFNOTrainer(_BaseTrainer):
    """Physics-informed Fourier Neural Operator trainer for 1-D Burgers.

    The FNO maps (u0(x), x) → u(x, t) on a fixed (nx × nt) grid.
    Physics loss is enforced via central finite differences on the grid output.

    data expected by train_step: (u0, x_grid, t_grid)
        u0      (nx,)  float32 — initial condition
        x_grid  (nx,)  float32 — spatial grid in [-1, 1]
        t_grid  (nt,)  float32 — time grid in [0, 1]
    """

    def __init__(self, model, optimizer, nu, nx, nt):
        super().__init__(model, optimizer)
        self.nu = float(nu)
        self.nx = nx
        self.nt = nt

    def _loss(self, data):
        u0, x_grid, t_grid = data
        dx = x_grid[1] - x_grid[0]
        dt = t_grid[1] - t_grid[0]

        fno_in = tf.concat([
            tf.reshape(u0,     [1, self.nx, 1]),
            tf.reshape(x_grid, [1, self.nx, 1]),
        ], axis=-1)                                   # (1, Nx, 2)

        u = self.model(fno_in)[0]                    # (Nx, Nt)

        # IC loss: u[:, 0] == u0
        l_ic = tf.reduce_mean(tf.square(u[:, 0] - u0))

        # BC loss: u[0, :] = u[-1, :] = 0
        l_bc = tf.reduce_mean(tf.square(u[0, :])) + tf.reduce_mean(tf.square(u[-1, :]))

        # PDE residual on interior points via central differences
        u_t  = (u[:, 2:]  - u[:, :-2])              / (2.0 * dt)   # (Nx,   Nt-2)
        u_x  = (u[2:, :]  - u[:-2, :])              / (2.0 * dx)   # (Nx-2, Nt)
        u_xx = (u[2:, :]  - 2.0 * u[1:-1, :] + u[:-2, :]) / dx**2 # (Nx-2, Nt)

        # Trim to shared interior: (Nx-2, Nt-2)
        residual = (u_t[1:-1, :] + u[1:-1, 1:-1] * u_x[:, 1:-1]
                    - self.nu * u_xx[:, 1:-1])

        l_pde = tf.reduce_mean(tf.square(residual))
        return l_pde + 10.0 * l_ic + 10.0 * l_bc


class PINOTrainer(_BaseTrainer):
    def _pde_residuals(self, x, t, nu_val):
        x = tf.convert_to_tensor(x)
        t = tf.convert_to_tensor(t)
        log_nu = tf.fill(tf.shape(x), float(np.log10(nu_val)))
        with tf.GradientTape(persistent=True) as t2:
            t2.watch([x, t])
            with tf.GradientTape(persistent=True) as t1:
                t1.watch([x, t])
                u = self.model(tf.concat([x, t, log_nu], axis=1))
            u_x = t1.gradient(u, x)
            u_t = t1.gradient(u, t)
        u_xx = t2.gradient(u_x, x)
        return u_t + u * u_x - float(nu_val) * u_xx

    def _loss(self, data):
        (x_f, t_f), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = data
        total = 0.0
        for nu_val in nu_range:
            log_nu = float(np.log10(nu_val))
            r    = self._pde_residuals(x_f, t_f, nu_val)
            l_f  = tf.reduce_mean(tf.square(r))
            l_ic = tf.reduce_mean(tf.square(
                self.model(tf.concat([x_ic, t_ic, tf.fill(tf.shape(x_ic), log_nu)], axis=1)) - u_ic))
            l_bc = tf.reduce_mean(tf.square(
                self.model(tf.concat([x_bc, t_bc, tf.fill(tf.shape(x_bc), log_nu)], axis=1)) - u_bc))
            total = total + l_f + 10 * l_ic + 10 * l_bc
        return total / len(nu_range)
