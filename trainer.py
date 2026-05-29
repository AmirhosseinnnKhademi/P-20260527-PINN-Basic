import tensorflow as tf

def pde_residuals(model, x, t, nu):
    x = tf.convert_to_tensor(x)
    t = tf.convert_to_tensor(t)
    with tf.GradientTape(persistent=True) as t2: # persistent = True: to keep the tape open for more than 1 time differentiation
        t2.watch([x, t]) # since x and t are not default network variables (like weights), we need to manually ask gradient to watch it
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([x, t])
            u = model(tf.concat([x, t], axis=1))
        u_x = t1.gradient(u, x)
        u_t = t1.gradient(u, t)
    u_xx = t2.gradient(u_x, x)
    return u_t + u*u_x - nu * u_xx

def loss_fn(model, x_f, t_f, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, nu):
    r = pde_residuals(model, x_f, t_f, nu)
    l_f = tf.reduce_mean(tf.square(r))
    l_ic = tf.reduce_mean(tf.square(model(tf.concat([x_ic, t_ic], axis=1)) - u_ic))
    l_bc = tf.reduce_mean(tf.square(model(tf.concat([x_bc, t_bc], axis=1)) - u_bc))
    return l_f + 10* l_ic + 10* l_bc

def get_grad(model, x_f, t_f, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, nu):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x_f, t_f, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, nu)
    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad

@tf.function
def train_step(model, data, nu, optim):
    (x_f, t_f), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc) = data
    loss, grad = get_grad(model, x_f, t_f, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, nu)
    optim.apply_gradients(zip(grad, model.trainable_variables))
    return loss
