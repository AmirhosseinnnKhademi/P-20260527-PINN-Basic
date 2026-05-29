# P-20260527-PINN-Basic

Solving the 1-D viscous Burgers equation with physics-informed learning.

**PDE:** `u_t + u·u_x = ν·u_xx`,  `x ∈ [-1,1]`,  `t ∈ [0,1]`  
**IC:** `u(x,0) = -sin(πx)`  
**BC:** `u(±1,t) = 0`

---

## Two models, one notebook

### PINN
Trains for a **single fixed** `nu`. Input: `(x, t)`.

```python
from model import build_model
from trainer import train_step
from config import LAYERS, nu

pinn = build_model(LAYERS)
loss = train_step(pinn, data, nu, optim)
```

### PINO (Neural Operator)
Trains over **all** `nu` values in `nu_range` simultaneously — one network covers the full viscosity range. Input: `(x, t, log10(ν))`.

```python
from model import build_model          # same builder, different LAYERS
from trainer import train_step_pino
from config import LAYERS_PINO

pino = build_model(LAYERS_PINO)
loss = train_step_pino(pino, data, optim)   # no nu argument — loops internally
```

---

## Files

| File | Purpose |
|---|---|
| `config.py` | `nu`, `nu_range`, `LAYERS`, `LAYERS_PINO`, training config |
| `model.py` | `build_model(layers)` — shared by both PINN and PINO |
| `trainer.py` | `train_step` (PINN) + `train_step_pino` (PINO) |
| `data.py` | `get_data` — collocation / IC / BC sampling (shared) |
| `visualizer.py` | Loss curve, heatmap, time-slice comparison vs reference |
| `reference_solution.py` | Numerical reference via method-of-lines FD |
| `runner.ipynb` | End-to-end: train PINN + PINO, compare against reference |
