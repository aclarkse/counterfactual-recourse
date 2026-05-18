# Environment Setup — uv

## One-time setup

```powershell
# 1. Create the environment and install all dependencies, including a
#    CUDA 12.8 PyTorch build (pinned in pyproject.toml / uv.lock). cu128
#    ships the sm_120 kernels required by Blackwell / RTX 50-series GPUs.
uv sync

# 2. Register the Jupyter kernel so it appears in VS Code / Jupyter
uv run python -m ipykernel install --user --name causal-recourse --display-name "Python (causal-recourse)"
```

> Need a different CUDA build or a CPU-only install? Repoint `torch`/`torchvision`
> in `[tool.uv.sources]` (`pyproject.toml`) to another `pytorch-*` index — e.g.
> `pytorch-cpu` or `pytorch-cu126` — then re-run `uv lock && uv sync`. Run
> `nvidia-smi` to check your driver's CUDA version.

## Every time you work

```powershell
# Run anything inside the environment
uv run jupyter notebook
uv run python your_script.py

# Or activate the venv explicitly if you prefer
.venv\Scripts\Activate.ps1
jupyter notebook
```

## Adding a new package

```powershell
uv add some-package          # adds to pyproject.toml and installs
uv add "some-package>=1.2"   # with version constraint
```

## Checking your environment

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
