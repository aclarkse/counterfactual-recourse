# Environment Setup — uv

## One-time setup

```powershell
# 1. Create the environment and install all dependencies
uv sync

# 2. Install PyTorch — pick ONE of these based on your GPU:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu      # no GPU
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121    # CUDA 12.1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124    # CUDA 12.4

# Not sure which CUDA version? Run:
nvidia-smi

# 3. Register the Jupyter kernel so it appears in VS Code / Jupyter
uv run python -m ipykernel install --user --name causal-recourse --display-name "Python (causal-recourse)"
```

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
