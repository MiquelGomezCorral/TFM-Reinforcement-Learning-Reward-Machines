# Workflow

## Setup

The README documents the following setup:

```bash
conda create --name TFM_env python=3.13 -y
conda activate TFM_env
uv pip install -r requirements.txt
pip install -e .
uv pip install ipykernel
python -m ipykernel install --user --name=TFM_env --display-name "TFM_env (Conda)"
```

## Build

None yet.

## Test

None yet.

## Commit And PR

None yet.
