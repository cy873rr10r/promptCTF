# PromptCTF-Env

OpenEnv-compatible prompt injection CTF environment with local mock defender support.

## What You Can Run Now

- `env/` core logic with typed schemas
- FastAPI server for `create/reset/step`
- Gradio UI for manual attack testing
- Red/Blue modes with normalized rewards in `[0.0, 1.0]`
- W&B-compatible logging hooks in training scaffolds

This local setup does not require GPU. Defender behavior is mocked so you can validate env logic on an 8GB RAM machine.

## What Is Scaffolded For Later

- Qwen 2.5 7B attacker fine-tuning via Unsloth + TRL GRPO
- Blue defender training loop
- Real model execution once HF GPU credits are available

## Quick Start

```bash
cd /path/to/promptctf
python -m venv promptctf_venv
source promptctf_venv/bin/activate
pip install -r requirements.txt
```

Run API server:

```bash
python -m uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --reload
```

Run UI (new terminal):

```bash
python -m src.ui.app
```

- API docs: http://localhost:8000/docs
- UI: http://localhost:7860

## Modes

- Red mode: reward tracks extraction success (`1.0` for full flag leak)
- Blue mode: reward tracks block success (`1.0` when blocked with no leak)

## Core Layout

```text
src/
  env/
    models.py       # typed action/observation/reward schemas
    tasks.py        # easy/medium/hard tasks
    defender.py     # mock defender backend
    rewards.py      # reward logic for red/blue
    environment.py  # reset/step OpenEnv-style API
  server/main.py    # FastAPI service
  ui/app.py         # Gradio test UI
  training/
    red_train.py    # GRPO scaffold
    blue_train.py   # defender scaffold
```

## Training (Scaffold Only)

```bash
python -m src.training.red_train --task easy --config configs/openenv.yaml
python -m src.training.blue_train
```

These commands prepare output folders and configs but do not run GPU training yet.

## Docker

```bash
docker build -t promptctf-env .
docker-compose up
```

Container is CPU-oriented for local env validation.
