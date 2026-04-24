INSTALLATION GUIDE
==================

## Current Goal

Run PromptCTF environment logic locally without GPU, using a mock defender backend.

## 1. Prerequisites

- Python 3.10+
- 8GB RAM is enough for env/server/ui tests

## 2. Local Installation

```bash
cd /media/cybter/Labs/promptCTF
python -m venv prompctfvenv
source prompctfvenv/bin/activate
pip install -r requirements.txt
```

## 3. Run Services

Option A:

```bash
bash scripts/start.sh
```

Option B:

```bash
# Terminal 1
python -m uvicorn src.server.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2
python -m src.ui.app
```

## 4. Access

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- UI: http://localhost:7860

## 5. Verify Environment Logic

```bash
python scripts/test.py
```

## 6. Docker (CPU)

```bash
docker build -t promptctf-env .
docker-compose up
```

## 7. Training Scaffolds (No GPU Run Yet)

```bash
python -m src.training.red_train --task easy --config configs/openenv.yaml
python -m src.training.blue_train
```

These commands prepare output folders and placeholder artifacts. Actual Unsloth + TRL GRPO runs are postponed until HF GPU credits are available.

## 8. W&B Tracking

```bash
export WANDB_PROJECT=promptctf-env
export WANDB_ENTITY=your-username
```

W&B is ready for future real training runs.
