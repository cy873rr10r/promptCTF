# Quick Reference: PromptCTF-Env Implementation

## 7 Requirements - ALL COMPLETE ✅

### 1. Baseline Script
```bash
python scripts/baseline.py
# Outputs: task_id, difficulty, reward, steps to baseline_results.json
```

### 2. Graders Module
```python
from src.env.graders import grade_extraction
reward, explanation = grade_extraction(defender_response, task)
```

### 3. State Endpoint
```bash
GET /env/{env_id}/state
# Returns: conversation, attempts_used, flag_status, difficulty
```

### 4. Config Verified
- ✅ Action space: `text`
- ✅ Observation space: `text` + float reward
- ✅ Red + Blue modes defined
- ✅ All 3 tasks (easy/medium/hard) with max_steps

### 5. Qwen2.5-0.5B Defender
```python
env = PromptCTFEnv(defender_backend="qwen")  # Real defender
# Uses: Qwen 2.5 0.5B + 4-bit quantization + ModelLoader
```

### 6. Config-Based Model Swap
```yaml
# Local dev
models:
  attacker:
    model_name: "Qwen/Qwen2.5-0.5B-Instruct"

# For Kaggle T4 - just change one line:
models:
  attacker:
    model_name: "Qwen/Qwen2.5-7B-Instruct"
```

### 7. No Hardcoded Paths
- ✅ All `/media/cybter/Labs/` → `/path/to/promptctf`
- ✅ All usernames removed
- ✅ Generic paths in scripts

---

## File Manifest

```
✅ scripts/baseline.py                  NEW - evaluation script
✅ src/env/graders.py                   NEW - grading system
✅ src/server/main.py                   UPDATED - added /state endpoint
✅ configs/openenv.yaml                 UPDATED - 0.5B + proper structure
✅ src/env/qwen_defender.py            EXISTS - real Qwen defender
✅ README.md                            UPDATED - generic paths
✅ INSTALLATION.md                      UPDATED - generic paths
✅ scripts/validate_install.py          UPDATED - graders.py + baseline.py
✅ REQUIREMENTS_COMPLETE.md             NEW - this checklist
```

---

## Dependencies

**Always required**:
```
fastapi wandb tqdm pyyaml requests pydantic
```

**For training**:
```
torch transformers trl peft unsloth[colab-new]
```

**For UI** (optional):
```
gradio
```

---

## Quick Start Commands

```bash
# 1. Setup
cd /path/to/promptctf
pip install -r requirements.txt
pip install wandb  # Mandatory for training

# 2. Validate
python scripts/validate_install.py

# 3. Test mock environment
python scripts/test_grpo_training.py

# 4. Run baseline
python scripts/baseline.py

# 5. Start API server
python -m uvicorn src.server.main:app --reload

# 6. Test API
curl http://localhost:8000/health
```

---

## API Endpoints Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | /health | Health check |
| GET | /tasks | List all tasks |
| GET | /tasks/{task_id} | Get task details |
| POST | /env/create | Create environment |
| POST | /env/{id}/reset | Reset environment |
| POST | /env/{id}/step | Execute action |
| GET | /env/{id}/state | Get state (**NEW**) |
| GET | /env/{id}/info | Get environment info |
| DELETE | /env/{id} | Delete environment |
| GET | /env | List all environments |

---

## Graders Output Format

```python
# All graders return:
(reward: float [0.0-1.0], explanation: str)

# Examples:
(1.0, "Full flag extracted")
(0.5, "2/4 segments extracted")
(0.25, "Partial flag: 25% extracted")
(0.0, "No flag extracted")
```

---

## Environment State Structure

```json
{
  "env_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_id": "easy",
  "difficulty": "easy",
  "mode": "red",
  "current_step": 1,
  "max_steps": 8,
  "done": false,
  "episode_reward": 0.5,
  "attempts_used": 1,
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "flag_status": {
    "extracted": false,
    "leaked_ratio": 0.0
  }
}
```

---

## Configuration Model Scaling

### Local Development (CPU)
```yaml
model_name: "Qwen/Qwen2.5-0.5B-Instruct"
device: "cpu"
use_4bit: true
```

### Kaggle T4 GPU
```yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
device: "cuda"
use_4bit: true
```

---

## Status: ✅ COMPLETE

All 7 requirements implemented, tested, and documented.
Ready for deployment.
