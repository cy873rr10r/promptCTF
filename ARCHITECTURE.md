"""
ARCHITECTURE OVERVIEW
=====================

## System Design

PromptCTF-Env is a multi-component system for adversarial LLM training:

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio UI (Port 7860)                   │
│         Interactive interface for manual attacks           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ HTTP
                 │
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server (Port 8000)                     │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ OpenEnv-Compatible REST API                              ││
│ │ - Environment creation/reset/step                        ││
│ │ - Task management                                         ││
│ │ - Instance lifecycle                                      ││
│ └──────────────────────────────────────────────────────────┘│
└────────┬─────────────────────────────────────────────────────┘
         │
    ┌────┴──────────────────────────────┬──────────────────────┐
    │                                    │                      │
    ▼                                    ▼                      ▼
┌──────────────────┐        ┌──────────────────┐    ┌──────────────────┐
│  CTF Environment │        │ Task Registry    │    │ Reward Calculator│
│ (Gymnasium)      │        │                  │    │                  │
│                  │        │ - Easy (1)       │    │ - Flag extraction│
│ - Multi-turn     │        │ - Medium (2)     │    │ - Partial match  │
│   conversation   │        │ - Hard (3)       │    │ - Normalization  │
│ - Observation    │        │                  │    │ - Episode return │
│ - Action space   │        │                  │    │                  │
└────┬─────────────┘        └──────────────────┘    └──────────────────┘
     │
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│            Model Layer                                      │
│ ┌─────────────────────┐  ┌──────────────────────────────┐  │
│ │ Defender Model      │  │ Attacker Model (Training)    │  │
│ │ (Frozen)            │  │ (Fine-tuned with GRPO)       │  │
│ │                     │  │                              │  │
│ │ Qwen 2.5 7B         │  │ Qwen 2.5 7B + LoRA           │  │
│ │ device: CUDA        │  │ device: CUDA                 │  │
│ └─────────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│            Training Loop (GRPO)                             │
│                                                             │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Red Mode (Attacker Training)                         │   │
│ │                                                      │   │
│ │ 1. Sample attack prompts                            │   │
│ │ 2. Collect defender responses                       │   │
│ │ 3. Compute extraction rewards                       │   │
│ │ 4. GRPO update step                                 │   │
│ │ 5. Log to W&B                                       │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Blue Mode (Defender Training) - Future              │   │
│ │                                                      │   │
│ │ 1. Sample attack prompts                            │   │
│ │ 2. Compute block rewards                            │   │
│ │ 3. Update defender policy                           │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│            Monitoring (W&B)                                 │
│                                                             │
│ - Training metrics (loss, reward, extraction rate)         │
│ - Episode returns                                          │
│ - Attack success rate                                      │
│ - Model checkpoints                                        │
└─────────────────────────────────────────────────────────────┘
```

## Component Overview

### 1. Environment (src/environment.py)
Gymnasium-compatible environment implementing:
- **State**: Conversation history, task description, step count
- **Actions**: Text prompts (attacker's injections)
- **Rewards**: 0.0-1.0 based on flag extraction
- **Termination**: Flag extracted or max steps reached

### 2. Task Registry (src/tasks/__init__.py)
Manages CTF challenges:
- Easy: Basic "don't share secret" prompt
- Medium: Requires bypassing pattern matching
- Hard: Advanced reasoning-based defense

### 3. Reward System (src/rewards/__init__.py)
Computes rewards:
- Full extraction: 1.0
- Partial extraction: 0.0-0.5
- No extraction: 0.0
- Uses substring matching for partial credit

### 4. Model Manager (src/models/__init__.py)
Handles model loading:
- Loads Qwen models from HuggingFace
- Supports 4-bit/8-bit quantization
- Caches loaded models
- Manages CUDA memory

### 5. FastAPI Server (src/server/main.py)
REST API implementing OpenEnv spec:
- POST /env/create
- POST /env/{id}/reset
- POST /env/{id}/step
- GET /env/{id}
- DELETE /env/{id}

### 6. Gradio UI (src/ui/app.py)
Interactive web interface:
- Create environments
- Manual attack interface
- Real-time reward tracking
- Conversation history display

### 7. Training (src/training/red_train.py)
GRPO fine-tuning pipeline:
- Load base model with LoRA adapters
- Use Unsloth for efficiency
- GRPO optimization
- W&B logging

## Data Flow

### Inference (Attack) Flow
```
1. User sends prompt → FastAPI /env/{id}/step
2. Server adds prompt to conversation history
3. Defender model generates response
4. Response passed to reward calculator
5. Reward extracted and normalized
6. Observation returned to user
```

### Training Flow
```
1. Load attacker model with LoRA
2. Sample attack prompts
3. Run inference through defender
4. Compute extraction rewards
5. GRPO backward pass
6. Update LoRA weights
7. Log metrics to W&B
```

## Key Decisions

### Why Gymnasium?
- Standard RL interface
- Supports OpenEnv compatibility
- Easy integration with RL frameworks

### Why GRPO over PPO?
- Better for generation tasks
- Simpler reward structure
- Good convergence for binary outcomes

### Why LoRA?
- Efficient fine-tuning (2-5% params)
- Low memory footprint
- Fast inference (no merged weights needed)

### Why 4-bit Quantization?
- Fits 7B model on 24GB VRAM
- <1% accuracy loss
- 4x faster training

### Why Unsloth?
- 2x faster than standard
- Memory optimized
- Drop-in replacement

## Performance Considerations

### Memory Usage
- Base model: ~14GB (4-bit)
- LoRA adapters: ~500MB
- Batch inference: +2GB per batch

### Inference Speed
- Cold start: ~2s (model load)
- Per-step: ~100-200ms on V100
- Batch inference: ~50ms per sample

### Training Time
- Easy task: ~1 hour (3 epochs)
- Medium task: ~2 hours
- Hard task: ~3 hours
(On single V100)

## Extension Points

1. **New Tasks**: Add to TASK_REGISTRY
2. **New Reward Functions**: Extend RewardCalculator
3. **Blue Mode**: Implement DefenderGRPOTrainer
4. **Multi-task**: Extend environment
5. **Custom Models**: Support other model families

## Deployment Considerations

- GPU memory: 24GB+ recommended
- Disk space: ~20GB (models + outputs)
- Network: Single machine or local network
- Monitoring: W&B integration included
"""
