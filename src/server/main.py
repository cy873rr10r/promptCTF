"""FastAPI OpenEnv server for local PromptCTF testing."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import uuid
import logging

from src.environment import PromptCTFEnv
from src.env.models import Mode
from src.env.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PromptCTF-Env",
    description="OpenEnv-compatible prompt injection CTF with mock local defender",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment instances
environments: Dict[str, PromptCTFEnv] = {}


# Request/Response models
class CreateEnvRequest(BaseModel):
    """Request to create a new environment"""
    task_id: str = "easy"
    mode: str = "red"
    defender_backend: str = "mock"
    seed: Optional[int] = None


class EnvResponse(BaseModel):
    """Response containing environment info"""
    env_id: str
    task_id: str
    mode: str
    status: str
    max_steps: int


class ResetRequest(BaseModel):
    """Request to reset environment"""
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    """Request to step environment"""
    action: str


class StepResponse(BaseModel):
    """Response from step action"""
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class ListTasksResponse(BaseModel):
    """Response listing available tasks"""
    tasks: List[str]
    total: int


# API Endpoints

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "PromptCTF-Env",
        "version": "1.0.0"
    }


@app.get("/tasks", response_model=ListTasksResponse, tags=["Tasks"])
async def list_tasks():
    """List available CTF tasks"""
    tasks = TASK_REGISTRY.list_ids()
    return ListTasksResponse(tasks=tasks, total=len(tasks))


@app.get("/tasks/{task_id}", tags=["Tasks"])
async def get_task(task_id: str):
    """Get details about a specific task"""
    try:
        task = TASK_REGISTRY.get(task_id)
        return {
            "task_id": task.task_id,
            "name": task.name,
            "description": task.description,
            "difficulty": task.difficulty.value,
            "max_steps": task.max_steps
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/env/create", response_model=EnvResponse, tags=["Environment"])
async def create_environment(request: CreateEnvRequest):
    """Create a new environment instance"""
    try:
        env_id = str(uuid.uuid4())

        try:
            mode = Mode(request.mode.lower())
        except ValueError as mode_error:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}") from mode_error

        env = PromptCTFEnv(
            task_id=request.task_id,
            mode=mode,
            defender_backend=request.defender_backend,
            seed=request.seed,
        )

        environments[env_id] = env
        
        logger.info(f"Created environment: {env_id} for task: {request.task_id}")

        return EnvResponse(
            env_id=env_id,
            task_id=request.task_id,
            mode=mode.value,
            status="created",
            max_steps=env.max_steps
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        raise HTTPException(status_code=500, detail="Failed to create environment")


@app.post("/env/{env_id}/reset", tags=["Environment"])
async def reset_environment(env_id: str, request: ResetRequest):
    """Reset an environment"""
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = environments[env_id]
    obs, info = env.reset(seed=request.seed, options=request.options)
    
    return {
        "observation": obs,
        "info": info
    }


@app.post("/env/{env_id}/step", response_model=StepResponse, tags=["Environment"])
async def step_environment(env_id: str, request: StepRequest):
    """Execute one step in the environment"""
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = environments[env_id]

    try:
        result = env.step(request.action)

        return StepResponse(
            observation=result.observation.to_dict(),
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Step failed for env {env_id}: {e}")
        raise HTTPException(status_code=500, detail="Step execution failed")


@app.get("/env/{env_id}/info", tags=["Environment"])
async def get_env_info(env_id: str):
    """Get information about an environment"""
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = environments[env_id]
    
    return {
        "env_id": env_id,
        "task_id": env.task.task_id,
        "mode": env.mode.value,
        "current_step": env.current_step,
        "max_steps": env.max_steps,
        "done": env.done,
        "episode_reward": env.episode_reward
    }


@app.get("/env/{env_id}/state", tags=["Environment"])
async def get_env_state(env_id: str):
    """Get detailed state of an environment including conversation and progress."""
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")

    env = environments[env_id]

    # Build conversation history
    conversation = []
    for msg in env.conversation[1:]:  # Skip system message
        conversation.append({
            "role": msg.role,
            "content": msg.content
        })

    return {
        "env_id": env_id,
        "task_id": env.task.task_id,
        "difficulty": env.task.difficulty.value,
        "mode": env.mode.value,
        "current_step": env.current_step,
        "max_steps": env.max_steps,
        "done": env.done,
        "episode_reward": env.episode_reward,
        "attempts_used": env.current_step,
        "conversation": conversation,
        "flag_status": {
            "extracted": False,  # Placeholder: could track if flag was leaked
            "leaked_ratio": 0.0,  # Placeholder: from last step
        }
    }


@app.delete("/env/{env_id}", tags=["Environment"])
async def delete_environment(env_id: str):
    """Delete an environment"""
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = environments.pop(env_id)
    env.close()
    
    logger.info(f"Deleted environment: {env_id}")
    
    return {"status": "deleted", "env_id": env_id}


@app.get("/env", tags=["Environment"])
async def list_environments():
    """List all active environments"""
    snapshot = {
        env_id: {
            "task_id": env.task.task_id,
            "mode": env.mode.value,
            "step": env.current_step,
            "done": env.done,
        }
        for env_id, env in environments.items()
    }
    return {
        "active_environments": len(environments),
        "environments": snapshot,
    }


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("PromptCTF-Env server starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Cleaning up environments...")
    for env_id, env in environments.items():
        env.close()
    environments.clear()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
