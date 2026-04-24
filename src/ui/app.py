"""Gradio UI for PromptCTF-Env local testing."""

import gradio as gr
import requests
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Configuration
API_URL = "http://localhost:8000"
TASKS = ["easy", "medium", "hard"]


class PromptCTFUI:
    """Gradio UI for PromptCTF-Env"""
    
    def __init__(self):
        self.current_env_id: Optional[str] = None
        self.conversation_history: list = []
        self.episode_reward: float = 0.0
    
    def create_environment(self, task_id: str, mode: str) -> Tuple[str, str]:
        """Create a new environment"""
        try:
            response = requests.post(
                f"{API_URL}/env/create",
                json={"task_id": task_id, "mode": mode, "defender_backend": "mock"},
                timeout=10
            )
            
            if response.status_code != 200:
                return "Error", f"Failed to create environment: {response.text}"
            
            data = response.json()
            self.current_env_id = data["env_id"]
            
            # Reset environment
            reset_response = requests.post(
                f"{API_URL}/env/{self.current_env_id}/reset",
                json={},
                timeout=10
            )
            
            if reset_response.status_code != 200:
                return "Error", f"Failed to reset environment: {reset_response.text}"
            
            self.conversation_history = []
            self.episode_reward = 0.0

            status = (
                f"Environment created: {self.current_env_id}\n"
                f"Task: {task_id}\n"
                f"Mode: {mode}\n"
                "Ready to attack."
            )
            return "Success", status
        
        except Exception as e:
            logger.error(f"Create environment failed: {e}")
            return "Error", str(e)
    
    def attack_step(self, user_prompt: str) -> Tuple[str, float, str, str]:
        """Execute one attack step"""
        if not self.current_env_id:
            return "Please create an environment first.", 0.0, "Error", ""
        
        try:
            # Send step request
            response = requests.post(
                f"{API_URL}/env/{self.current_env_id}/step",
                json={"action": user_prompt},
                timeout=30
            )
            
            if response.status_code != 200:
                return f"Error: {response.text}", 0.0, "Error", ""
            
            data = response.json()
            
            obs = data["observation"]
            reward = data["reward"]
            terminated = data["terminated"]
            truncated = data["truncated"]
            info = data["info"]
            
            # Update state
            self.episode_reward += reward
            
            # Format defender response
            defender_response = info.get("defender_response", "No response")
            
            # Format reward info
            reward_info = info.get("reward_breakdown", {})
            extracted = reward_info.get("extracted_flag", False)
            blocked = reward_info.get("blocked_attack", False)
            ratio = reward_info.get("leak_ratio", 0.0)

            result_text = f"Defender: {defender_response}\n\n"
            result_text += f"Step reward: {reward:.3f}\n"
            result_text += f"Episode reward: {self.episode_reward:.3f}\n"
            result_text += f"Blocked: {blocked}\n"
            result_text += f"Leak ratio: {ratio:.2f}\n"

            if extracted:
                result_text += "\nResult: flag extracted."

            if terminated:
                result_text += "\nEpisode complete."
                status = "Flag Extracted"
            elif truncated:
                result_text += f"\nMax steps reached ({info.get('step')} steps)."
                status = "Max Steps"
            else:
                status = f"Step {info.get('step')}/{info.get('max_steps', 'N/A')}"

            return result_text, self.episode_reward, status, ""
        
        except Exception as e:
            logger.error(f"Step failed: {e}")
            return f"Error: {str(e)}", 0.0, "Error", ""
    
    def get_task_info(self, task_id: str) -> str:
        """Get information about a task"""
        try:
            response = requests.get(
                f"{API_URL}/tasks/{task_id}",
                timeout=5
            )
            
            if response.status_code != 200:
                return "Task not found"
            
            data = response.json()
            info = f"**Task:** {data['name']}\n\n"
            info += f"**Description:** {data['description']}\n\n"
            info += f"**Difficulty:** {data['difficulty']}\n"
            info += f"**Max Steps:** {data['max_steps']}\n"
            
            return info
        
        except Exception as e:
            logger.error(f"Get task info failed: {e}")
            return f"Error: {str(e)}"


def build_ui() -> gr.Blocks:
    """Build Gradio UI"""
    ui = PromptCTFUI()
    
    with gr.Blocks(title="PromptCTF-Env", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# PromptCTF-Env - Prompt Injection CTF")
        gr.Markdown("Extract flags through adversarial prompt injections")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Setup")
                
                task_selector = gr.Dropdown(
                    choices=TASKS,
                    value="easy",
                    label="Select Task",
                    interactive=True
                )

                mode_selector = gr.Dropdown(
                    choices=["red", "blue"],
                    value="red",
                    label="Mode",
                    interactive=True,
                )
                
                task_info = gr.Markdown(label="Task Info", value="Select a task to see details")
                
                def update_task_info(task_id):
                    return ui.get_task_info(task_id)
                
                task_selector.change(update_task_info, inputs=[task_selector], outputs=[task_info])
                
                create_btn = gr.Button("Create Environment", size="lg", variant="primary")
                create_status = gr.Textbox(label="Status", interactive=False, lines=3)
                
                def on_create(task_id, mode):
                    result, status = ui.create_environment(task_id, mode)
                    return status

                create_btn.click(on_create, inputs=[task_selector, mode_selector], outputs=[create_status])
            
            with gr.Column(scale=2):
                gr.Markdown("## Attack Interface")
                
                # User input
                with gr.Row():
                    attack_input = gr.Textbox(
                        label="Your Prompt Injection",
                        lines=3,
                        placeholder="Enter your attack prompt here...",
                        interactive=True
                    )
                
                # Submit button
                submit_btn = gr.Button("Submit Attack", size="lg", variant="primary")
                
                # Results
                with gr.Row():
                    with gr.Column(scale=2):
                        defender_response = gr.Textbox(
                            label="Defender Response & Analysis",
                            lines=6,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        reward_display = gr.Number(
                            label="Episode Reward",
                            value=0.0,
                            interactive=False
                        )
                        
                        status_display = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Idle"
                        )
                
                def on_submit(user_prompt):
                    result, reward, status, clear_input = ui.attack_step(user_prompt)
                    return result, reward, status, clear_input
                
                submit_btn.click(
                    on_submit,
                    inputs=[attack_input],
                    outputs=[defender_response, reward_display, status_display, attack_input]
                )
        
        with gr.Row():
            gr.Markdown(
                """
                ## How It Works
                
                1. **Select a task** (Easy/Medium/Hard) based on difficulty
                2. **Create environment** to start a new CTF challenge
                3. **Send prompt injections** to try to extract the flag
                4. **Monitor the reward** - reaches 1.0 when flag is extracted
                
                ### Modes
                - **Red**: maximize flag extraction
                - **Blue**: maximize block rate
                """
            )
    
    return interface


if __name__ == "__main__":
    interface = build_ui()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
