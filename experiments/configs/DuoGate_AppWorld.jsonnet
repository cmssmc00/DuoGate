local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
// Provider-specific model ID for the full DuoGate example.
local model_name = "MiniMax-M2.7";

local generator_model_config = {
    "name": model_name,
    "provider": "openai",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

local reflector_model_config = {
    "name": model_name,
    "provider": "openai",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

local curator_model_config = {
    "name": model_name,
    "provider": "openai",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

# Confidence assessor model.  Temperature MUST be 0 and scope MUST be
# risk_only so we don't pay the LLM cost on every step -- only mutations,
# terminal complete_task, mixed-risk, and unknown-risk code trigger the
# assessor.
local confidence_model_config = {
    "name": model_name,
    "provider": "openai",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

{
    "type": "ace",
    "config": {
        "run_type": "ace-adaptation",
        "agent": {
            "type": "ace_adaptation_react",
            "generator_model_config": generator_model_config,
            "reflector_model_config": reflector_model_config,
            "curator_model_config": curator_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_prompt.txt",
            "reflector_prompt_file_path": experiment_prompts_path + "/appworld_react_reflector_no_gt_prompt.txt",
            "curator_prompt_file_path": experiment_prompts_path + "/appworld_react_curator_prompt.txt", 
            "initial_playbook_file_path": experiment_playbooks_path + "/appworld_initial_playbook.txt", 
            "trained_playbook_file_path": experiment_playbooks_path + "/duogate_trained_playbook.txt",
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "confidence_model_config": confidence_model_config,
            "confidence_prompt_file_path": experiment_prompts_path + "/appworld_confidence_assessor_prompt.txt",
            "confidence_mode": "control",
            "confidence_scope": "risk_only",
            "confidence_log_dir": project_home_path + "/experiments/outputs/DuoGate_AppWorld/confidence_logs",
            "enable_complete_task_gate": true,
            "enable_mutation_gate": true,
            "enable_api_doc_gate": true,
            "enable_pagination_gate": true,
        },
        "dataset": "test_normal",
    }
}
