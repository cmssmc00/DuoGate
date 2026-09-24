import copy
import json
import os
import re
from typing import Any

from jinja2 import Template

from appworld import AppWorld
from appworld.common.utils import read_file
from appworld_experiments.code.ace.adaptation_agent import StarAgent, ExecutionIO
from appworld_experiments.code.ace.lite_llm_generator import LiteLLMGenerator
from appworld_experiments.code.ace.appworld_confidence import AppWorldConfidenceController
from .playbook import (
    apply_curator_operations,
    extract_json_from_text,
    get_next_global_id,
    sanitize_curator_operations,
    select_playbook_for_task,
)


def _build_confidence_controller(
    confidence_model_config: dict | None,
    confidence_prompt_file_path: str | None,
    confidence_mode: str = "control",
    confidence_scope: str = "risk_only",
    confidence_log_dir: str | None = None,
    enable_complete_task_gate: bool = True,
    enable_mutation_gate: bool = True,
    enable_api_doc_gate: bool = True,
    enable_pagination_gate: bool = True,
    experiment_name: str | None = None,
) -> AppWorldConfidenceController | None:
    """Construct a confidence controller in a backward-compatible way.

    If `confidence_model_config` is None we still build the controller so
    that the local heuristic gates (complete_task / mutation) are active,
    but `confidence_model` will be None and the LLM assessor will be
    skipped.
    """
    confidence_model = None
    if confidence_model_config:
        try:
            confidence_model = LiteLLMGenerator(**confidence_model_config)
        except Exception as e:
            print(f"[confidence] failed to build confidence model: {e}")
            confidence_model = None
    return AppWorldConfidenceController(
        confidence_model=confidence_model,
        prompt_file_path=confidence_prompt_file_path,
        mode=confidence_mode or "control",
        scope=confidence_scope or "risk_only",
        log_dir=confidence_log_dir,
        enable_complete_task_gate=bool(enable_complete_task_gate),
        enable_mutation_gate=bool(enable_mutation_gate),
        enable_api_doc_gate=bool(enable_api_doc_gate),
        enable_pagination_gate=bool(enable_pagination_gate),
        experiment_name=experiment_name,
    )


@StarAgent.register("ace_adaptation_react")
class SimplifiedReActStarAgent(StarAgent):
    def __init__(
        self,
        generator_prompt_file_path: str | None = None,
        reflector_prompt_file_path: str | None = None,
        curator_prompt_file_path: str | None = None,
        initial_playbook_file_path: str | None = None,
        trained_playbook_file_path: str | None = None,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        confidence_model_config: dict | None = None,
        confidence_prompt_file_path: str | None = None,
        confidence_mode: str = "control",
        confidence_scope: str = "risk_only",
        confidence_log_dir: str | None = None,
        enable_complete_task_gate: bool = True,
        enable_mutation_gate: bool = True,
        enable_api_doc_gate: bool = True,
        enable_pagination_gate: bool = True,
        enable_playbook_retrieval: bool = True,
        max_injected_playbook_chars: int = 60000,
        min_injected_playbook_chars: int = 15000,
        max_injected_playbook_bullets: int = 80,
        max_curator_new_bullets_per_task: int = 2,
        max_curator_bullet_chars: int = 800,
        max_curator_update_chars: int = 1600,
        enable_curator_dedup: bool = True,
        learning_update_policy: str = "failure_or_high_risk",
        skip_learning_on_clean_success: bool = True,
        high_risk_step_threshold: int = 25,
        enable_periodic_learning_update: bool = False,
        periodic_learning_update_every: int = 20,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.generator_prompt_template = read_file(generator_prompt_file_path.replace("/", os.sep)).lstrip()
        self.reflector_prompt = read_file(reflector_prompt_file_path.replace("/", os.sep))
        self.curator_prompt_file_path = curator_prompt_file_path
        self.curator_prompt = read_file(curator_prompt_file_path.replace("/", os.sep))
        self.trained_playbook_file_path = trained_playbook_file_path
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
        self.ignore_multiple_calls = ignore_multiple_calls
        self.partial_code_regex = r".*```python\n(.*)"
        self.full_code_regex = r"```python\n(.*?)```"
        self.world_gt_code = None  # Store ground truth code for STAR reflection
        self.enable_playbook_retrieval = bool(enable_playbook_retrieval)
        self.max_injected_playbook_chars = int(max_injected_playbook_chars or 60000)
        self.min_injected_playbook_chars = int(min_injected_playbook_chars or 15000)
        self.max_injected_playbook_bullets = int(max_injected_playbook_bullets or 80)
        self.max_curator_new_bullets_per_task = int(max_curator_new_bullets_per_task or 2)
        self.max_curator_bullet_chars = int(max_curator_bullet_chars or 800)
        self.max_curator_update_chars = int(max_curator_update_chars or 1600)
        self.enable_curator_dedup = bool(enable_curator_dedup)
        self.learning_update_policy = learning_update_policy or "failure_or_high_risk"
        self.skip_learning_on_clean_success = bool(skip_learning_on_clean_success)
        self.high_risk_step_threshold = int(high_risk_step_threshold or 25)
        self.enable_periodic_learning_update = bool(enable_periodic_learning_update)
        self.periodic_learning_update_every = int(periodic_learning_update_every or 20)
        self.selected_playbook = ""
        self.playbook_selection_metadata: dict[str, Any] = {}
        self.confidence_events: list[dict[str, Any]] = []
        self._task_api_error_count = 0
        self._task_python_error_count = 0

        if os.path.exists(initial_playbook_file_path):
            self.playbook = read_file(initial_playbook_file_path.replace("/", os.sep))
        else:
            self.playbook = "(empty)" # default empty playbook

        self.next_global_id = get_next_global_id(self.playbook)

        # ---- Confidence controller (AppWorld-specific) --------------------
        self.confidence_controller = _build_confidence_controller(
            confidence_model_config=confidence_model_config,
            confidence_prompt_file_path=confidence_prompt_file_path,
            confidence_mode=confidence_mode,
            confidence_scope=confidence_scope,
            confidence_log_dir=confidence_log_dir,
            enable_complete_task_gate=enable_complete_task_gate,
            enable_mutation_gate=enable_mutation_gate,
            enable_api_doc_gate=enable_api_doc_gate,
            enable_pagination_gate=enable_pagination_gate,
            experiment_name=None,
        )
        # Cache project home for API doc path resolution.
        self._project_home_path = os.environ.get("APPWORLD_PROJECT_PATH", "")

    def initialize(self, world: AppWorld):
        super().initialize(world)
        template = Template(self.generator_prompt_template)
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for (k, v) in world.task.app_descriptions.items()],
            indent=1,
        )
        self.confidence_events = []
        self._task_api_error_count = 0
        self._task_python_error_count = 0
        if self.enable_playbook_retrieval:
            self.selected_playbook, self.playbook_selection_metadata = select_playbook_for_task(
                self.playbook,
                world.task.instruction,
                max_chars=self.max_injected_playbook_chars,
                min_chars=self.min_injected_playbook_chars,
                max_bullets=self.max_injected_playbook_bullets,
            )
        else:
            self.selected_playbook = self.playbook
            self.playbook_selection_metadata = {
                "full_playbook_chars": len(self.playbook or ""),
                "selected_playbook_chars": len(self.playbook or ""),
                "selected_ratio": 1.0,
                "detected_apps": ["disabled"],
                "selected_bullet_count": None,
                "selected_section_count": None,
                "always_included_count": 0,
                "retrieval_mode": "disabled",
            }
        self._log_jsonl("playbook_selection.jsonl", {
            "task_id": getattr(world, "task_id", None),
            **self.playbook_selection_metadata,
            "max_chars": self.max_injected_playbook_chars,
            "min_chars": self.min_injected_playbook_chars,
            "max_bullets": self.max_injected_playbook_bullets,
        })
        template_params = {
            "input_str": world.task.instruction,
            "main_user": world.task.supervisor,
            "app_descriptions": app_descriptions,
            "relevant_apis": str(world.task.ground_truth.required_apis),
            "playbook": self.selected_playbook,
        }
        output_str = template.render(template_params)
        output_str = self.truncate_input(output_str) + "\n\n"
        self.messages = self.text_to_messages(output_str)
        self.num_instruction_messages = len(self.messages)

    def record_execution_outputs(self, execution_outputs: list[ExecutionIO]) -> None:
        if not execution_outputs:
            return
        assert len(execution_outputs) == 1, "React expects exactly one last_execution_output."
        output_content = (
            "Output:\n```\n" + self.truncate_output(execution_outputs[0].content) + "```\n\n"
        )
        self._record_execution_risk(output_content)
        self.messages.append({"role": "user", "content": output_content})

    def next_execution_inputs_and_cost(
        self, last_execution_outputs: list[ExecutionIO], world_gt_code: str = None, reasoning_text: str = ""
    ) -> tuple[ExecutionIO, float, str | None]:
        # Store ground truth code for later use in STAR reflection
        if world_gt_code is not None:
            self.world_gt_code = world_gt_code
        
        if reasoning_text != "" and reasoning_text is not None:
            self.messages.append({
                "role": "user",
                "content": "In your previous attempt, the code failed to match the ground truth outputs during unit testing. Provide reflection on what might have gone wrong and how to fix it."
            })
            self.messages.append({
                "role": "assistant",
                "content": reasoning_text + "\n\n"
            })
            self.messages.append({
                "role": "user",
                "content": "Use the reasoning above, along with the playbook of identified issues, to improve your code in all future attempts."
            })
            self.logger.show_message(role="user", message=reasoning_text, step_number=self.step_number)
        
        elif last_execution_outputs:
            self.record_execution_outputs(last_execution_outputs)

        





        '''messages = self.trimmed_messages
        output = self.generator_model.generate(messages=messages)
        
        # ========================================================
        # 🚦 置信度物理拦截器 (Confidence Interceptor) 开始
        # ========================================================
        import re
        content_text = output["content"]
        
        # 1. 解析大模型给出的置信度 (如果模型忘写了，默认给 10 放行)
        confidence_match = re.search(r'Confidence:\s*(\d+)', content_text)
        confidence_score = int(confidence_match.group(1)) if confidence_match else 10 
        
        # 2. 提取出它打算执行的 Python 代码
        code, fixed_output_content = self.extract_code_and_fix_content(content_text)
        
        # 3. 升级版：精准匹配危险 API 的函数调用，避免误伤文档查询等字符串参数
        # 必须满足：前面有点(调用方法) + 后面带括号(执行函数) 才会触发拦截
        is_dangerous_action = bool(re.search(
            r'\.(create_|update_|delete_|remove_|add_|send_|login)[a-zA-Z0-9_]*\s*\(', 
            code
        ))

        if confidence_score <= 5 and is_dangerous_action:
            code = (
                f"print('System Error: Execution Blocked! Your confidence score is "
                f"{confidence_score}/10 (<= 5). You are NOT allowed to execute state-changing "
                f"APIs or complete_task() when uncertain. Please write code to query more "
                f"information using show_ or search_ APIs first.')"
            )
        # ========================================================
        # 🚦 拦截器结束
        # ========================================================

        self.messages.append({"role": "assistant", "content": fixed_output_content + "\n\n"})'''
        
        messages = self.trimmed_messages
        output = self.generator_model.generate(messages=messages)
        code, fixed_output_content = self.extract_code_and_fix_content(output["content"])

        # ---- Confidence controller intercept -------------------------------
        # Runs BEFORE the code is executed.  Falls back to original code on
        # any internal error so the agent never crashes from this layer.
        if self.confidence_controller is not None and code:
            try:
                proposed_code_for_confidence = code
                task_instruction = ""
                world = getattr(self, "world", None)
                if world is not None and getattr(world, "task", None) is not None:
                    task_instruction = getattr(world.task, "instruction", "") or ""
                task_id_value = None
                if world is not None:
                    task_id_value = getattr(world, "task_id", None)

                api_docs_root = None
                if self._project_home_path:
                    api_docs_root = os.path.join(
                        self._project_home_path, "data", "api_docs"
                    )

                output_dir = None
                if world is not None and getattr(world, "output_misc_directory", None):
                    output_dir = world.output_misc_directory

                confidence_result = self.confidence_controller.control(
                    proposed_code=code,
                    proposed_content=fixed_output_content,
                    messages=self.messages,
                    task_instruction=task_instruction,
                    step_index=self.step_number,
                    api_docs_root=api_docs_root,
                    output_dir=output_dir,
                    task_id=task_id_value,
                )
                # IMPORTANT: code AND content must move together so that
                # what we execute and what we record in `self.messages`
                # stay consistent.  The controller guarantees that
                # `final_content` always reflects `final_code`.
                code = confidence_result.get("final_code", code) or code
                self._record_confidence_event(confidence_result, proposed_code=proposed_code_for_confidence)
                new_content = confidence_result.get("final_content")
                if isinstance(new_content, str) and new_content:
                    fixed_output_content = new_content
            except Exception as _conf_err:
                # Never let the controller crash the agent.
                print(f"[confidence] controller error (continuing): {_conf_err}")

        self.messages.append({"role": "assistant", "content": fixed_output_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_output_content, step_number=self.step_number
        )
        return [ExecutionIO(content=code)], output["cost"], None
    

    def extract_code_and_fix_content(self, text: str) -> tuple[str, str]:
        if text is None:
            return "", ""
        original_text = text
        output_code = ""
        match_end = 0
        # Handle multiple calls
        for re_match in re.finditer(self.full_code_regex, original_text, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if self.ignore_multiple_calls:
                text = original_text[: re_match.end()]
                return code, text
            output_code += code + "\n"
            match_end = re_match.end()
        # Check for partial code match at end (no terminating ```)  following the last match
        partial_match = re.match(
            self.partial_code_regex, original_text[match_end:], flags=re.DOTALL
        )
        if partial_match:
            output_code += partial_match.group(1).strip()
            # Terminated due to stop condition; add stop condition to output
            if not text.endswith("\n"):
                text = text + "\n"
            text = text + "```"
        if len(output_code) == 0:
            return "", text
        else:
            return output_code, text

    def truncate_input(self, input_str: str) -> str:
        if self.max_prompt_length is None:
            return input_str
        max_prompt_length = self.max_prompt_length
        goal_index = input_str.rfind("Task:")
        if goal_index == -1:
            raise ValueError(f"No goal found in input string:\n{input_str}")
        next_new_line_index = input_str.find("\n", goal_index) + 1
        init_prompt = input_str[:next_new_line_index]
        prompt = input_str[next_new_line_index:]
        if len(init_prompt) > max_prompt_length:
            raise ValueError("Input prompt longer than max allowed length")
        if len(prompt) > max_prompt_length - len(init_prompt):
            new_prompt = prompt[-(max_prompt_length - len(init_prompt)) :]
            cmd_index = new_prompt.find("ASSISTANT:") if "ASSISTANT:" in new_prompt else 0
            prompt = "\n[TRIMMED HISTORY]\n\n" + new_prompt[cmd_index:]
        return init_prompt + prompt
    
    def truncate_output(self, execution_output_content: str) -> str:
        if len(execution_output_content) > 20000:
            execution_output_content = execution_output_content[:20000] + "\n[REST NOT SHOWN FOR BREVITY]"
        return execution_output_content

    def text_to_messages(self, input_str: str) -> list[dict]:
        messages_json = []
        last_start = 0
        for m in re.finditer("(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
            last_end = m.span()[0]
            if len(messages_json) == 0:
                if last_end != 0:
                    raise ValueError(
                        f"Start of the prompt has no assigned role: {input_str[:last_end]}"
                    )
            else:
                messages_json[-1]["content"] = input_str[last_start:last_end]
            role = m.group(1).lower()
            messages_json.append({"role": role, "content": None})
            last_start = m.span()[1]
        messages_json[-1]["content"] = input_str[last_start:]
        return messages_json

    def messages_to_text(self, messages: list[dict]) -> str:
        output_str = ""
        for message in messages:
            role = message["role"]
            if role == "system":
                output_str += "SYSTEM:\n" + message["content"]
            if role == "assistant":
                output_str += "ASSISTANT:\n" + message["content"]
            elif role == "user":
                output_str += "USER:\n" + message["content"]
            else:
                raise ValueError(f"Unknown message role {role} in: {message}")
        return output_str

    @property
    def trimmed_messages(self) -> list[dict]:
        messages = copy.deepcopy(self.messages)
        pre_messages = messages[: self.num_instruction_messages - 1]
        post_messages = messages[self.num_instruction_messages - 1 :]
        output_str = self.messages_to_text(post_messages)
        remove_prefix = output_str[: output_str.index("Task: ") + 6]
        output_str = output_str.removeprefix(
            remove_prefix
        )  # not needed, it's only to match the original code
        observation_index = 0
        while len(output_str) > self.max_output_length:
            found_block = False
            # Dont remove observations from the last 5 blocks
            if observation_index < len(post_messages) - 5:
                # Find the next observation block to remove
                for message_index, message in enumerate(post_messages[observation_index:]):
                    # Only keep the code blocks and remove observations
                    if message["role"] == "user" and message["content"].startswith("Output:"):
                        message["content"] = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
                        found_block = True
                        observation_index += message_index + 1
                        break
                if not found_block:
                    observation_index = len(post_messages)
            # If no observation block left to trim, we need to start removing complete history blocks
            if not found_block and len(post_messages):
                first_post_message = copy.deepcopy(post_messages[0])
                if not first_post_message["content"].endswith("[TRIMMED HISTORY]\n\n"):
                    first_post_message["content"] += "[TRIMMED HISTORY]\n\n"
                post_messages = [first_post_message] + post_messages[2:]
                found_block = True
            if not found_block:
                raise ValueError(f"No blocks found to be removed!\n{post_messages}")
            output_str = self.messages_to_text(
                post_messages
            )  # not needed, it's only to match the original code
            output_str = output_str.removeprefix(remove_prefix)
        messages = pre_messages + post_messages
        return messages

    def _log_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        try:
            world = getattr(self, "world", None)
            output_dir = getattr(world, "output_misc_directory", None)
            if not output_dir:
                return
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)
            with open(path, "a", encoding="utf-8") as file:
                file.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            return

    def _record_execution_risk(self, output_text: str) -> None:
        text = (output_text or "").lower()
        if any(marker in text for marker in (
            "traceback", "exception", "error:", "api error", "validationerror",
            "not found", "permission denied", "insufficient", "balance",
        )):
            self._task_api_error_count += 1
        if any(marker in text for marker in ("traceback", "syntaxerror", "nameerror", "typeerror", "keyerror")):
            self._task_python_error_count += 1

    def _record_confidence_event(self, confidence_result: dict, proposed_code: str) -> None:
        try:
            final_code = confidence_result.get("final_code", proposed_code) or ""
            decision = confidence_result.get("decision", "") or ""
            event = {
                "step_index": self.step_number,
                "decision": decision,
                "code_was_replaced": final_code.strip() != (proposed_code or "").strip(),
                "risk": confidence_result.get("risk"),
            }
            gate = getattr(self.confidence_controller, "_last_set_level_gate", None)
            if gate is not None:
                event.update({
                    "set_level_triggered": bool(getattr(gate, "triggered", False)),
                    "set_level_mode": getattr(gate, "mode", "allow"),
                    "set_level_mismatch_type": getattr(gate, "mismatch_type", "none"),
                })
            self.confidence_events.append(event)
        except Exception:
            return

    def _evaluation_failed(self, evaluation_result=None) -> bool | None:
        if evaluation_result is None:
            return None
        try:
            failures = getattr(evaluation_result, "failures", None)
            if failures is not None:
                return len(failures) > 0
        except Exception:
            pass
        try:
            return not bool(getattr(evaluation_result, "success"))
        except Exception:
            return None

    def should_run_learning_update(
        self,
        task_state=None,
        confidence_events=None,
        set_level_events=None,
        trajectory_stats=None,
        evaluation_result=None,
    ) -> tuple[bool, str, dict]:
        confidence_events = confidence_events if confidence_events is not None else self.confidence_events
        trajectory_stats = trajectory_stats or {}
        task_failed = self._evaluation_failed(evaluation_result)
        step_count = int(trajectory_stats.get("step_count", self.step_number) or 0)
        high_risk_signals = []

        if task_failed is True:
            high_risk_signals.append("task_failed")
        if step_count >= self.max_steps:
            high_risk_signals.append("max_steps_reached")
        if step_count >= self.high_risk_step_threshold:
            high_risk_signals.append("high_step_count")
        if step_count >= max(1, self.max_steps - 3):
            high_risk_signals.append("near_max_steps")

        replacement_count = 0
        complete_task_block_count = 0
        set_level_soft_verify_count = 0
        for event in confidence_events or []:
            decision = str(event.get("decision") or "")
            if event.get("code_was_replaced"):
                replacement_count += 1
            if decision == "block_complete_task":
                complete_task_block_count += 1
            mode = str(event.get("set_level_mode") or "allow")
            mismatch = str(event.get("set_level_mismatch_type") or "none")
            if mode in ("soft_verify", "hard_block_complete", "hard_block_mutation"):
                set_level_soft_verify_count += 1
            if mismatch in ("missing", "extra", "wrong_action", "mixed"):
                high_risk_signals.append(f"set_level_{mismatch}")
        if replacement_count:
            high_risk_signals.append("confidence_code_replacement")
        if complete_task_block_count:
            high_risk_signals.append("complete_task_block")
        if complete_task_block_count > 1:
            high_risk_signals.append("repeated_complete_task_block")
        if set_level_soft_verify_count:
            high_risk_signals.append("set_level_verify_or_block")
        if self._task_api_error_count:
            high_risk_signals.append("api_or_runtime_error")
        if self._task_python_error_count > 1:
            high_risk_signals.append("repeated_python_error")

        task_text = ""
        try:
            task_text = getattr(getattr(self.world, "task", None), "instruction", "") or ""
        except Exception:
            task_text = ""
        lower_task = task_text.lower()
        money_or_external = any(k in lower_task for k in ("venmo", "splitwise", "pay", "request", "send", "email", "message", "delete", "remove", "update"))
        if money_or_external and (replacement_count or complete_task_block_count or self._task_api_error_count):
            high_risk_signals.append("high_risk_mutation_recovery")
        if "balance" in " ".join(str(e) for e in confidence_events).lower() or "funding" in " ".join(str(e) for e in confidence_events).lower():
            high_risk_signals.append("venmo_balance_or_funding")

        high_risk_signals = sorted(set(high_risk_signals))
        metadata = {
            "task_id": getattr(getattr(self, "world", None), "task_id", None),
            "run_learning_update": None,
            "learning_update_reason": "",
            "high_risk_signals": high_risk_signals,
            "step_count": step_count,
            "task_failed": task_failed,
            "confidence_replacement_count": replacement_count,
            "complete_task_block_count": complete_task_block_count,
            "set_level_soft_verify_count": set_level_soft_verify_count,
            "api_error_count": self._task_api_error_count,
            "python_error_count": self._task_python_error_count,
        }

        if self.learning_update_policy != "failure_or_high_risk":
            metadata["run_learning_update"] = True
            metadata["learning_update_reason"] = "policy_all"
            return True, "policy_all", metadata
        if task_failed is True:
            metadata["run_learning_update"] = True
            metadata["learning_update_reason"] = "failed_task"
            return True, "failed_task", metadata
        if high_risk_signals:
            metadata["run_learning_update"] = True
            metadata["learning_update_reason"] = "high_risk_success_or_unknown"
            return True, "high_risk_success_or_unknown", metadata
        if self.enable_periodic_learning_update and self.periodic_learning_update_every > 0:
            if (self.current_task_index + 1) % self.periodic_learning_update_every == 0:
                metadata["run_learning_update"] = True
                metadata["learning_update_reason"] = "periodic_safety_valve"
                return True, "periodic_safety_valve", metadata
        if self.skip_learning_on_clean_success and task_failed is False:
            metadata["run_learning_update"] = False
            metadata["learning_update_reason"] = "clean_low_risk_success"
            return False, "clean_low_risk_success", metadata
        metadata["run_learning_update"] = True
        metadata["learning_update_reason"] = "unknown_result_conservative"
        return True, "unknown_result_conservative", metadata

    def maybe_learning_update(self, evaluation_result=None) -> None:
        run_update, reason, metadata = self.should_run_learning_update(
            evaluation_result=evaluation_result,
            trajectory_stats={"step_count": self.step_number},
        )
        metadata["run_learning_update"] = bool(run_update)
        metadata["learning_update_reason"] = reason
        self._log_jsonl("learning_update_decisions.jsonl", metadata)
        if not run_update:
            print(f"[learning] skipped reflector/curator: {reason}")
            return
        print(f"[learning] running reflector/curator: {reason}")
        self.curator_call()
    
    def reflector_call(self):
        """
        Let the reflector generate insights based on the full conversation history, i.e. all messages and ground truths (if any).
        """
        filled_prompt = (
            self.reflector_prompt
            .replace("{{ground_truth_code}}", self.world_gt_code or "")
            .replace("{{test_report}}", self.test_report or "")
            .replace("{{generated_code}}", "See full conversation history below")
            .replace("{{generated_rationale}}", "See full conversation history below")
            .replace("{{spec_or_api_docs}}", "See full conversation history below")
            .replace("{{execution_error}}", "See full conversation history below")
            .replace("{{playbook}}", self.selected_playbook or self.playbook or "N/A")
            .replace("{{previous_reflection}}", "N/A")
        )
        
        # add full conversation history
        conversation_history = "\n\n=== FULL CONVERSATION HISTORY ===\n"
        for i, msg in enumerate(self.trimmed_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_history += f"[{i}] {role.upper()}: {content}\n\n"
        
        filled_prompt += conversation_history

        message_ = self.reflector_model.generate(messages=[{"role": "user", "content": filled_prompt}])
        reasoning_text = message_.get("content", "")
        if reasoning_text != "" and reasoning_text is not None:
            self.logger.show_message(role="user", message=reasoning_text, step_number=self.step_number)
        else:
            self.logger.show_message(role="user", message="[WARN] reasoning_text is empty or None", step_number=self.step_number)

        return reasoning_text
    
    def curator_call(self):
        """
        Let the curator update the playbook based on the full conversation history, i.e. all messages and reflections.
        """
        
        reasoning_text = None
        if self.use_reflector:
            reasoning_text = self.reflector_call()
        # Current playbook and question context
        current_playbook = self.selected_playbook or self.playbook or ""
        question_context = getattr(getattr(self, "world", None), "task", None)
        question_context = getattr(question_context, "instruction", "") if question_context else ""

        # add conversation history
        conversation_history = "\n\n=== FULL CONVERSATION HISTORY ===\n"
        for i, msg in enumerate(self.trimmed_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_history += f"[{i}] {role.upper()}: {content}\n\n"

        # Build curator prompt with explicit response format
        content = self.curator_prompt.format(
            initial_generated_code="See full conversation history below",
            final_generated_code="See full conversation history below",
            guidebook=reasoning_text,
            current_playbook=current_playbook,
            question_context=question_context,
            gt=self.world_gt_code
        )
        
        content += conversation_history

        self.curation_messages = [{"role": "user", "content": content}]
        curator_raw = self.curator_model.generate(messages=self.curation_messages)
        curator_response = curator_raw.get("content", "")

        # Parse JSON (must match explicit response schema: {"reasoning": str, "operations": [...]})
        operations_info = extract_json_from_text(curator_response, "operations")

        try: 
            # Strict validation
            if not operations_info:
                raise ValueError("Failed to extract valid JSON from curator response")

            if "reasoning" not in operations_info:
                raise ValueError("JSON missing required 'reasoning' field")
            if "operations" not in operations_info:
                raise ValueError("JSON missing required 'operations' field")

            if not isinstance(operations_info["reasoning"], str):
                raise ValueError("'reasoning' field must be a string")
            if not isinstance(operations_info["operations"], list):
                raise ValueError("'operations' field must be a list")

            # Only ADD operations supported
            allowed_sections = {
                "strategies_and_hard_rules",
                "apis_to_use_for_specific_information", 
                "useful_code_snippets_and_templates",
                "common_mistakes_and_correct_strategies",
                "problem_solving_heuristics_and_workflows",
                "verification_checklist",
                "troubleshooting_and_pitfalls",
                "others",
            }
            filtered_ops: list[dict] = []
            for i, op in enumerate(operations_info["operations"]):
                if not isinstance(op, dict):
                    raise ValueError(f"Operation {i} must be a dictionary")
                if "type" not in op:
                    raise ValueError(f"Operation {i} missing required 'type' field")
                if op["type"] != "ADD":
                    raise ValueError(f"Operation {i} has invalid type '{op['type']}'. Only 'ADD' operations are supported in this file")

                required_fields = {"type", "section", "content"}
                missing_fields = required_fields - set(op.keys())
                if missing_fields:
                    raise ValueError(f"ADD operation {i} missing fields: {list(missing_fields)}")
                # Enforce section whitelist
                section_name = str(op.get("section", "")).strip().lower().replace(" ", "_").replace("&", "and").rstrip(":")
                if section_name not in allowed_sections:
                    print(f"⏭️  Skipping operation {i}: disallowed section '{op.get('section')}' (normalized: '{section_name}'). Allowed: {sorted(allowed_sections)}")
                    continue
                filtered_ops.append(op)

            operations, curator_limit_metadata = sanitize_curator_operations(
                filtered_ops,
                self.playbook,
                max_new_bullets=self.max_curator_new_bullets_per_task,
                max_bullet_chars=self.max_curator_bullet_chars,
                max_total_chars=self.max_curator_update_chars,
                enable_dedup=self.enable_curator_dedup,
            )
            self._log_jsonl("curator_limits.jsonl", {
                "task_id": getattr(getattr(self, "world", None), "task_id", None),
                **curator_limit_metadata,
            })
            print(f"✅ Curator JSON schema validated successfully: {len(filtered_ops)} operations; kept {len(operations)}")
            # Apply curated updates
            self.playbook, self.next_global_id = apply_curator_operations(
                self.playbook, operations, self.next_global_id
            )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            print(f"❌ Curator JSON parsing failed: {e}")
            if curator_response is not None:
                print(f"📄 Raw curator response preview: {curator_response[:300]}...")
            else:
                print(f"📄 Raw curator response preview: None")
            
            print("⏭️  Skipping curator operation due to invalid JSON format")
            # Don't update playbook - continue with existing playbook    
        except Exception as e:
            print(f"❌ Curator operation failed: {e}")
            if curator_response is not None:
                print(f"📄 Raw curator response preview: {curator_response[:300]}...")
            else:
                print(f"📄 Raw curator response preview: None")
            
            print("⏭️  Skipping curator operation and continuing training")

        # Persist updated playbook
        with open(self.trained_playbook_file_path, "w") as file:
            file.write(self.playbook)

        if curator_response is not None:
            self.logger.show_message(role="user", message=curator_response, step_number=self.step_number)
        else:
            self.logger.show_message(role="user", message="[WARN] curator_response is None", step_number=self.step_number)
