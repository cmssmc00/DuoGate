# DuoGate: AppWorld Implementation

This repository contains the AppWorld implementation of **DuoGate: Action and
Context Control for Reliable Language Agents**. The supplied configuration enables
the full method: execution-time action control, conditional reflection and
curation, admission checks, and reuse of the updated playbook across tasks.

This release covers AppWorld only. It contains source code, prompts, one full
DuoGate configuration, and the original eight-entry initial playbook. Benchmark
data, learned playbooks, experiment outputs, and ablation configurations are not
included. Internal `ace` module and registration names are retained for
compatibility with the upstream ACE-AppWorld implementation.

## Getting started

Follow the steps below in order to go from a fresh checkout to one evaluated
task, then optionally run the full benchmark. All commands below use a Bash-style
shell on Linux or macOS. On Windows, use a Linux shell in WSL2 for these commands;
they are not PowerShell commands.

### 1. Prepare the prerequisites

Install the following on the computer or server where you will run the agent:

- **Python 3.11**, including support for `venv`.
- **Git** and **[Git LFS](https://github.com/git-lfs/git-lfs#installing)**.
  Git LFS must be installed as a program before you run `git lfs install`;
  that command initializes its Git configuration.
- Network access to GitHub, Python package downloads, the AppWorld data download,
  and your model API provider.
- An API key and an OpenAI-compatible endpoint that serves your chosen model.

Check that the tools are available in your terminal:

```bash
git --version
git lfs version
python3.11 --version
```

The last command should report Python 3.11.x. If a command is not found, install
that tool before continuing. The supplied configuration calls a remote model API;
you do not need to download model weights or have a local GPU for this workflow.

### 2. Download the anonymous source archive

Open the anonymous repository linked in the paper and click **Full repo ZIP**
in the upper-right corner.

The following example assumes that `DuoGate-56CF.zip` was downloaded to
`~/Downloads`. Adjust the download directory if needed, and extract into
a new directory:

```bash
cd ~/Downloads
unzip DuoGate-56CF.zip -d DuoGate-anonymous
cd DuoGate-anonymous
```

You should now see `README.md`, `pyproject.toml`, `src/`, and `experiments/`.
Run all remaining commands from this project root.

This archive-based installation does not require Git or Git LFS.
Continue with Step 3 to create the Python environment and install both
local packages.

### 3. Install both Python packages

Create and activate an isolated Python environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

Install both packages from this checkout with the supplied dependency constraints:

```bash
python -m pip install -c constraints-appworld.txt -e . -e 'experiments[simplified]'
```

Here, `-e .` installs the local AppWorld package, and
`-e 'experiments[simplified]'` installs the experiment code and its required
extras. When updating an existing environment, rerun this complete constrained
installation command so that both packages' dependencies are resolved together.

### 4. Prepare the environment and download task data

With the virtual environment active and your terminal still in the repository root:

```bash
export APPWORLD_ROOT="$PWD"
export APPWORLD_PROJECT_PATH="$PWD"
python -m appworld.cli install --repo
python -m appworld.cli download data
```

The `install --repo` command unpacks the protected AppWorld installation assets.
The `download data` command downloads and extracts the benchmark data into
`data/`; the task data is not included in this source repository. The download
command replaces an existing `data/` directory, so it is intended for initial
setup or a deliberate data refresh, not every run.

Verify the installation:

```bash
python -m pip check
python -m appworld.cli verify tests
```

Before running the agent, confirm that `pip check` reports no broken requirements
and both verification phases report `All tests passed.` These checks validate
the environment; they do not run the DuoGate agent against a model API.

### 5. Configure API credentials and the model

Supply your own credentials through environment variables:

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_BASE_URL='https://your-provider.example/v1'
```

Both values above are placeholders. Replace them with a valid key and the
OpenAI-compatible API endpoint for your provider. The `OPENAI_` variable names
refer to the client interface; the model can be served by another provider that
supports that interface.

Alternatively, copy `.env.example` to a local `.env` and fill in the key and
endpoint there. Existing exported environment variables take precedence over
values loaded from `.env`. No service credentials or personal endpoints are
distributed with this release.

Open `experiments/configs/DuoGate_AppWorld.jsonnet` in a text editor and set the
model ID near the top:

```jsonnet
local model_name = "MiniMax-M2.7";
```

Keep this value only if your endpoint serves that exact ID; otherwise replace it
with the exact model ID exposed by your provider. This one variable is shared by
the Agent, Reflector, Curator, and action assessor. The API key, endpoint, and
model ID must match a service you can access.

## Run and evaluate one task first

Use a separate experiment name for the first trial so its outputs and learned
playbook do not share paths with a later full run. After configuring the model
above, create a trial configuration:

```bash
python - <<'PY'
from pathlib import Path

source = Path("experiments/configs/DuoGate_AppWorld.jsonnet")
target = Path("experiments/configs/DuoGate_Smoke.jsonnet")
config = source.read_text()
config = config.replace(
    "/duogate_trained_playbook.txt", "/duogate_smoke_playbook.txt"
).replace(
    "/experiments/outputs/DuoGate_AppWorld/confidence_logs",
    "/experiments/outputs/DuoGate_Smoke/confidence_logs",
)
with target.open("x") as file:
    file.write(config)
print(f"Created {target}")
PY
```

This copies the full DuoGate configuration and changes its trained-playbook and
controller-log paths. It deliberately refuses to overwrite an existing trial
configuration. To repeat an independent trial, use a new experiment name and
matching output paths as described below.

Select the first task from the development split and run only that task:

```bash
DUOGATE_TASK_ID="$(python -c 'from appworld.task import load_task_ids; print(load_task_ids("dev")[0])')"
echo "Running task: $DUOGATE_TASK_ID"

export APPWORLD_CACHE="$(mktemp -d)"
python -m appworld.cli run DuoGate_Smoke --task-id "$DUOGATE_TASK_ID" --num-processes 1
```

The explicit `--task-id` selects just that task, overriding the dataset selection
in the copied configuration. **Running the agent makes paid model calls.** Keep
`--num-processes 1` for this sequential online-learning configuration.

In the same terminal, evaluate the task that just ran:

```bash
python -m appworld.cli evaluate DuoGate_Smoke on_only --task-id "$DUOGATE_TASK_ID"
```

`on_only` is the exact special value required by the evaluator when
`--task-id` is supplied. Evaluation reads existing task artifacts rather than
running the agent again. A completed run and a generated evaluation report show
that the workflow is connected; check the report to determine whether the agent
actually solved the task. Single-task results are not full benchmark results,
and the scenario-level completion metric is not meaningful for this trial.

| Output | Location |
| --- | --- |
| Task artifacts and logs | `experiments/outputs/DuoGate_Smoke/tasks/<task_id>/` |
| Evaluation reports (`.json` and `.txt`) | `experiments/outputs/DuoGate_Smoke/evaluations/` |
| Action-controller logs | `experiments/outputs/DuoGate_Smoke/confidence_logs/` |
| Learned playbook, when persisted by a learning update | `experiments/playbooks/duogate_smoke_playbook.txt` |

A clean success can skip learning, so the absence of a newly written learned
playbook alone does not mean that the task failed.

## Run full DuoGate

After the single-task trial, use `DuoGate_AppWorld` for the full run.
`experiments/configs/DuoGate_AppWorld.jsonnet` uses `test_normal`, 40 agent steps
per task, and `control` / `risk_only` action assessment. Without `--task-id`,
the command runs the entire configured dataset, not one example task.

From the repository root, with the environment and credentials configured above:

```bash
export APPWORLD_CACHE="$(mktemp -d)"
python -m appworld.cli run DuoGate_AppWorld --num-processes 1
python -m appworld.cli evaluate DuoGate_AppWorld test_normal
```

Results are written under `experiments/outputs/DuoGate_AppWorld/`, including
evaluation reports in its `evaluations/` subdirectory. When learning persists
the working playbook, it writes to
`experiments/playbooks/duogate_trained_playbook.txt`.

The initial playbook always comes from
`experiments/playbooks/appworld_initial_playbook.txt`, which contains exactly the
eight original ACE entries. Retrieval selects only existing playbook content;
it does not prepend an additional static rule summary. Learning can update the
working playbook between tasks within the same run. For another independent run, copy the configuration
under a new name, change both its trained-playbook and controller-log paths, and
use a fresh `APPWORLD_CACHE`. Do not reuse an existing output directory. Run tasks
serially when evaluating this sequential online-learning configuration.

## Returning in a new terminal

Virtual-environment activation and exported variables apply to the current shell.
When you return later, change into your checkout and restore the environment:

```bash
cd /absolute/path/to/DuoGate
source .venv/bin/activate
export APPWORLD_ROOT="$PWD"
export APPWORLD_PROJECT_PATH="$PWD"
```

Replace the path with your actual checkout location. Restore your API environment
variables from step 5, or use the local `.env` file. Set a fresh
`APPWORLD_CACHE` before starting an independent experiment, and use a new
experiment name and output paths if the previous ones already contain results.
You do not need to repeat the initial data download every time you open a terminal.

## Common setup issues

| Symptom | What to check |
| --- | --- |
| `git: 'lfs' is not a git command` | Install the Git LFS program, then rerun `git lfs install` and `git lfs pull` in the checkout. |
| `python3.11` is not found or `venv` cannot be created | Install Python 3.11 and its virtual-environment support before creating `.venv`. |
| Missing bundles or an unpacking error during `install --repo` | Check that you are in the repository root and that `git lfs pull` completed successfully. |
| Missing data or an undefined `APPWORLD_PROJECT_PATH` | Restore the root variables and confirm that the data download completed in this checkout. |
| Import errors or broken requirements | Activate `.venv`, rerun the complete constrained installation command, and run `python -m pip check`. |
| Authentication, model-not-found, or API connection errors | Check the API key, the provider's endpoint, the exact model ID, and network access. |
| Evaluation rejects the dataset name for one task | Use `on_only --task-id "$DUOGATE_TASK_ID"` for single-task evaluation. |

## Implementation map

| Stage | Source |
| --- | --- |
| Task loop and environment execution | `experiments/code/ace/adaptation_agent.py` |
| Agent proposals, learning eligibility, reflection, and curation | `experiments/code/ace/adaptation_react.py` |
| Action assessment, local checks, recovery, and feedback | `experiments/code/ace/appworld_confidence.py` |
| Playbook retrieval, candidate filtering, and merging | `experiments/code/ace/playbook.py` |
| Provider calls and model-call logging | `experiments/code/ace/lite_llm_generator.py` |
| Agent, Reflector, Curator, and assessor prompts | `experiments/prompts/` |

The source defaults select updates for failures or recognized risk, skip clean
successes, admit at most two new entries per update, and apply 800-character
per-entry and 1,600-character per-update limits. The retrieved playbook view has
a 60,000-character target.

The Generator retains the interface instructions, demonstrations, and eight
original key instructions. It does not require a candidate table or an extra
confidence/execution-policy block. Confidence assessment and recovery remain in
the runtime controller.

Collection checks use explicit candidate eligibility/action evidence and
unambiguous current task instructions. App/entity names select identifier fields,
not default operations. Contact membership, pending/open status, and note metadata
alone do not determine eligibility or an operation; unresolved sets/actions remain
unknown. The payment-direction checks, collection mismatch checks, and context
admission checks are retained.

## Attribution and licensing

The execution environment and packaging are derived from
[AppWorld](https://github.com/stonybrooknlp/appworld). The reflection and curation
workflow is adapted from [ACE-AppWorld](https://github.com/ace-agent/ace-appworld).
Upstream notices and package metadata are retained. See `LICENSE` and
`THIRD_PARTY_NOTICES.md` for attribution and the protected-bundle boundary.

The execution environment and packaging are derived from
[AppWorld](https://github.com/stonybrooknlp/appworld). The reflection and curation
workflow is adapted from [ACE-AppWorld](https://github.com/ace-agent/ace-appworld).
Upstream notices and package metadata are retained. See `LICENSE` and
`THIRD_PARTY_NOTICES.md` for attribution and the protected-bundle boundary.
