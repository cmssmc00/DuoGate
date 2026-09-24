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

## Installation

Use Python 3.11 and install both packages from this checkout. From the repository
root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-appworld.txt -e . -e 'experiments[simplified]'
export APPWORLD_ROOT="$PWD"
export APPWORLD_PROJECT_PATH="$PWD"
python -m appworld.cli install --repo
python -m appworld.cli download data
```

The encrypted `.bundle` files under `src/appworld/.source/` and
`generate/.source/` are upstream installation assets required by
`appworld install --repo`. They are not experiment outputs. A Git checkout that
stores these files through Git LFS requires Git LFS and `git lfs pull` before
installation. Downloaded data and unpacked protected files should remain local.

## API credentials

Supply your own credentials through environment variables or a local `.env`
created from `.env.example`. The example configuration uses the OpenAI-compatible
client:

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_BASE_URL='https://your-provider.example/v1'
```

Replace the endpoint with the provider that serves the selected model. No service
credentials or personal endpoints are distributed with this release.

## Run full DuoGate

`experiments/configs/DuoGate_AppWorld.jsonnet` uses `MiniMax-M2.7` as the example
model for the Agent, Reflector, Curator, and action assessor. Set `model_name` to
the exact model ID exposed by your provider. The configuration uses `test_normal`,
40 agent steps per task, and `control` / `risk_only` action assessment.

From the repository root, with the environment configured above:

```bash
export APPWORLD_CACHE="$(mktemp -d)"
python -m appworld.cli run DuoGate_AppWorld --num-processes 1
python -m appworld.cli evaluate DuoGate_AppWorld test_normal
```

Running the agent makes paid model calls. Evaluation consumes existing task
artifacts. Results are written under `experiments/outputs/DuoGate_AppWorld/`, and
the learned playbook is written to
`experiments/playbooks/duogate_trained_playbook.txt`.

The initial playbook always comes from
`experiments/playbooks/appworld_initial_playbook.txt`, which contains exactly the
eight original ACE entries. Learning can update the working playbook between
tasks within the same run. For another independent run, copy the configuration
under a new name, change both its trained-playbook and controller-log paths, and
use a fresh `APPWORLD_CACHE`. Do not reuse an existing output directory. Run tasks
serially when evaluating this sequential online-learning configuration.

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
a 60,000-character target; prefixed critical guidance can exceed that target.

## Attribution and licensing

The execution environment and packaging are derived from
[AppWorld](https://github.com/stonybrooknlp/appworld). The reflection and curation
workflow is adapted from [ACE-AppWorld](https://github.com/ace-agent/ace-appworld).
Upstream notices and package metadata are retained. See `LICENSE` and
`THIRD_PARTY_NOTICES.md` for attribution and the protected-bundle boundary.
