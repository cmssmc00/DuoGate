# Third-party notices

## AppWorld

The `src/appworld/` environment, public tests, generation utilities, encrypted
installation bundles, and associated packaging originate from
[AppWorld](https://github.com/stonybrooknlp/appworld).

The public portion is distributed under the Apache License 2.0, reproduced in
`LICENSE`. AppWorld's protected portion is distributed in encrypted `.bundle`
files with the additional requirement that public redistribution of that portion
or its derivatives must also use an encrypted format. Preserve the notices
installed from those bundles, and do not publish the locally unpacked protected
files or downloaded benchmark data as ordinary source files.

## ACE-AppWorld

The agent scaffolding, reflection and curation workflow, prompts, and original
initial playbook are adapted from
[ACE-AppWorld](https://github.com/ace-agent/ace-appworld).

The initial playbook contains the original eight entries:
`shr-00001`, `shr-00005`, `shr-00006`, `api-00004`, `psw-00002`, `psw-00007`,
`misc-00003`, and `misc-00008`. The reference is the
[initial playbook at commit 9f3e92155345](https://github.com/ace-agent/ace-appworld/blob/9f3e92155345/experiments/playbooks/appworld_initial_playbook.txt).

DuoGate adds the AppWorld action controller and connects its feedback to update
eligibility, candidate admission, and guidance reuse. Retaining upstream module
names does not identify the supplied full DuoGate configuration as the ACE
baseline.
