# Scale-test pre-requirements

This folder contains small bootstrap scripts that can be executed on remote hosts before running a scale test.

Typical use-case: install system/third-party tools (e.g. Miniforge) that are needed before `conda run ...` or `pip install -r ...` can work.

## Miniforge

- Script: `install_miniforge3_linux_x86_64.sh`

It downloads and installs Miniforge3, appends an `export PATH=...` line into `~/.bashrc`, and exports `PATH` for the current session.

Configuration (multi-host SSH dispatch):

- Set `run.servers[*].pre_requirements_file` to `scripts/scale-test/pre-requirements/install_miniforge3_linux_x86_64.sh`.
- Optionally set `MINIFORGE_PREFIX` via your remote environment if you need a non-default install prefix.
