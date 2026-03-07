# gofai-vs-learnt-control

## Setup

Create the project environment and install all runtime and development dependencies:

```powershell
uv sync --dev
```

Run the main verification checks:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Gymnasium vs CasADi Comparisons

The repo includes student-facing comparison scripts for both the single and double pendulum.
Each script can:

- plot Gymnasium and CasADi state trajectories
- show a live animation
- save a GIF or MP4 animation
- write CSV and JSON comparison artifacts under `analysis/artifacts/`

### Single Pendulum

Recommended module:

```powershell
uv run python -m analysis.compare_casadi_vs_gym_single --mode free --steps 100
uv run python -m analysis.compare_casadi_vs_gym_single --mode control --steps 100 --control-amplitude 3.0
```

Backwards-compatible alias:

```powershell
uv run python -m analysis.compare_casadi_vs_gym --mode free --steps 100
```

Animation commands:

```powershell
uv run python -m analysis.compare_casadi_vs_gym_single --mode free --steps 100 --no-plot --animate
uv run python -m analysis.compare_casadi_vs_gym_single --mode control --steps 100 --control-amplitude 3.0 --no-plot --animate
```

Save a GIF or MP4:

```powershell
uv run python -m analysis.compare_casadi_vs_gym_single --mode free --steps 100 --no-plot --animation-path analysis/artifacts/ip_free_100.gif
uv run python -m analysis.compare_casadi_vs_gym_single --mode control --steps 100 --control-amplitude 3.0 --no-plot --animation-path analysis/artifacts/ip_control_100.gif
```

### Double Pendulum

Recommended module:

```powershell
uv run python -m analysis.compare_casadi_vs_gym_double --mode free --steps 80
uv run python -m analysis.compare_casadi_vs_gym_double --mode control --steps 80 --control-amplitude 0.1
```

Animation commands:

```powershell
uv run python -m analysis.compare_casadi_vs_gym_double --mode free --steps 80 --no-plot --animate
uv run python -m analysis.compare_casadi_vs_gym_double --mode control --steps 80 --control-amplitude 0.1 --no-plot --animate
```

Save a GIF or MP4:

```powershell
uv run python -m analysis.compare_casadi_vs_gym_double --mode free --steps 80 --no-plot --animation-path analysis/artifacts/idp_free_80.gif
uv run python -m analysis.compare_casadi_vs_gym_double --mode control --steps 80 --control-amplitude 0.1 --no-plot --animation-path analysis/artifacts/idp_control_80.gif
```

Useful optional flags for both scripts:

```powershell
--plot-path analysis/artifacts/plot.png
--fps 20
--artifact-dir analysis/artifacts
```

### Notes

- The single pendulum uses `dt = 0.04 s`, so `100` steps is about `4` seconds.
- The double pendulum uses `dt = 0.05 s`, so `80` steps is about `4` seconds.
- The single pendulum should remain visually aligned over the full comparison horizon.
- The double pendulum now matches closely on short horizons and in one-step prediction tests.
- Longer free `IDP` rollouts can still diverge because the system is highly sensitive to small local differences.
- Longer controlled `IDP` rollouts will also diverge once the cart reaches the slider limit at about `x = +/-1`, because MuJoCo then applies joint-limit constraint forces that are not included in the current symbolic model.
