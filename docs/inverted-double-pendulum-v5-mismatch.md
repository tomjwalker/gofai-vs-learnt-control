# InvertedDoublePendulum-v5 mismatch: old behavior and fix

## Symptom

The Gymnasium and CasADi trajectories for `InvertedDoublePendulum-v5` diverged badly, even in free evolution with zero control. The single-pendulum comparison did not show the same issue, which pointed to a double-pendulum-specific bug rather than a generic plotting or rollout problem.

## What the old version was assuming

The old comparison path made three important assumptions that turned out to be wrong:

1. It hard-coded `frame_skip = 2` for the Gymnasium environment and therefore simulated the CasADi model with `dt = 0.02`.
2. It treated the raw MuJoCo double-pendulum state as if the second angle and angular velocity were already absolute.
3. In the analysis script, it imported `src/environments/double_pendulum_dynamics_copy.py` and, in control mode, scaled the control by the actuator gear before calling dynamics code that already applies the gear internally.

Any one of these will distort the comparison. The first two were the main source of the big free-response mismatch.

## What Gymnasium/MuJoCo actually does

For `InvertedDoublePendulum-v5` in Gymnasium/MuJoCo:

- The MuJoCo model timestep is `0.01` seconds.
- The environment uses `frame_skip = 5`.
- The effective environment step is therefore `dt = 0.05` seconds.
- The raw MuJoCo state is:
  - `qpos = [x, q1, q2_rel]`
  - `qvel = [x_dot, q1_dot, q2_rel_dot]`
- The second hinge is stored relative to the first hinge in raw MuJoCo coordinates.
- The native Gymnasium observation is not the raw state. It uses sinusoidal features for the angles:
  - `[x, sin(q1), sin(q2_rel), cos(q1), cos(q2_rel), x_dot, q1_dot, q2_rel_dot, constraint_force]`

So if the CasADi model uses absolute second-link coordinates, the correct conversion is:

- `theta2_abs = q1 + q2_rel`
- `theta2_dot_abs = q1_dot + q2_rel_dot`

And the inverse conversion is:

- `q2_rel = theta2_abs - q1`
- `q2_rel_dot = theta2_dot_abs - q1_dot`

## Exact fix applied in the repo

The fix was split into four parts:

1. The double-pendulum wrapper now exposes one explicit state boundary.
   - `gym_raw` means raw MuJoCo `[x, q1, q2_rel, x_dot, q1_dot, q2_rel_dot]`
   - `casadi_absolute` means `[x, theta1_abs, theta2_abs, x_dot, theta1_dot_abs, theta2_dot_abs]`
2. Wrapper reset and step now convert between these conventions explicitly instead of mixing them implicitly.
3. The comparison harness now reads Gymnasium's real `dt` from the environment instead of hard-coding it.
4. The comparison script now uses the canonical `src/environments/double_pendulum_dynamics.py` module rather than the stale duplicate copy.

## What changed numerically

After fixing the timestep and state-convention mismatch:

- short-horizon free rollouts became close enough to use as regression tests
- short-horizon controlled rollouts also became close enough to use in pytest
- the catastrophic mismatch from the old plotting setup disappeared

There is still some longer-horizon drift between Gymnasium and the symbolic model. At this point that remaining drift looks like a genuine model-physics discrepancy, not the original interface bug.

## What to change in lecture notes

If your notes describe the double pendulum using the Gymnasium/MuJoCo raw state directly, they should make this distinction explicit:

- The raw simulator state stores the second hinge relative to the first hinge.
- Absolute second-link angles only appear after a manual conversion.
- The Gymnasium observation vector is feature-based (`sin/cos`) and is not the same thing as raw `qpos/qvel`.
- The correct environment step for `InvertedDoublePendulum-v5` in this repo is `0.05 s`, not `0.02 s`.

A clean teaching convention is:

- derive the equations in absolute coordinates if that is easier mathematically
- convert to and from Gym/MuJoCo raw state at one named boundary in code
- never mix raw and absolute coordinates inline inside plotting or reset logic

## Files changed for this fix

- `src/environments/wrappers.py`
- `src/utils/trajectory_comparison.py`
- `tests/environments/test_casadi_dynamics.py`
- `analysis/compare_casadi_vs_gym_double.py`

These now act as the source of truth for the corrected comparison path.