# InvertedDoublePendulum-v5 mismatch: old behavior and fix

## Symptom

The Gymnasium and CasADi trajectories for `InvertedDoublePendulum-v5` diverged badly, even in free evolution with zero control. The single-pendulum comparison did not show the same issue, which pointed to a double-pendulum-specific bug rather than a generic plotting or rollout problem.

## What the old version was assuming

The old comparison path made four important assumptions that turned out to be wrong:

1. It hard-coded `frame_skip = 2` for the Gymnasium environment and therefore simulated the CasADi model with `dt = 0.02`.
2. Even after moving to Gymnasium's reported `dt = 0.05`, it advanced the CasADi model with one large RK4 step of `0.05` instead of reproducing MuJoCo's five internal `0.01` substeps.
3. It treated the raw MuJoCo double-pendulum state as if the second angle and angular velocity were already absolute.
4. The checked-in parameter JSON had stale cart-loss coefficients, with `b_slide = 500` and `b_fric = 500`, which are actuator-scale values and not the MuJoCo joint damping for the slider.

Any one of these will distort the comparison. The first and fourth were the biggest drivers of the catastrophic mismatch.

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

The fix was split into five parts:

1. The double-pendulum wrapper now exposes one explicit state boundary.
   - `gym_raw` means raw MuJoCo `[x, q1, q2_rel, x_dot, q1_dot, q2_rel_dot]`
   - `casadi_absolute` means `[x, theta1_abs, theta2_abs, x_dot, theta1_dot_abs, theta2_dot_abs]`
2. Wrapper reset and step now convert between these conventions explicitly instead of mixing them implicitly.
3. The comparison harness now reads Gymnasium's real `dt` from the environment instead of hard-coding it.
4. The CasADi rollout now reproduces Gymnasium's step semantics as five substeps of `0.01` rather than one large step of `0.05`.
5. The symbolic parameter file now uses the actual slider damping from MuJoCo (`b_slide = 0.05`) and no extra cart-friction drag term (`b_fric = 0.0`) for this comparison path.

## What changed numerically

After fixing the timestep, stepping, state-convention, and stale-parameter mismatch:

- the catastrophic `1e44` and `1e93` blow-ups disappeared
- short-horizon free rollouts dropped to about `0.09` max absolute error and about `0.023` overall RMSE
- short-horizon controlled rollouts dropped to about `0.16` max absolute error and about `0.033` overall RMSE

There is still some longer-horizon drift between Gymnasium and the symbolic model. After the fix, two diagnostics help interpret it:

- In free evolution, a tiny perturbation of about `1e-4 rad` to the initial Gymnasium angle causes visible divergence by roughly `1.5 s`, which shows the system is sensitive enough for small local errors to amplify quickly.
- If CasADi is re-initialized from the exact Gymnasium state at every free-evolution step and asked to predict only the next step, the one-step error stays small over the full `4 s` horizon.

So the remaining free-response drift now looks much more like chaotic amplification of small local discrepancies than another large interface bug.

## What to change in lecture notes

If your notes describe the double pendulum using the Gymnasium/MuJoCo raw state directly, they should make this distinction explicit:

- The raw simulator state stores the second hinge relative to the first hinge.
- Absolute second-link angles only appear after a manual conversion.
- The Gymnasium observation vector is feature-based (`sin/cos`) and is not the same thing as raw `qpos/qvel`.
- One Gymnasium environment step is five MuJoCo internal steps of `0.01 s`, so matching Gym means reproducing those substeps rather than using one large `0.05 s` integrator step.
- The slider damping in the MuJoCo model is `0.05`, not `500`, and the actuator gear should not be reused as a damping coefficient.

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
