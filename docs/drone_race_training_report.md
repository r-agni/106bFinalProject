# DroneRace Training Report

This report explains the current DroneRace reinforcement-learning setup, the
training loop, the reward and curriculum strategy, the adaptive entropy logic,
and the main design decisions behind them.

The working tree is treated as the source of truth. The active training metrics
below are a point-in-time snapshot from
`.logs/DroneRace-rewardfix-12h-simple-2048-20260419-162107.log`; that run was
still active when inspected, so the final numbers may move.

## Source Files

The main implementation and configuration live in:

| File | Role |
| --- | --- |
| `omni_drones/envs/drone_race/drone_race.py` | DroneRace environment, observations, gate detection, reward, termination, curriculum stats |
| `cfg/task/DroneRace.yaml` | Track layout, environment count, episode length, reward scales, crash thresholds |
| `cfg/algo/DroneRace.yaml` | PPO hyperparameters, actor/critic network sizes, adaptive entropy config |
| `scripts/train.py` | Hydra config merge, collector loop, W&B logging, adaptive entropy updates, checkpointing |
| `omni_drones/learning/ppo/ppo.py` | PPO policy, actor/critic modules, GAE, PPO update step |

## Executive Summary

DroneRace is currently trained with PPO on a 13-gate race track. The policy is
being trained first for accurate ordered gate completion and only later for
speed. This is an intentional design choice: racing fast before the drone can
reliably pass gates creates unstable behavior and often rewards fly-bys,
collisions, or shortcut-like progress.

The current active run uses 2048 parallel Isaac Sim environments, 256 policy
steps per rollout, and therefore 524,288 frames per PPO update. The default task
configuration still lists 512 environments, but the active command overrides
that to 2048.

At the latest inspected complete metrics block, the policy reliably reaches the
first elevated section and averages about 6.8 gates per episode, but has not yet
completed a lap. The current bottleneck is the transition after the first
elevated gate: YAML gate 7 to YAML gate 8, which is a same-XY vertical drop with
a direction reversal.

## Environment And Track Setup

The task uses `DroneRaceEnv` with an Iris quadrotor controlled through a
`RateController`. The configured track has 13 gates:

- Gates 1 through 12 are the actual course gates.
- Gate 13 duplicates gate 1 and acts as the lap-closure target.
- Gates 7 and 11 are elevated.
- Gates 7 -> 8 and 11 -> 12 share the same XY position but differ in altitude
  and gate facing. These are deliberately hard drop/reversal sections.

Important environment settings:

| Setting | Value | Why it matters |
| --- | ---: | --- |
| `sim.dt` | `0.002` | Physics integration timestep |
| `sim.substeps` | `5` | Policy acts every `0.002 * 5 = 0.01s` |
| Policy rate | `100 Hz` | Fast enough for rate control and racing maneuvers |
| `max_episode_length` | `2000` steps | 20 seconds per episode |
| Default `num_envs` | `512` | Stable default for training |
| Active-run `num_envs` | `2048` | Higher throughput and broader rollout diversity |
| `train_every` | `256` | 2.56 seconds of experience per env per PPO batch |

The track is wide enough for early learning but contains several failure modes:
long turns, elevated gates, same-position vertical transitions, and reversed gate
normals. Because the course is ordered, skipping ahead does not count; the
current target gate must be crossed.

## Observation Design

The policy observes a 33-dimensional vector per drone. It is compact and mostly
relative rather than absolute.

| Observation component | Size | Purpose |
| --- | ---: | --- |
| Body-frame linear velocity | 3 | Tells the policy how it is moving relative to its body axes |
| Rotation matrix | 9 | Orientation without quaternion sign discontinuity |
| Angular velocity | 3 | Stabilization and rate-awareness |
| Previous action | 4 | Helps the policy learn smoother control |
| Distance to gate center | 1 | Gives the policy the norm directly |
| Current gate relative position in drone frame | 3 | Main navigation vector |
| Next-to-next gate position in current/next gate frame | 3 | Lets the policy anticipate turns |
| Current gate orientation, first two world-frame columns | 6 | Encodes gate facing and lateral direction |
| Normalized gate index | 1 | Sim-only signal for gate-specific behavior |

Absolute position is intentionally not included. The policy instead receives the
relative geometry it needs to fly the current segment. The normalized gate index
is a useful sim-only shortcut because some gates require special behavior, such
as climbing, dropping, or reversing.

The observation points to the gate center, not the gate origin. This matters
because gate assets have their origin at the bottom center. The environment
adds `gate_height / 2` in the gate frame to align observations, rewards, and
crossing detection around the actual aperture center.

## Action And Control Design

The policy emits a 4D action:

| Action part | Meaning |
| --- | --- |
| First 3 values | Desired body rates |
| Last value | Desired thrust command |

The actor samples from an independent Gaussian distribution. Before execution,
the environment clamps the sampled action to `[-1, 1]`. The `RateController`
then maps the clamped command to body-rate targets and thrust, then to rotor
commands.

This gives the policy a relatively high-level control interface. PPO does not
need to learn raw motor mixing from scratch, but it still needs to learn racing
timing, turns, climb/drop maneuvers, and gate crossing.

The action clamp is an important safety fix: without it, Gaussian samples can be
arbitrarily large even if the action spec is bounded. Large commands can produce
runaway body-rate requests and actuator saturation. The tradeoff is that PPO
still computes log probabilities for the original sampled action while the
simulator executes the clamped action. This creates a mild policy-gradient
mismatch, especially when entropy is high and many samples hit the clamp.

## PPO Training Loop

Training is handled by `scripts/train.py` and `omni_drones/learning/ppo/ppo.py`.

The loop is:

1. Hydra loads `scripts/train.yaml`.
2. `task=DroneRace` loads `cfg/task/DroneRace.yaml`.
3. `DroneRace.yaml` points to `ppo_cfg: DroneRace.yaml`, so
   `cfg/algo/DroneRace.yaml` is merged over the base PPO config.
4. Isaac Sim creates the vectorized `DroneRaceEnv`.
5. A `SyncDataCollector` gathers `num_envs * train_every` frames.
6. Episode stats are aggregated and logged to W&B.
7. Adaptive entropy may update `policy.entropy_coef`.
8. `policy.train_op(...)` runs PPO updates.
9. Checkpoints are saved every configured interval.

With the active 2048-env run:

| Quantity | Value |
| --- | ---: |
| Envs | `2048` |
| Rollout length | `256` steps |
| Frames per PPO batch | `524,288` |
| PPO epochs | `10` |
| Minibatches per epoch | `16` |
| Samples per minibatch | `32,768` |
| Optimizer updates per rollout | `160` |

With the default 512-env config:

| Quantity | Value |
| --- | ---: |
| Envs | `512` |
| Rollout length | `256` steps |
| Frames per PPO batch | `131,072` |
| Samples per minibatch | `8,192` |

The long rollout length is deliberate. A 256-step rollout covers 2.56 seconds
per env at 100 Hz, which is enough to include several meaningful gate approach
and crossing events once the policy starts moving.

## PPO Algorithm Details

The actor and critic are both MLP-based.

| Module | Current config |
| --- | --- |
| Actor hidden units | `[256, 256]` |
| Critic hidden units | `[256, 256, 128]` |
| Activation | `leaky_relu` |
| Layer norm | Enabled |
| Actor LR | `3e-4` |
| Critic LR | `3e-4` |
| Optimizer | Adam |
| Gradient clip | `5.0` |

The actor outputs the mean and state-independent standard deviation of an
independent normal action distribution. The critic predicts a scalar value.

Key PPO settings:

| PPO setting | Value | Reason |
| --- | ---: | --- |
| `clip_param` | `0.2` | Limits policy update size and protects against destructive jumps |
| `gamma` | `0.998` | Long horizon for delayed gate and lap rewards |
| `gae_lambda` | `0.95` | Bias/variance tradeoff for advantage estimates |
| `ppo_epochs` | `10` | More reuse of expensive Isaac Sim rollouts |
| `num_minibatches` | `16` | Large, stable minibatches with vectorized envs |
| `max_grad_norm` | `5.0` | Prevents unstable gradient spikes |
| Critic loss | Huber, delta `10.0` | More robust to outlier returns during chaotic early training |

The PPO objective uses the usual clipped surrogate:

```text
ratio = exp(new_log_prob - old_log_prob)
surrogate = min(adv * ratio, adv * clamp(ratio, 1 - clip, 1 + clip))
policy_loss = -mean(surrogate)
```

Advantages are computed with generalized advantage estimation (GAE), then
normalized before training. Returns are also normalized through `ValueNorm1`
before critic regression. This helps because reward scale can vary dramatically
between early crashing policies and later gate-crossing policies.

One caveat: the current PPO code builds its GAE mask from `next/terminated`.
In this environment, `terminated` is only crash termination. Timeouts and lap
completion are represented through `done`/`truncated`. That means successful lap
completion is not treated as terminal for value bootstrap in PPO. This has not
affected the current run yet because full-lap completion is still zero, but it
should be fixed once successes start appearing.

## Privileged Actor And Critic

Both `priv_actor` and `priv_critic` are enabled.

The environment exposes drone intrinsics such as mass, inertia, center of mass,
rotor force constants, moment constants, rotor time constants, and drag
coefficient. With privileged mode enabled, those intrinsics are encoded and
concatenated with observation features.

Why this choice makes sense here:

- The project is optimizing sim performance, not sim-to-real transfer.
- The drone model is fixed, so intrinsics are not unrealistic within this setup.
- The critic can produce better value estimates when it knows physical
  parameters.
- Better critic estimates improve GAE advantages and can speed up PPO.

Tradeoff:

- If the goal changes to real-world deployment or policy robustness across
  random vehicles, the actor should probably not receive privileged intrinsics.
  Keeping `priv_critic=true` and `priv_actor=false` would be a cleaner
  asymmetric-training setup.

## Adaptive Entropy

Entropy controls exploration. In PPO, the entropy term is added as:

```text
entropy_loss = -entropy_coef * mean(entropy)
loss = policy_loss + entropy_loss + value_loss
```

Because the optimizer minimizes `loss`, a negative entropy loss encourages the
policy to keep its action distribution broad. Higher `entropy_coef` means more
random exploration. Lower `entropy_coef` means the policy becomes more
deterministic and exploits what it has learned.

The static config contains `entropy_coef: 0.001`, but adaptive entropy is
enabled, so `scripts/train.py` overwrites `policy.entropy_coef` at runtime.

Adaptive entropy settings:

| Setting | Value | Meaning |
| --- | ---: | --- |
| `phase0_coef_max` | `0.012` | High exploration while learning accurate gate completion |
| `phase0_coef_min` | `0.002` | Lower exploration once phase-0 accuracy improves |
| `phase1_rebump_coef` | `0.010` | Re-increase exploration when speed shaping unlocks |
| `phase1_coef_min` | `0.0005` | Low exploration for mature fast policies |
| `speed_ema_alpha` | `0.05` | EMA smoothing for speed performance |
| `speed_signal_low` | `0.05` | Low gates-per-second baseline |
| `speed_signal_high` | `0.60` | High gates-per-second target |
| `max_delta_per_update` | `0.0005` | Slew-rate limit for entropy coefficient changes |

Phase 0 behavior:

```text
accuracy_norm = curriculum_accuracy_ema / phase_speed_unlock_accuracy_rate
target_coef = phase0_coef_max
              - accuracy_norm * (phase0_coef_max - phase0_coef_min)
```

In words: while lap-completion accuracy is low, keep entropy high so the policy
continues exploring ways through the gates. As full-lap completion EMA rises
toward the unlock threshold, gradually lower entropy so the policy becomes more
consistent.

Phase 1 behavior:

1. When curriculum phase changes from 0 to 1, entropy is immediately rebumped to
   `phase1_rebump_coef`.
2. After that, the code tracks gates-per-second with an EMA.
3. As gates-per-second rises from `speed_signal_low` to `speed_signal_high`,
   entropy decays toward `phase1_coef_min`.

Why rebump in phase 1? The task objective changes. Phase 0 rewards accurate
completion. Phase 1 adds stronger time pressure. A policy that is accurate but
slow may need to discover new faster racing lines, so exploration is temporarily
increased again.

The slew-rate limit prevents the entropy coefficient from jumping around due to
noisy batch metrics. This is especially important because each batch contains
many environments but still only a small slice of the learning process.

Current observed entropy state:

- `curriculum/phase = 0.0`
- `curriculum/accuracy_ema = 0.0`
- `entropy/current_coef = 0.012`
- `entropy/target_coef = 0.012`

So adaptive entropy is currently doing exactly what it was designed to do:
hold exploration at the phase-0 maximum because no full laps have been achieved.

## Reward Design

The reward is built to prioritize ordered gate completion first and speed second.

### Positive rewards

| Reward | Scale | Purpose |
| --- | ---: | --- |
| Path-projection progress | `1.0` | Dense guidance along the segment between previous and target gate |
| Gate passage | `60.0` | Strong sparse reward for crossing the correct target gate |
| Ordered sequence | `20.0 * streak` | Increasing reward for chaining gates in order |
| Lap completion | `500.0` | Makes finishing the full course the primary objective |
| Lap speed bonus | up to `300.0` | Rewards earlier completion, but only in phase 1 |

The path-projection term is inspired by racing-line progress rewards: it
projects the drone's step displacement onto the vector from the previous gate
center to the current target gate center. This gives smooth dense feedback
without directly rewarding raw speed.

The scale was reduced to `1.0` in the current working tree. That decision is
important: if dense progress is too large, the policy can earn large reward by
moving along the rough track direction while missing apertures. Sparse gate
rewards now dominate phase 0, which aligns the objective with the actual task.

The sequence bonus increases with the number of gates already crossed in the
episode. This makes later ordered gates more valuable and pushes the policy
toward complete laps instead of treating each gate as an isolated event.

### Penalties

| Penalty | Scale | Purpose |
| --- | ---: | --- |
| Elevated-gate altitude mismatch | `0.5` | Encourages matching high gate centers |
| Near-gate retreat | `0.5` | Penalizes moving away when already close |
| Gate-centering miss | `0.03` | Discourages fly-bys near the gate plane |
| Angular-rate penalty | `0.1`, decays | Stabilizes early training |
| Action smoothness | `0.002` | Reduces abrupt actuator changes |
| Crash penalty | `80.0` | Penalizes collisions, ground hits, and distance divergence |

The altitude penalty is targeted at elevated gates. The track includes segments
where the required motion is strongly vertical, and basic path projection can be
weak or ambiguous there. A bounded altitude mismatch penalty nudges the policy
toward the gate center height without creating a huge reward exploit.

The gate-centering penalty is active near the target gate plane. It uses the
drone's position in the gate frame and penalizes lateral/vertical miss. This
protects against a common shaping failure: moving forward along the course but
not through the aperture.

The near-gate retreat penalty only fires within 6 meters of the target gate.
Instead of paying the drone for every tiny distance improvement, it subtracts
reward when the drone backs away after entering the close-approach region. This
is less exploitable than a pure distance-reduction bonus.

The angular-rate penalty decays over `100_000_000` frames. Early on, it helps
prevent violent spinning. Later, it fades so a mature racing policy is not
over-penalized for aggressive maneuvers.

## Gate Detection

Gate crossing uses the drone position in the current gate frame:

1. Track previous and current gate-frame `x`.
2. Detect plane crossing from negative `x` to positive `x`.
3. Require the current `y` and `z` to be inside the gate aperture.
4. If successful, advance `gate_indices`.

This is simple and fast. It also enforces ordered course completion because only
the current target gate can be crossed.

Tradeoff: endpoint sign-change detection can miss some fast or awkward
crossings. A segment-based plane intersection would be more robust because it
would check whether the line segment between previous and current drone
positions intersects the gate aperture.

## Curriculum And Termination

The curriculum currently has two phases:

| Phase | Meaning |
| --- | --- |
| `0` | Accuracy-first: learn ordered full-lap completion |
| `1` | Speed-shaping: keep completion objective and add lap-time pressure |

In phase 0, resets always place the drone behind gate 0. This forces the policy
to learn the full ordered course from the true start rather than overfitting to
random partial starts.

Phase 1 unlocks only when both conditions are true:

```text
frames_in_phase >= curriculum_min_phase_frames
lap_completion_rate_ema >= phase_speed_unlock_accuracy_rate
```

With current config:

```text
curriculum_min_phase_frames = 2_000_000
phase_speed_unlock_accuracy_rate = 0.40
curriculum_ema_alpha = 0.01
```

The frame floor is already satisfied in the active run. The blocking condition
is full-lap completion EMA, which remains zero.

Termination conditions:

| Condition | Meaning |
| --- | --- |
| `track_completed` | Full ordered lap completed |
| `progress_buf >= max_episode_length` | 20-second timeout |
| Physical contact | Base-link contact force above threshold |
| Ground crash | Drone `z < crash_z_min` |
| Distance crash | Too far from current target gate after grace window |

The distance-crash grace window lasts `100` steps after spawn or gate crossing.
This prevents false crashes immediately after a target gate advances, because
the next gate can be many meters away.

## Logging And Diagnostics

The training loop logs raw stats and easier-to-read derived metrics.

Important racing metrics:

- `race/gates_passed_per_ep`
- `race/lap_completion_rate`
- `race/mean_speed_ms`
- `race/max_speed_ms`
- `race/furthest_gate_reached`
- `race/final_dist_to_gate_m`
- `race/gates_per_second`

Important reward diagnostics:

- `reward/progress_cumul`
- `reward/gates_cumul`
- `reward/penalties_cumul`
- `reward/centering_cumul`
- `reward/total_return`
- Reward fractions such as `reward/gates_fraction`

Important crash diagnostics:

- `crash/total_rate`
- `crash/ground_rate`
- `crash/contact_rate`
- `crash/distance_rate`

Important curriculum and entropy diagnostics:

- `curriculum/phase`
- `curriculum/accuracy_ema`
- `curriculum/furthest_gate_ema`
- `entropy/current_coef`
- `entropy/target_coef`
- `entropy/rebump_applied`

Per-gate crossing metrics are especially useful:

```text
gates/gate_00_crosses
gates/gate_01_crosses
...
gates/gate_11_crosses
```

These act like a heatmap of where learning fails. If all early gate rates are
near 1.0 but one later gate is zero, the bottleneck is localized.

The training script also supports resume frame offsets. If the checkpoint path
looks like `checkpoint_<frames>.pt`, `scripts/train.py` can initialize the
collector's frame count from that value. This keeps resumed W&B plots aligned
with total training progress instead of restarting frame count at zero.

## Current Observed Behavior

Latest complete snapshot from
`.logs/DroneRace-rewardfix-12h-simple-2048-20260419-162107.log`, inspected after
the log had reached approximately epoch 660:

| Metric | Snapshot value |
| --- | ---: |
| Complete stats block frame | `344,457,216` |
| Latest train line frame nearby | `346,030,080` |
| `train/stats.return` | `943.7242` |
| `race/mean_speed_ms` | `4.0272` |
| `race/max_speed_ms` | `11.2011` |
| `race/gates_passed_per_ep` | `6.8137` |
| `race/lap_completion_rate` | `0.0` |
| `race/furthest_gate_reached` | `6.8137` |
| `race/final_dist_to_gate_m` | `5.4545` |
| `race/gates_per_second` | `0.3538` |
| `crash/total_rate` | `0.0706` |
| `crash/ground_rate` | `0.0263` |
| `crash/contact_rate` | `0.0447` |
| `crash/distance_rate` | `0.0` |
| `curriculum/phase` | `0.0` |
| `curriculum/accuracy_ema` | `0.0` |
| `curriculum/furthest_gate_ema` | `6.8120` |
| `entropy/current_coef` | `0.012` |

Latest per-gate snapshot:

| Gate metric | Snapshot value |
| --- | ---: |
| `gates/gate_00_crosses` | `0.9995` |
| `gates/gate_01_crosses` | `0.9889` |
| `gates/gate_02_crosses` | `0.9834` |
| `gates/gate_03_crosses` | `0.9783` |
| `gates/gate_04_crosses` | `0.9645` |
| `gates/gate_05_crosses` | `0.9548` |
| `gates/gate_06_crosses` | `0.9437` |
| `gates/gate_07_crosses` | `0.0005` |
| `gates/gate_08_crosses` | `0.0` |
| `gates/gate_09_crosses` | `0.0` |
| `gates/gate_10_crosses` | `0.0` |
| `gates/gate_11_crosses` | `0.0` |

Because the gate metrics are zero-based, `gate_06_crosses` corresponds to
crossing YAML gate 7. The policy is therefore usually clearing the first
elevated gate. The near-zero `gate_07_crosses` means it almost never clears YAML
gate 8, the same-XY lower gate after the elevated gate.

The current bottleneck is therefore not generic early navigation. It is the
specific elevated-to-low drop/reversal at gates 7 -> 8.

## Design Rationale And Tradeoffs

### Why accuracy before speed?

Drone racing rewards can easily become misaligned. If speed is rewarded too
early, the policy may learn to move fast near the course without passing gates.
The current curriculum delays speed pressure until the policy can complete full
ordered laps with meaningful reliability.

### Why sparse gate rewards dominate phase 0?

The true task is crossing gates in order. Sparse gate rewards directly encode
that. Dense progress is useful for exploration, but if it dominates, the policy
can exploit it by flying along the route direction while missing apertures.

The current balance uses:

```text
reward_progress_scale = 1.0
reward_gate_passage = 60.0
reward_gate_sequence_scale = 20.0
```

This makes gate crossing the main source of positive return.

### Why keep dense progress at all?

Pure sparse rewards are hard for early PPO training because random policies may
rarely pass gates. Path-projection reward gives the agent a gradient-like signal
for moving in roughly the right direction. It is intentionally small enough to
guide rather than replace the gate objective.

### Why include next-to-next gate information?

Racing lines depend on the next turn. A policy that only aims at the current
gate center may pass the gate in a bad pose or velocity for the following gate.
The next-to-next vector gives lookahead without requiring a recurrent policy.

### Why include gate index?

The track has gate-specific events: elevated gates, vertical drops, reversals,
and duplicated closure gate. Relative geometry helps, but the normalized gate
index makes these special cases easier to identify. This is a sim-only design
choice and is acceptable if the goal is course-specific performance.

### Why use privileged intrinsics?

Privileged intrinsics can speed up training by giving the critic and actor exact
physical context. This is acceptable for a fixed sim-only project. It should be
reconsidered for generalization or real-world transfer.

### Why use high adaptive entropy now?

The policy has not completed a lap. The system therefore keeps entropy at the
phase-0 maximum to continue exploring. Reducing entropy too early would make the
current partial-course behavior more deterministic and could lock in the
gate-7-to-gate-8 failure.

### What are the main current risks?

1. Clamped Gaussian mismatch: the policy trains on unclamped action log-probs
   but the simulator executes clamped actions.
2. Terminal masking: PPO GAE uses `terminated`, so lap completion may be
   bootstrapped like a nonterminal transition.
3. Hard-gate data scarcity: always starting from gate 0 means the policy gets
   relatively little practice on gate 8 compared with early gates.
4. Endpoint crossing detection: fast or awkward crossings can be missed.
5. Phase unlock is very strict: phase 1 cannot begin until full-lap EMA reaches
   0.40, so the policy may never receive explicit speed pressure if it remains
   stuck at the first hard reversal.

## Recommended Next Steps

1. Add targeted reset curriculum near hard gates.

   After early gates are reliable, allow some episodes to start before gate 7 or
   gate 8. This increases training data on the actual bottleneck instead of
   requiring the policy to fly six gates before every attempt.

2. Replace endpoint gate detection with segment-based crossing.

   Use previous and current drone positions to intersect the gate plane, then
   check the intersection point against aperture bounds. This is more robust for
   fast flight and same-XY reversal sections.

3. Add signed gate-frame approach shaping.

   For the active target gate, reward progress in the gate's local crossing
   direction and penalize being on the wrong side/orientation near the aperture.
   This is especially useful for gates 8 and 12, where direction reversal is the
   main challenge.

4. Consider a bounded or squashed action distribution.

   A tanh-squashed Gaussian or bounded distribution would align PPO log-probs
   with the action actually executed by the simulator. This may become more
   important while entropy is high.

5. Fix PPO terminal masking for lap completion.

   Use full `done` semantics, or at least include `completed_task`, when
   computing GAE masks. This prevents bootstrapping through successful terminal
   lap-completion transitions.

6. Keep the accuracy-first curriculum, but add a bottleneck-practice branch.

   The current curriculum is directionally right. The issue is not that phase 0
   exists; it is that phase 0 gives too little direct practice on the first
   hard vertical reversal.

## Bottom Line

The current training strategy is coherent: PPO with high-throughput vectorized
rollouts, dense-but-small progress shaping, dominant ordered gate rewards,
accuracy-first curriculum, and adaptive entropy that stays high until full laps
exist.

The active run shows that this design has learned meaningful racing behavior:
the drone clears the early course and reaches the first elevated drop section
with low distance-crash rate. The next improvement should focus less on global
PPO tuning and more on the localized failure at the gate 7 -> gate 8
drop/reversal.
