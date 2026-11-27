# Smart Irrigation with Reinforcement Learning

Mission: an agent that switches the irrigation taps of a small farm **ON and OFF** each day,
keeping crops healthy through a full growing season with a limited water reservoir and uncertain rain.

The project compares a value-based method (DQN) with three policy-gradient methods
(REINFORCE, A2C, PPO), all built with **TorchRL**, on the same custom Gymnasium environment
rendered in **MuJoCo**.

## The environment (`environment/custom_env.py`)

A 3×3 farm: 9 crop plots, each with its own solenoid tap, fed from one reservoir.
One step is one day; one episode is one 120-day growing season. The simulator is
the source of all state and transition data. MuJoCo visualizes that state but
does not calculate the irrigation, weather, or crop dynamics.

| Element | Definition |
|---|---|
| **Actions** | 9 taps, each ON/OFF → `MultiBinary(9)`. Agents use the shared `Discrete(512)` wrapper (`environment/wrappers.py`): action *k* is the 9-bit tap pattern (0 = all OFF, 511 = all ON). |
| **Observations** | `Box(0,1,(41,))`: 9 noisy soil-moisture readings, 9 crop-health values, 9 fixed plot exposures, 9 current tap states, then reservoir fraction, rain forecast, temperature, season progress, and crop coefficient (Kc). The detailed encoding and provenance are below. |
| **Dynamics** | Each day's chosen taps add water before weather and evapotranspiration (ET) are applied. Rain occurs with yesterday's forecast probability; temperature, rain amount, and plot exposure affect water loss. ET is base ET × growth-stage Kc × temperature factor × plot exposure. Excess soil water drains. Moisture and crop health persist between decisions; irrigation also depletes a shared, finite reservoir. |
| **Rewards** | Each day: +1 per living plot in the 0.45–0.75 moisture band; −1.5 × drought stress and −1.5 × waterlogging per living plot; −0.08 per open tap; −0.03 per tap-state change; and −25 per newly dead plot. At harvest, add +10 × sum of crop health. This trades off crop survival/health against water and valve costs over the season. |
| **Start state** | Moisture is sampled independently from 0.45–0.65 per plot; health is 1, all taps are OFF, and the reservoir is full. Plot exposure is sampled from 0.8–1.2 and held for the season. Temperature starts at 0.5; a stochastic forecast is sampled for the first day. `reset(seed=...)` controls the random sequence. |
| **Terminal conditions** | The episode terminates early if all plots die, or is truncated after 120 days. A plot dies when health reaches 0 and cannot be revived or irrigated. The harvest bonus is applied at season end if the farm has not already terminated. |
| **Visualization** | MuJoCo 3D scene: soil color shows moisture, cyan taps are open, crop color shows health/death, and reservoir water level shows remaining capacity. `play.py` renders live or records an MP4. |

### Observation specification

The observation is a float32 `Box(0, 1, shape=(41,))`. The environment
constructs it in `_get_obs`; the action wrapper leaves it unchanged, and each
agent receives it under the TorchRL TensorDict key `observation`.

| Indices / field | Meaning, encoding and range | Provenance and update cadence |
|---|---|---|
| `0:9` moisture | One value per plot, clipped to `[0,1]` | True soil moisture plus independent Gaussian sensor noise (`σ=0.02` by default), clipped to `[0,1]`; sampled at reset and after every daily transition. |
| `9:18` health | One crop-health fraction per plot, `[0,1]`; 0 means dead | Simulated health from water stress, waterlogging, and healthy-band time; initialized to 1 and updated daily. |
| `18:27` exposure | One plot-specific value, normalized as `(exposure−0.8)/0.4` into `[0,1]` | Per-season sun/drainage factor sampled uniformly from `[0.8,1.2]` at reset; fixed for that season. |
| `27:36` tap states | One binary 0/1 value per tap | The tap pattern used for the most recent transition; initialized to all 0 and updated each day. |
| `36` reservoir | Remaining reservoir fraction, `[0,1]` | Remaining water divided by configured capacity; initialized full and reduced by daily water demand. |
| `37` rain forecast | Probability in `[0,1]` that it rains on the next transition | Sampled from a two-regime beta model at reset and after each transition; the next day's rain is a Bernoulli draw from this probability. |
| `38` temperature | Daily temperature normalized/clipped to `[0,1]` | At reset set to 0.5; thereafter sampled daily from a normal distribution (mean `0.5 + heat_shift`, standard deviation `0.12`) and clipped. |
| `39` season progress | `day / season_length`, `[0,1]` | Simulated day counter; 0 at reset, updated once per daily transition. |
| `40` crop coefficient | Growth-stage Kc divided by 1.15, `[0.45/1.15, 1]` | Deterministic FAO-56-style growth-stage curve evaluated at the current day; updated with season progress. |

### Why reinforcement learning

This is a sequential control problem: an action changes persistent soil
moisture, crop health, tap wear, and a shared reservoir, while weather and
rain are uncertain. A greedy response to the current sensor can waste water
before forecast rain or leave later crop stages without reserves. The agent
must balance near-term moisture against season-long survival and harvest
reward. Rule-based policies are important baselines; RL is being evaluated for
this delayed, uncertain control problem, not assumed to beat every hand-built
rule.

The water budget is deliberately tight: a simple "water when dry" rule uses ~90% of the reservoir,
and opening taps carelessly empties it mid-season.

## Algorithms (`training/algorithms.py`)

| Method | TorchRL components |
|---|---|
| DQN | `QValueActor`, `DQNLoss` (double DQN), `HardUpdate` target network, `TensorDictReplayBuffer` |
| REINFORCE | `ProbabilisticActor`, `ReinforceLoss` (Monte Carlo returns, optional learned baseline) |
| A2C | `ProbabilisticActor`, `ValueOperator`, `A2CLoss`, `GAE` |
| PPO | `ProbabilisticActor`, `ValueOperator`, `ClipPPOLoss`, `GAE` |

All four use the same network body. Because a plain 512-way output layer learns very slowly,
the default head is **per-plot weight sharing**: one small network scores each tap from that plot's
readings plus the farm-wide readings, and the 512 action scores are built from the 9 tap scores.
`head="factored"` (no sharing) and `head="flat"` (plain 512 outputs) are included as
hyperparameter runs to show the effect.

## Hyperparameter experiments and analysis

`training/experiments.py` defines ten configurations for each algorithm. Run 1
is the default; other runs change one or two controls. The controls are
algorithm-specific so DQN exploration/replay/target-network settings are not
incorrectly applied to policy-gradient methods.

| Algorithm | Ten varied controls |
|---|---|
| DQN | default; learning rate `3e-4`; learning rate `3e-3`; `gamma=0.95`; batch size 128 and replay capacity 100,000; target update interval 100; target update interval 2,000; longer epsilon decay (50,000 steps) and `eps_end=0.02`; disable Double DQN; flat 512-output head. |
| REINFORCE | default with learned baseline; learning rate `1e-4`; learning rate `1e-3`; `gamma=0.95`; `gamma=0.999`; disable baseline; 1 episode per batch; 8 episodes per batch; hidden size 64; factored head. |
| A2C | default; learning rate `1e-4`; learning rate `1e-3`; `gamma=0.95`; GAE lambda `0.8`; GAE lambda `1.0`; entropy coefficient `0`; entropy coefficient `0.05`; critic coefficient 1.0 and 8 episodes per batch; factored head. |
| PPO | default; learning rate `1e-4`; learning rate `1e-3`; `gamma=0.95`; clip epsilon `0.1`; clip epsilon `0.3`; 10 PPO epochs; entropy coefficient `0`; 8 episodes per batch and minibatch size 512; factored head. |

Every full run uses 600 training seasons and the same season-seed sequence
across configurations. Each configuration is evaluated on 20 fixed validation
seasons; the highest validation mean reward selects that algorithm's checkpoint.
The held-out final comparison uses 50 different seasons by default, with
separate drought, heatwave, and reduced-reservoir stress tests. `--quick` is
only a smoke test, not experiment evidence: it trains two configurations per
algorithm for 20 seasons and writes its artifacts under `quick/` so it cannot
replace the full-run results.

Inspect `logs/hyperparameter_results.csv` for validation results and
`logs/{algo}_run_{k}.csv` for training histories. Interpret reward together
with plots alive, full-season rate, water use, and reward variability; training
reward alone is not evidence of better held-out behavior. A partial quick-run
summary must not be presented as a completed hyperparameter sweep.

The current workspace summary has only two result rows per algorithm, so it
does not yet evidence the full grid above. Its default-versus-lower-learning-rate
validation means are: DQN `29.7 → 103.3` (full-season rate `0.40 → 0.50`);
REINFORCE `26.7 → 35.1` (`0.55 → 0.55`); A2C `56.7 → 45.5`
(`0.55 → 0.55`); and PPO `80.1 → 26.7` (`0.65 → 0.60`). These small-sample
contrasts suggest learning-rate sensitivity, but cannot establish effects for
the other controls or a reliable ranking. The generated DQN histories for runs
1 and 2 also contain only 20 episodes while some other run histories contain
600, so the existing artifacts are mixed and should not be treated as a fair,
complete sweep. Run the full command below to regenerate coherent evidence.

## Running

```bash
uv sync                               # or: pip install -e .
python main.py --quick --jobs 2       # smoke test; artifacts go to quick/
python main.py --jobs 4               # full experiment: 4 algorithms × 10 configs × 600 seasons
python main.py --jobs 4 --output-dir full_run  # full run without replacing existing artifacts
python main.py --skip-training        # re-evaluate and re-plot from saved models/logs

python play.py --agent ppo            # watch the best PPO agent in the MuJoCo viewer
python play.py --agent random         # random agent (for the demo video)
```

On macOS use `mjpython play.py ...` for the live viewer.

The environment also exposes a Gymnasium API through
`environment.wrappers.make_gym_env`: `reset(seed=...)` returns
`(observation, info)` and `step(action_index)` returns the standard
`(observation, reward, terminated, truncated, info)` tuple. Checkpoints use
PyTorch serialization (`training.algorithms.save_agent` / `load_agent`) and
contain the algorithm name, configuration, and network state. `play.py` loads
those checkpoints and executes the same policy against the environment; this
is the project's integration path, with no separate frontend or mobile client.

## Outputs

| File | Contents |
|---|---|
| `logs/hyperparameter_results.csv` | Every full-sweep run: configuration, validation reward, plots alive, water used, episodes to converge |
| `logs/{algo}_run_{k}.csv` | Per-episode training history (reward, loss, entropy, water, plots alive) |
| `logs/final_comparison.csv` | Best agent per algorithm vs rule-based baselines on 50 unseen test seasons |
| `logs/stress_test.csv` | Best agents on unseen harsher climates (drought, heatwave, smaller reservoir) |
| `assets/learning_curves.png` | Recorded configurations from the summary CSV, best recorded run highlighted; title reports the count |
| `assets/best_runs_comparison.png` | Best run of each algorithm on one plot |
| `assets/objective_and_entropy.png` | DQN TD-loss curve and policy-entropy curves |
| `assets/final_comparison.png`, `assets/stress_test.png` | Test and robustness results |
| `models/{algo}/best.pt` | Best model per algorithm (selected on validation seasons) |
| `quick/` | Separate smoke-test logs, models, tables, and figures created by `--quick` |

Training, validation and test use different season seeds, so reported scores are on weather the
agents never trained on. Every run trains on the same sequence of seasons for a fair comparison.

## Project structure

```
environment/
  custom_env.py        farm simulation + MuJoCo scene
  wrappers.py          shared Discrete(512) action interface
training/
  algorithms.py        DQN and policy-gradient agents (TorchRL)
  dqn_training.py      DQN training loop
  pg_training.py       REINFORCE / A2C / PPO training loop
  experiments.py       10 hyperparameter configurations per algorithm
  evaluation.py        evaluation, rule-based baselines, stress scenarios
main.py                run everything, save tables and figures
play.py                watch or record an agent
```
