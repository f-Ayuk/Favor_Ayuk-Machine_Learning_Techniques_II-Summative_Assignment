"""
Smart irrigation farm environment (Gymnasium + MuJoCo).

A 3x3 farm of crop plots. Each plot has its own solenoid tap that the
agent switches ON or OFF once per simulated day. The agent must keep soil
moisture in the healthy band for a full growing season while using a
limited reservoir of water, anticipating rain from a noisy forecast, and
avoiding excessive valve switching.

Soil-water model (simplified FAO-56 style water balance, per plot, per day):
    moisture(t+1) = moisture(t) + irrigation + rain - ET
    ET = base_ET * Kc(growth stage) * temperature factor * plot exposure

MuJoCo is used for the 3D visualisation of the farm. The water dynamics
are the custom model above, not MuJoCo physics.

Native action space:   MultiBinary(9)   (one ON/OFF bit per tap)
Observation space:     Box(0, 1, (41,))  (see OBS_LAYOUT below)
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np


# ---------------------------------------------------------------------------
# Scene description (MJCF)
# ---------------------------------------------------------------------------

def _plot_xml(r, c):
    """One plot: soil slab, a field of plants, a tap and a feeder pipe."""
    x = (c - 1) * 2.2
    y = (1 - r) * 2.2
    plants = []
    for i in range(5):
        for j in range(5):
            px = -0.56 + i * 0.28
            py = -0.56 + j * 0.28
            plants.append(
                f'<geom name="plant_{r}_{c}_{i}_{j}" type="capsule" '
                f'fromto="{px:.2f} {py:.2f} 0.03 {px:.2f} {py:.2f} 0.30" '
                f'size="0.045" material="crop_mat" contype="0" conaffinity="0"/>'
            )
    return f"""
    <body name="plot_{r}_{c}" pos="{x:.2f} {y:.2f} 0.0">
      <geom name="soil_{r}_{c}" type="box" size="0.9 0.9 0.03" pos="0 0 0.03"
            rgba="0.55 0.38 0.22 1"/>
      {''.join(plants)}
      <geom name="tap_{r}_{c}" type="cylinder" pos="-0.95 0.95 0.18"
            size="0.10 0.14" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <geom name="feed_{r}_{c}" type="cylinder" fromto="-0.95 0.95 0.05 -0.95 1.10 0.05"
            size="0.04" material="pipe_mat" contype="0" conaffinity="0"/>
    </body>"""


def build_mjcf():
    plots = "".join(_plot_xml(r, c) for r in range(3) for c in range(3))
    # Main header pipes run between the rows of plots.
    headers = "".join(
        f'<geom name="header_{k}" type="cylinder" '
        f'fromto="-3.4 {y:.2f} 0.05 3.4 {y:.2f} 0.05" size="0.06" material="pipe_mat"/>'
        for k, y in enumerate([3.3, 1.1, -1.1])
    )
    return f"""
<mujoco model="smart_irrigation_farm">
  <option timestep="0.02"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35"/>
    <global azimuth="120" elevation="-40" offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.75 0.95" rgb2="0.2 0.3 0.45"
             width="512" height="3072"/>
    <texture name="grass_tex" type="2d" builtin="checker" rgb1="0.22 0.42 0.18"
             rgb2="0.26 0.47 0.21" width="512" height="512"/>
    <material name="grass_mat" texture="grass_tex" texrepeat="12 12"/>
    <material name="pipe_mat" rgba="0.30 0.32 0.36 1" specular="0.6" shininess="0.6"/>
    <material name="crop_mat" rgba="0.20 0.70 0.22 1"/>
  </asset>
  <worldbody>
    <light pos="3 -3 8" dir="-0.3 0.3 -1" directional="true" castshadow="true"/>
    <geom name="ground" type="plane" size="8 8 0.1" material="grass_mat"/>
    <body name="reservoir" pos="4.6 0 0">
      <geom name="tank" type="cylinder" size="0.7 0.6" pos="0 0 0.6" rgba="0.6 0.6 0.65 0.35"/>
      <geom name="tank_water" type="cylinder" size="0.65 0.55" pos="0 0 0.55"
            rgba="0.15 0.45 0.9 0.9" contype="0" conaffinity="0"/>
    </body>
    {headers}
    {plots}
  </worldbody>
</mujoco>
"""


# Index layout of the flat observation vector.
OBS_LAYOUT = {
    "moisture": slice(0, 9),        # sensor reading, noisy
    "health": slice(9, 18),         # crop health per plot (0 = dead)
    "exposure": slice(18, 27),      # plot sun/drainage exposure, rescaled
    "tap_states": slice(27, 36),    # taps currently ON
    "reservoir": 36,                # fraction of water left
    "rain_forecast": 37,            # forecast probability of rain tomorrow
    "temperature": 38,              # today's temperature, rescaled
    "season_progress": 39,          # day / season_length
    "crop_coefficient": 40,         # Kc for the current growth stage, rescaled
}


class SmartIrrigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Soil moisture thresholds (volumetric fraction, normalised to 0-1)
    OPT_LOW, OPT_HIGH = 0.45, 0.75   # healthy band
    STRESS = 0.35                     # below this the crop is water stressed
    WATERLOG = 0.85                   # above this roots are waterlogged

    def __init__(
        self,
        render_mode=None,
        season_length=120,
        reservoir_capacity=1.0,
        water_per_tap=0.011,       # reservoir fraction used by one open tap per day
        irrigation_gain=0.07,      # soil moisture added per day by an open tap
        base_et=0.032,             # reference evapotranspiration per day
        rain_scale=1.0,            # < 1 gives a drier season (used for stress tests)
        heat_shift=0.0,            # > 0 gives a hotter season (used for stress tests)
        sensor_noise=0.02,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.season_length = season_length
        self.reservoir_capacity = reservoir_capacity
        self.water_per_tap = water_per_tap
        self.irrigation_gain = irrigation_gain
        self.base_et = base_et
        self.rain_scale = rain_scale
        self.heat_shift = heat_shift
        self.sensor_noise = sensor_noise

        self.n_plots = 9
        self.action_space = spaces.MultiBinary(self.n_plots)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(41,), dtype=np.float32)

        # MuJoCo scene (visualisation only)
        self.model = mujoco.MjModel.from_xml_string(build_mjcf())
        self.data = mujoco.MjData(self.model)
        gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self._soil_ids = [gid(f"soil_{r}_{c}") for r in range(3) for c in range(3)]
        self._tap_ids = [gid(f"tap_{r}_{c}") for r in range(3) for c in range(3)]
        self._plant_ids = [
            [gid(f"plant_{r}_{c}_{i}_{j}") for i in range(5) for j in range(5)]
            for r in range(3) for c in range(3)
        ]
        self._water_id = gid("tank_water")
        self.viewer = None
        self.renderer = None

    # ------------------------------------------------------------------
    # Crop and weather helpers
    # ------------------------------------------------------------------
    def _crop_coefficient(self):
        """FAO-56 style Kc curve: initial, development, mid-season, late."""
        p = self.day / self.season_length
        if p < 0.2:
            return 0.45
        if p < 0.45:
            return 0.45 + (1.15 - 0.45) * (p - 0.2) / 0.25
        if p < 0.8:
            return 1.15
        return 1.15 - (1.15 - 0.65) * (p - 0.8) / 0.2

    def _new_forecast(self):
        """Probability of rain tomorrow. Tomorrow's rain is drawn from it,
        so the forecast is genuinely informative but not perfect."""
        rainy_spell = self.np_random.random() < 0.25
        p = self.np_random.beta(4, 2) if rainy_spell else self.np_random.beta(1, 6)
        return float(p)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Start state: randomised moisture and heterogeneous plots
        self.moisture = self.np_random.uniform(0.45, 0.65, self.n_plots)
        self.health = np.ones(self.n_plots)
        self.alive = np.ones(self.n_plots, dtype=bool)
        self.exposure = self.np_random.uniform(0.8, 1.2, self.n_plots)
        self.tap_states = np.zeros(self.n_plots, dtype=np.int8)
        self.reservoir = self.reservoir_capacity
        self.day = 0
        self.temperature = 0.5
        self.rain_forecast = self._new_forecast()

        self.total_water = 0.0
        self.total_switches = 0
        self.total_rain = 0.0
        self.days_in_band = 0.0

        if self.render_mode is not None:
            self._update_visuals()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._info(0.0, 0)

    def step(self, action):
        action = np.asarray(action, dtype=np.int8).reshape(self.n_plots)
        action = action * self.alive          # dead plots are never irrigated

        # 1. Valve switching (mechanical wear)
        switches = int(np.sum(action != self.tap_states))
        self.total_switches += switches
        self.tap_states = action.copy()

        # 2. Irrigation, limited by the reservoir. If there is not enough
        #    water for every open tap, it is shared between them.
        open_taps = int(action.sum())
        demand = open_taps * self.water_per_tap
        supplied = min(demand, self.reservoir)
        fraction = supplied / demand if demand > 0 else 0.0
        self.reservoir -= supplied
        self.total_water += supplied
        self.moisture += action * self.irrigation_gain * fraction

        # 3. Weather: rain today was forecast yesterday
        rained = self.np_random.random() < self.rain_forecast
        rain = self.np_random.uniform(0.03, 0.10) * self.rain_scale if rained else 0.0
        self.total_rain += rain
        self.moisture += rain

        self.temperature = float(np.clip(
            self.np_random.normal(0.5 + self.heat_shift, 0.12), 0, 1))
        temp_factor = 0.7 + 0.6 * self.temperature
        et = self.base_et * self._crop_coefficient() * temp_factor * self.exposure
        if rained:
            et *= 0.5                           # cloudy, humid day
        self.moisture -= et

        # Deep drainage when soil is above field capacity
        self.moisture -= 0.3 * np.maximum(0.0, self.moisture - 0.80)
        self.moisture = np.clip(self.moisture, 0.0, 1.0)

        # 4. Crop health
        stress = np.clip((self.STRESS - self.moisture) / self.STRESS, 0, 1)
        waterlog = np.clip((self.moisture - self.WATERLOG) / (1 - self.WATERLOG), 0, 1)
        in_band = (self.moisture >= self.OPT_LOW) & (self.moisture <= self.OPT_HIGH)
        self.health -= 0.12 * stress + 0.10 * waterlog
        self.health += 0.01 * in_band
        self.health = np.clip(self.health, 0, 1) * self.alive

        newly_dead = self.alive & (self.health <= 0.0)
        self.alive &= ~newly_dead

        # 5. Reward
        alive_f = self.alive.astype(float)
        reward = (
            1.0 * np.sum(in_band * alive_f)
            - 1.5 * np.sum(stress * alive_f)
            - 1.5 * np.sum(waterlog * alive_f)
            - 0.08 * open_taps               # water / pumping cost
            - 0.03 * switches                # valve wear
            - 25.0 * int(newly_dead.sum())   # crop loss
        )
        self.days_in_band += float(np.mean(in_band * alive_f))

        self.day += 1
        self.rain_forecast = self._new_forecast()

        terminated = bool(not self.alive.any())       # whole farm lost
        truncated = bool(self.day >= self.season_length)
        if truncated and not terminated:
            reward += 10.0 * float(np.sum(self.health))   # harvest bonus

        if self.render_mode is not None:
            self._update_visuals()
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, self._info(rain, open_taps)

    # ------------------------------------------------------------------
    def _get_obs(self):
        noisy = self.moisture + self.np_random.normal(0, self.sensor_noise, self.n_plots)
        obs = np.concatenate([
            np.clip(noisy, 0, 1),
            self.health,
            (self.exposure - 0.8) / 0.4,
            self.tap_states,
            [self.reservoir / self.reservoir_capacity,
             self.rain_forecast,
             self.temperature,
             self.day / self.season_length,
             self._crop_coefficient() / 1.15],
        ])
        return np.clip(obs, 0, 1).astype(np.float32)

    def _info(self, rain, open_taps):
        return {
            "day": self.day,
            "water_used": float(self.total_water),
            "reservoir": float(self.reservoir),
            "plots_alive": int(self.alive.sum()),
            "mean_health": float(np.mean(self.health)),
            "switches": int(self.total_switches),
            "rain": float(rain),
            "open_taps": int(open_taps),
            "time_in_band": float(self.days_in_band / max(self.day, 1)),
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def _update_visuals(self):
        for k in range(self.n_plots):
            m = float(self.moisture[k])
            # Dry soil is light tan, wet soil is dark brown
            self.model.geom_rgba[self._soil_ids[k]] = [
                0.72 - 0.45 * m, 0.55 - 0.38 * m, 0.36 - 0.26 * m, 1.0]
            # Tap glows cyan when open
            self.model.geom_rgba[self._tap_ids[k]] = (
                [0.0, 0.85, 1.0, 1.0] if self.tap_states[k] else [0.3, 0.3, 0.3, 1.0])
            # Plants: green when healthy, yellow when stressed, brown when dead
            h = float(self.health[k])
            colour = ([0.45, 0.30, 0.12, 1.0] if not self.alive[k]
                      else [0.85 - 0.65 * h, 0.55 + 0.20 * h, 0.12, 1.0])
            for pid in self._plant_ids[k]:
                self.model.geom_rgba[pid] = colour
        # Reservoir water level (the water cylinder shrinks as it empties)
        level = max(self.reservoir / self.reservoir_capacity, 0.02)
        self.model.geom_size[self._water_id][1] = 0.55 * level
        self.model.geom_pos[self._water_id][2] = 0.05 + 0.55 * level
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=540, width=960)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.6, 0.0, 0.0]
            cam.distance, cam.azimuth, cam.elevation = 11.5, 120, -42
            self.renderer.update_scene(self.data, camera=cam)
            return self.renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
