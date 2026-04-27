# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec, BinaryDiscreteTensorSpec

import isaacsim.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate, quat_rotate_inverse, quat_axis, quat_mul
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView

from omni_drones.robots import ASSET_PATH

from pxr import UsdPhysics

# Debug visualization
try:
    from isaacsim.util.debug_draw import _debug_draw
    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    _debug_draw = None

class DroneRaceEnv(IsaacEnv):
    r"""
    A drone racing task where the agent must navigate through a sequence of gates
    in a racing track. The gates are arranged in a track pattern and the agent
    must pass through them in order.

    ## Observation

    - `drone_state` (15): Custom state vector `[lin_vel(3) | rot_mat_flat(9) | ang_vel(3)]`. Note that
      position is **not** included.
    - `next_gate_rpos` (3): The relative position of the next gate to the drone in the drone's local frame.
    - `next_to_next_gate_pos` (3): The position of the gate after the immediate next gate, expressed in
      the next gate's local frame. Clamped at the last gate (no wrap-around).
    - `next_gate_rot_mat_2col` (6): The first two columns of the next gate's rotation matrix in the world frame
      (i.e. the gate's local x- and y-axes expressed in world coordinates), flattened to a 6-vector.

    ## Reward  *(student implementation required)*

    **Your task is to implement the reward function in `_compute_reward_and_done`.**

    ## Episode End

    The episode ends when the drone crashes (physical contact or off-course),
    completes the full lap, or the maximum episode length is reached.
    **You should also implement the crash/termination condition** in
    `_compute_reward_and_done` (currently returns all-zeros as a placeholder).

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "Hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `track_config`          | dict  | None          | Optional dictionary defining gate positions and orientations. If provided, gates are placed according to this config. Format: `{"1": {"pos": (x, y, z), "yaw": angle}, ...}`. If None, uses circular track.                            |
    | `num_gates`             | int   | 8             | Number of gates in the racing track (only used if `track_config` is None).                                                                                                                                                              |
    | `track_radius`          | float | 5.0           | Radius of the circular track (only used if `track_config` is None).                                                                                                                                                                     |
    | `gate_spacing`          | float | 3.0           | Spacing between gates along the track (only used if `track_config` is None).                                                                                                                                                           |
    | `gate_height`           | float | 2.0           | Height of the gates (only used if `track_config` is None).                                                                                                                                                                              |
    | `gate_scale`            | float | 1.0           | Scale of the gate assets.                                                                                                                                                                                                              |
    | `gate_asset_path`       | str   | None          | Path to the gate USD asset. Defaults to `ASSET_PATH/gate/gate.usd` (isaac_drone_racer style). Can be overridden in config.                                                                                                        |
    """
    REWARD_CONFIG_KEYS = (
        "reward_progress_scale",
        "reward_speed_scale",
        "reward_speed_target_ms",
        "reward_speed_scale_phase2",
        "reward_speed_target_ms_phase2",
        "reward_stacked_direction_scale",
        "reward_stacked_entry_bonus",
        "reward_stacked_clear_bonus",
        "reward_gate_passage",
        "reward_gate_sequence_scale",
        "reward_lap_completion",
        "reward_lap_speed_bonus",
        "reward_angular_penalty",
        "reward_action_smooth_scale",
        "reward_crash_scale",
        "reward_altitude_scale",
        "reward_approach_scale",
        "reward_gate_centering_scale",
    )
    TERMINATION_CONFIG_KEYS = (
        "crash_dist_threshold",
        "crash_z_min",
        "stacked_reversal_dist_threshold",
    )

    def __init__(self, cfg, headless):
        # Keep reward and termination scalars in the active task YAML so the
        # environment code cannot drift from the configured track settings.
        self._load_task_scalars(
            cfg.task,
            self.REWARD_CONFIG_KEYS + self.TERMINATION_CONFIG_KEYS,
        )

        self.gate_scale = cfg.task.gate_scale
        self.phase_speed_unlock_on_first_success = bool(
            cfg.task.get("phase_speed_unlock_on_first_success", False)
        )
        self.phase2_speed_focus_accuracy_rate = float(
            cfg.task.get("phase2_speed_focus_accuracy_rate", 0.40)
        )
        self.phase2_disable_dense_shaping = bool(
            cfg.task.get("phase2_disable_dense_shaping", True)
        )
        self.reset_curriculum_enabled = bool(
            cfg.task.get("reset_curriculum_enabled", False)
        )
        self.reset_curriculum_gate_pass_threshold = float(
            cfg.task.get("reset_curriculum_gate_pass_threshold", 0.90)
        )
        self.reset_curriculum_gate0_prob = float(
            cfg.task.get("reset_curriculum_gate0_prob", 1.0)
        )
        self.reset_curriculum_prev_gate_prob = float(
            cfg.task.get("reset_curriculum_prev_gate_prob", 0.0)
        )
        self.reset_curriculum_random_gate_prob = float(
            cfg.task.get("reset_curriculum_random_gate_prob", 0.0)
        )
        self.reset_curriculum_metric_alpha = float(
            cfg.task.get("reset_curriculum_metric_alpha", 0.01)
        )
        reset_prob_sum = (
            self.reset_curriculum_gate0_prob
            + self.reset_curriculum_prev_gate_prob
            + self.reset_curriculum_random_gate_prob
        )
        if self.reset_curriculum_enabled and abs(reset_prob_sum - 1.0) > 1e-6:
            raise ValueError(
                "reset curriculum probabilities must sum to 1.0."
            )
        self.sparse_only_gate_indices = tuple(
            int(idx) for idx in cfg.task.get("sparse_only_gate_indices", [])
        )
        self.sparse_disable_smoothness_penalties = bool(
            cfg.task.get("sparse_disable_smoothness_penalties", False)
        )
        self.stacked_reversal_entry_gates = tuple(
            int(idx) for idx in cfg.task.get("stacked_reversal_entry_gates", [])
        )
        self.stacked_reversal_target_gates = tuple(
            int(idx) for idx in cfg.task.get("stacked_reversal_target_gates", [])
        )
        if len(self.stacked_reversal_entry_gates) != len(self.stacked_reversal_target_gates):
            raise ValueError(
                "stacked_reversal_entry_gates and stacked_reversal_target_gates must have the same length."
            )
        self.stacked_reversal_gate_pairs = tuple(
            zip(self.stacked_reversal_entry_gates, self.stacked_reversal_target_gates)
        )
        self.stacked_reversal_grace_steps = int(cfg.task.get("stacked_reversal_grace_steps", 0))
        self.step_dt = float(cfg.sim.dt * cfg.sim.substeps)
        
        # Gate asset path - default to isaac_drone_racer gate asset
        # User can override this in config: gate_asset_path: "path/to/gate.usd"
        # If not specified, defaults to gate/gate.usd (isaac_drone_racer style)
        # Gate asset is in ASSET_PATH/gate/gate.usd
        self.gate_asset_path = cfg.task.get("gate_asset_path", ASSET_PATH + "/gate/gate_a2rl.usd")
        
        # Track configuration: support both config-based and circular track
        self.track_config = cfg.task.get("track_config", None)
        if self.track_config is not None:
            # Config-based track: gates defined with positions and yaw angles
            self.num_gates = len(self.track_config)
            self.track_type = "config"
            # For config-based tracks, use gate_height from config or default
            self.gate_height = cfg.task.get("gate_height", 2.0)
        else:
            # Circular track (backward compatibility)
            self.num_gates = int(cfg.task.num_gates)
            self.track_radius = cfg.task.track_radius
            self.gate_spacing = cfg.task.gate_spacing
            self.gate_height = cfg.task.gate_height
            self.track_type = "circular"
        self.num_course_gates = max(min(self.num_gates - 1, 12), 1)
        self.phase1_unlock_gate_index = int(
            cfg.task.get("phase1_unlock_gate_index", min(5, self.num_course_gates - 1))
        )
        self.phase1_unlock_gate_index = min(
            max(self.phase1_unlock_gate_index, 0), self.num_course_gates - 1
        )
        self.phase1_unlock_gate_pass_rate = float(
            cfg.task.get("phase1_unlock_gate_pass_rate", 0.70)
        )
        self.hard_turn_target_gates = tuple(
            sorted(
                {
                    gate_idx
                    for gate_idx in (
                        int(idx) for idx in cfg.task.get("hard_turn_target_gates", [5])
                    )
                    if 0 <= gate_idx < self.num_course_gates
                }
            )
        )
        self.reward_hard_turn_direction_scale = float(
            cfg.task.get("reward_hard_turn_direction_scale", 0.0)
        )
        self.reward_hard_turn_clear_bonus = float(
            cfg.task.get("reward_hard_turn_clear_bonus", 0.0)
        )
        self.hard_turn_corridor_x_min = float(
            cfg.task.get("hard_turn_corridor_x_min", -10.0)
        )
        self.hard_turn_corridor_x_max = float(
            cfg.task.get("hard_turn_corridor_x_max", 4.0)
        )
        self.hard_turn_corridor_halfwidth = float(
            cfg.task.get("hard_turn_corridor_halfwidth", float(cfg.task.get("gate_width", 1.0)) * 3.0)
        )
        self.hard_turn_corridor_height = float(
            cfg.task.get("hard_turn_corridor_height", self.gate_height * 2.0)
        )
        self.hard_turn_penalty_scale = float(
            cfg.task.get("hard_turn_penalty_scale", 1.0)
        )
        
        import traceback
        import sys
        
        print(f"[DroneRaceEnv] Initializing, num_gates={self.num_gates}")
        try:
            super().__init__(cfg, headless)
            print(f"[DroneRaceEnv] super().__init__ completed")
        except Exception as e:
            print("=" * 80)
            print("ERROR: Failed in super().__init__")
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)
            raise

        try:
            self.drone.initialize(track_contact_forces=True)
            print(f"[DroneRaceEnv] drone.initialize() completed")
        except Exception as e:
            print("=" * 80)
            print("ERROR: Failed to initialize drone")
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)
            raise

        self.hover_cmd_thrust = None

        print(f"[DroneRaceEnv] num_envs={self.num_envs}, num_gates={self.num_gates}")
        # Track gate progress for each environment
        self.gate_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gate_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.track_completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.prev_distance_to_gate = torch.zeros(self.num_envs, device=self.device)
        # Gate crossing detection: drone position in gate frame from previous step
        self.gate_width = cfg.task.get("gate_width", 1.0)
        self.prev_drone_in_gate_frame = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_action = torch.zeros(self.num_envs, 1, self.drone.action_spec.shape[-1], device=self.device)
        self.effort = torch.zeros(self.num_envs, 1, self.drone.action_spec.shape[-1], device=self.device)
        self.prev_drone_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.total_frames_counter = 0
        self.angular_penalty_decay_frames = int(cfg.task.get("angular_penalty_decay_frames", 100_000_000))
        self.gate_just_passed_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.crash_grace_steps = int(cfg.task.get("crash_grace_steps", 100))
        self.stacked_reversal_grace = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gates_crossed_this_ep = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # Per-gate crossing counters (12 gates, gate 12 = lap closure = same as gate 0)
        self.per_gate_crosses = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.long)
        self.episode_start_gate = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_reset_bucket = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_gate_pass_ema = torch.ones(12, device=self.device)
        self.active_reset_curriculum_gate = -1
        # Furthest gate index reached this episode
        self.furthest_gate_this_ep = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # Altitude tracking
        self.min_altitude_this_ep = torch.full((self.num_envs,), float('inf'), device=self.device)
        self.mean_altitude_acc = torch.zeros(self.num_envs, device=self.device)
        # Action magnitude tracking
        self.mean_action_mag_acc = torch.zeros(self.num_envs, device=self.device)

        # --- Accuracy-first curriculum state ---
        # Phase 0: accuracy-first full laps from gate 0, with shaping support.
        # Phase 1: bridge phase with mild speed shaping still enabled.
        # Phase 2: reliable-full-lap speed phase with shaping removed.
        self.curriculum_phase = 0  # 0=accuracy-first, 1=bridge-speed, 2=speed-focus
        self.curriculum_ema_alpha = cfg.task.get("curriculum_ema_alpha", 0.01)
        self.lap_completion_rate_ema = 0.0
        self.furthest_gate_ema = 0.0
        self.phase_speed_unlock_accuracy_rate = cfg.task.get("phase_speed_unlock_accuracy_rate", 0.40)
        self.curriculum_min_phase_frames = int(cfg.task.get("curriculum_min_phase_frames", 2_000_000))
        self._phase_started_at_frames = 0  # frame counter at start of current phase (updated on transition)

        # Use a single view with wildcard pattern to access all gates
        try:
            print(f"[DroneRaceEnv] Creating RigidPrimView with pattern='/World/envs/env_*/Gate_*', shape=[{self.num_envs}, {self.num_gates}]")
            self.gates = RigidPrimView(
                "/World/envs/env_*/Gate_*",
                reset_xform_properties=False,
                shape=[self.num_envs, self.num_gates],
                track_contact_forces=False
            )
            print(f"[DroneRaceEnv] RigidPrimView created, calling initialize()...")
            self.gates.initialize()
            print(f"[DroneRaceEnv] gates.initialize() completed")
            if self.enable_viewport:
                self._apply_track_viewport_camera()
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed to initialize gates view with num_envs={self.num_envs}, num_gates={self.num_gates}")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f"ERROR: Failed to initialize gates view\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 80 + "\n")
            raise  # Re-raise to see the full error
        
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-1.0, -1.0, 1.5], device=self.device),
            torch.tensor([1.0, 1.0, 2.5], device=self.device)
        )
        # Reset noise is local to the chosen start gate, so keep it modest.
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.05, -0.05, -0.08], device=self.device) * torch.pi,
            torch.tensor([0.05, 0.05, 0.08], device=self.device) * torch.pi
        )

        self.offset_local = torch.tensor([-1.5, 0.0, self.gate_height / 2.0], device=self.device)
        self.alpha = 0.8
        
        # Debug visualization: enable via config (default: False)
        self.debug_gate_origins = cfg.task.get("debug_gate_origins", False)
        if self.debug_gate_origins and DEBUG_DRAW_AVAILABLE:
            self.draw = _debug_draw.acquire_debug_draw_interface()
            self.axis_length = cfg.task.get("debug_axis_length", 0.3)  # Length of axis lines
        else:
            self.draw = None

    def _load_task_scalars(self, task_cfg, keys):
        missing = [key for key in keys if key not in task_cfg]
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(
                "DroneRace task config is missing required scalar keys: "
                f"{missing_str}. Add them to the active cfg/task/*.yaml file."
            )

        for key in keys:
            setattr(self, key, task_cfg[key])

    def _draw_gate_origins(self, gate_world_pos, gate_world_rot, env_idx=0):
        """
        Draw coordinate axes at gate positions to visualize where the gate frame origin is.
        
        Args:
            gate_world_pos: (num_envs, num_gates, 3) tensor of gate positions in world coordinates
            gate_world_rot: (num_envs, num_gates, 4) tensor of gate rotations (quaternions) in world coordinates
            env_idx: Which environment to visualize (default: 0, first environment)
        """
        if self.draw is None:
            return
        
        # Clear previous lines
        self.draw.clear_lines()
        
        # Select one environment to visualize
        gate_pos = gate_world_pos[env_idx]  # (num_gates, 3) - world coordinates
        gate_rot = gate_world_rot[env_idx]  # (num_gates, 4) - world coordinates
        
        # Define axis directions in local frame (gate's local frame)
        # X-axis (red): forward direction
        # Y-axis (green): right direction  
        # Z-axis (blue): up direction
        axis_dirs_local = torch.tensor([
            [self.axis_length, 0.0, 0.0],  # X-axis (red)
            [0.0, self.axis_length, 0.0],  # Y-axis (green)
            [0.0, 0.0, self.axis_length],  # Z-axis (blue)
        ], device=self.device, dtype=torch.float32)  # (3, 3)
        
        # Colors: Red for X, Green for Y, Blue for Z (RGBA)
        axis_colors = [
            (1.0, 0.0, 0.0, 1.0),  # Red for X
            (0.0, 1.0, 0.0, 1.0),  # Green for Y
            (0.0, 0.0, 1.0, 1.0),  # Blue for Z
        ]
        
        # Draw axes for each gate
        for gate_idx in range(gate_pos.shape[0]):
            gate_origin = gate_pos[gate_idx]  # (3,)
            gate_quat = gate_rot[gate_idx]  # (4,)
            
            # Rotate axis directions from gate's local frame to world frame
            # quat_rotate expects (N, 4) and (N, 3), so we need to rotate each axis separately
            # Expand gate_quat to match number of axes (3 axes)
            gate_quat_expanded = gate_quat.unsqueeze(0).expand(3, -1)  # (3, 4)
            axis_dirs_world = quat_rotate(
                gate_quat_expanded,  # (3, 4)
                axis_dirs_local  # (3, 3) - each row is a 3D vector
            )  # (3, 3) - each row is a rotated 3D vector
            
            # Draw each axis
            for axis_idx, (axis_dir, color) in enumerate(zip(axis_dirs_world, axis_colors)):
                start_point = gate_origin.cpu().tolist()
                end_point = (gate_origin + axis_dir).cpu().tolist()
                
                # Draw line from origin to end point
                self.draw.draw_lines(
                    [start_point],
                    [end_point],
                    [color],
                    [3.0]  # Line width
                )

    def _apply_track_viewport_camera(self):
        """Re-frame the Kit viewport on the gate course (see _design_scene track bounds)."""
        if not getattr(self, "enable_viewport", False):
            return
        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except ImportError:
            return
        if not hasattr(self, "_track_cam_center_local"):
            return

        central = self.envs_positions[self.central_env_idx].detach().cpu().numpy()
        c = self._track_cam_center_local + central
        span = float(self._track_cam_span)
        # Isometric-style offset so a ~10 m square track stays in view.
        eye = c + np.array([span * 1.15, -span * 1.05, span * 0.85], dtype=np.float64)
        target = c + np.array([0.0, 0.0, 0.25 * span], dtype=np.float64)
        set_camera_view(eye=eye, target=target)

    def _design_scene(self):
        print("Designing scene")
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )
        if self.controller is not None:
            self.controller = self.controller.to(self.device)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        # Create gates based on track configuration
        scale = torch.ones(3) * self.gate_scale
        gate_positions_list = []
        gate_orientations_list = []
        
        # Config-based track: gates defined with positions and yaw angles
        # Sort gate keys to ensure correct order
        gate_keys = sorted(self.track_config.keys(), key=lambda x: int(x))
        
        for i, gate_key in enumerate(gate_keys):
            gate_cfg = self.track_config[gate_key]
            pos = gate_cfg.get("pos", (0.0, 0.0, 1.0))
            yaw = gate_cfg.get("yaw", 0.0)
            
            # Convert to torch tensors
            if isinstance(pos, (list, tuple)):
                gate_pos = torch.tensor(pos, device=self.device, dtype=torch.float32)
            else:
                gate_pos = torch.tensor([pos[0], pos[1], pos[2]], device=self.device, dtype=torch.float32)
            
            if isinstance(yaw, torch.Tensor):
                gate_yaw = yaw.item() if yaw.numel() == 1 else yaw
            else:
                gate_yaw = float(yaw)
            
            # Create quaternion from yaw (rotation around z-axis)
            gate_orientation = euler_to_quaternion(
                torch.tensor([0., 0., gate_yaw], device=self.device)
            )
            
            gate_positions_list.append(gate_pos)
            gate_orientations_list.append(gate_orientation)
            
            # Spawn gate using configured gate asset
            gate_prim = prim_utils.create_prim(
                f"/World/envs/env_0/Gate_{i}",
                usd_path=self.gate_asset_path,
                translation=(gate_pos[0].item(), gate_pos[1].item(), gate_pos[2].item()),
                orientation=(gate_orientation[0].item(), gate_orientation[1].item(), 
                                gate_orientation[2].item(), gate_orientation[3].item()),
                scale=scale
            )
            # Make gate static: disable gravity and make kinematic
            gate_prim_path = f"/World/envs/env_0/Gate_{i}"
            kit_utils.set_nested_rigid_body_properties(
                gate_prim_path,
                disable_gravity=True,
                linear_damping=1000.0,  # Very high damping to prevent movement
                angular_damping=1000.0,
            )
            # Set kinematic on all nested rigid bodies to prevent movement from collisions
            gate_prim_obj = prim_utils.get_prim_at_path(gate_prim_path)
            all_prims = [gate_prim_obj]
            while len(all_prims) > 0:
                child_prim = all_prims.pop(0)
                if child_prim.HasAttribute("physics:kinematicEnabled"):
                    child_prim.GetAttribute("physics:kinematicEnabled").Set(True)
                all_prims += child_prim.GetChildren()

        # Gate positions and orientations will be retrieved from views at runtime.
        # Store track bounds so we can aim the viewport at the whole course (default
        # cfg.viewer lookat/eye target the origin and miss spread-out gates).
        if gate_positions_list:
            pts = torch.stack(gate_positions_list, dim=0).cpu()
            self._track_cam_center_local = pts.mean(dim=0).numpy().astype("float64")
            lo = torch.amin(pts, dim=0)
            hi = torch.amax(pts, dim=0)
            span = torch.linalg.norm(hi - lo).item()
            self._track_cam_span = float(max(span, 6.0))
        else:
            self._track_cam_center_local = np.zeros(3, dtype=np.float64)
            self._track_cam_span = 12.0

        # Spawn drone at start position (behind first gate, in gate's local frame)
        # The gate's local x-axis points in the tangent direction (for circular) or forward (for config)
        # We want to spawn the drone behind the gate, so we offset backward along the gate's x-axis
        first_gate_pos = gate_positions_list[0]
        first_gate_rot = gate_orientations_list[0]
        # Offset backward in gate's local frame (negative x direction)
        offset_local = torch.tensor([-1.5, 0.0, self.gate_height / 2.0], device=self.device)
        # Rotate offset to world frame - quat_rotate expects batched inputs, so add batch dimension
        offset_world = quat_rotate(first_gate_rot.unsqueeze(0), offset_local.unsqueeze(0)).squeeze(0)
        start_pos = first_gate_pos + offset_world
        # Store first gate position as landing target after completing all laps
        self.first_gate_pos = first_gate_pos
        self.first_gate_rot = first_gate_rot
        self.drone.spawn(translations=[(start_pos[0].item(), start_pos[1].item(), start_pos[2].item())])


        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # Custom robot state: linear_vel(3) + rotation_matrix_flat(9) + angular_vel(3) = 15
        # Note: linear_vel uses BODY-FRAME velocity (see _build_robot_state).
        robot_state_dim = 3 + 9 + 3  # 15
        # Observation breakdown (33 total):
        #   robot_state(15): body_vel(3) + rot_mat(9) + ang_vel(3)
        #   prev_action(4): previous 4-dim action (body rates + thrust); helps learn smooth control
        #   dist_to_gate(1): scalar distance to gate center; pre-computed to avoid policy learning norm()
        #   next_gate_rpos_local(3): gate CENTER relative to drone in drone-local frame
        #   next_to_next_gate_pos(3): next-next gate CENTER in next gate's local frame
        #   next_gate_rot_mat_2col(6): gate orientation (x and y axes)
        #   gate_index_normalized(1): position on track [0,1] — required for gate-specific behaviour
        observation_dim = robot_state_dim + 4 + 1 + 3 + 3 + 6 + 1  # 33
        self.observation_spec = Composite({
            "agents": {
                "observation": Unbounded((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            },
            "info": {
                "drone_state": Unbounded((1, robot_state_dim), device=self.device),
            },
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": {
                "reward": Unbounded((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )
        stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "gates_passed": Unbounded(1),
            "drone_uprightness": Unbounded(1),
            "collision": Unbounded(1),
            "crashed_z": Unbounded(1),
            "crashed_z_gate": Unbounded(1),
            "crashed_distance": Unbounded(1),
            "success": BinaryDiscreteTensorSpec(1, dtype=bool),
            "truncated": Unbounded(1),
            # Speed tracking
            "mean_speed": Unbounded(1),         # mean linear speed (m/s) over episode
            "max_speed": Unbounded(1),          # max linear speed (m/s) seen in episode
            # Lap time (steps) — only meaningful when success=True
            "lap_time_steps": Unbounded(1),     # steps to complete lap (0 if no lap completed)
            # Reward component breakdown
            "reward_progress": Unbounded(1),    # cumulative progress reward
            "reward_speed": Unbounded(1),       # cumulative per-step speed reward term
            "reward_gates": Unbounded(1),       # cumulative gate passage reward
            "reward_stacked_direction": Unbounded(1), # cumulative signed stacked-gate local progress shaping
            "reward_stacked_bonus": Unbounded(1),     # cumulative stacked-gate entry/clear sparse bonuses
            "reward_penalties": Unbounded(1),   # cumulative angular + smoothness penalties
            "reward_altitude": Unbounded(1),    # cumulative altitude penalty term
            "reward_approach": Unbounded(1),    # cumulative near-gate retreat penalty term
            "reward_centering": Unbounded(1),   # cumulative near-gate centerline penalty term
            "reward_crash": Unbounded(1),       # cumulative crash penalties
            # Angular rate magnitude
            "mean_ang_rate": Unbounded(1),      # mean ||ang_vel|| over episode
            # Angular penalty decay (scalar, same for all envs)
            "ang_penalty_decay_frac": Unbounded(1),
            # Curriculum phase tracking
            "curriculum_phase": Unbounded(1),   # 0=accuracy-first, 1=bridge-speed, 2=speed-focus
            "curriculum_accuracy_ema": Unbounded(1),   # EMA full-lap completion rate used to unlock speed shaping
            "curriculum_furthest_ema": Unbounded(1),   # EMA furthest gate reached, kept as a diagnostic
            "reset_active_gate": Unbounded(1),         # earliest gate whose gate-0-start passage EMA is still below threshold
            "reset_from_gate0": Unbounded(1),          # episode started from gate 0 bucket
            "reset_from_prev_gate": Unbounded(1),      # episode started from previous-gate practice bucket
            "reset_from_random_gate": Unbounded(1),    # episode started from random-gate practice bucket
            "reset_gate_0_pass_ema": Unbounded(1),
            "reset_gate_1_pass_ema": Unbounded(1),
            "reset_gate_2_pass_ema": Unbounded(1),
            "reset_gate_3_pass_ema": Unbounded(1),
            "reset_gate_4_pass_ema": Unbounded(1),
            "reset_gate_5_pass_ema": Unbounded(1),
            "reset_gate_6_pass_ema": Unbounded(1),
            "reset_gate_7_pass_ema": Unbounded(1),
            "reset_gate_8_pass_ema": Unbounded(1),
            "reset_gate_9_pass_ema": Unbounded(1),
            "reset_gate_10_pass_ema": Unbounded(1),
            "reset_gate_11_pass_ema": Unbounded(1),
            # Per-gate visit counts — how many times each gate was crossed in the episode
            "gate_0_crosses": Unbounded(1),
            "gate_1_crosses": Unbounded(1),
            "gate_2_crosses": Unbounded(1),
            "gate_3_crosses": Unbounded(1),
            "gate_4_crosses": Unbounded(1),
            "gate_5_crosses": Unbounded(1),
            "gate_6_crosses": Unbounded(1),
            "gate_7_crosses": Unbounded(1),
            "gate_8_crosses": Unbounded(1),
            "gate_9_crosses": Unbounded(1),
            "gate_10_crosses": Unbounded(1),
            "gate_11_crosses": Unbounded(1),
            # Furthest gate reached (max gate index seen before crash/timeout)
            "furthest_gate": Unbounded(1),
            # Distance to gate at episode end (how close drone was when ep terminated)
            "final_dist_to_gate": Unbounded(1),
            # Action magnitude stats (how aggressively the drone is flying)
            "mean_action_magnitude": Unbounded(1),
            # Altitude stats
            "mean_altitude": Unbounded(1),
            "min_altitude": Unbounded(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        n = len(env_ids)

        # Reset curriculum: default to gate 0, but optionally allocate some resets
        # to the earliest weak gate's lead-in and a small random-gate coverage bucket.
        start_gates = torch.zeros(n, device=self.device, dtype=torch.long)
        reset_bucket = torch.zeros(n, device=self.device, dtype=torch.long)
        curriculum_active = (
            self.reset_curriculum_enabled
            and 0 <= self.active_reset_curriculum_gate < self.num_course_gates
        )
        if curriculum_active:
            random_draw = torch.rand(n, device=self.device)
            prev_bucket = (
                (random_draw >= self.reset_curriculum_gate0_prob)
                & (
                    random_draw
                    < self.reset_curriculum_gate0_prob + self.reset_curriculum_prev_gate_prob
                )
            )
            random_bucket = (
                random_draw
                >= self.reset_curriculum_gate0_prob + self.reset_curriculum_prev_gate_prob
            )
            prev_gate = (self.active_reset_curriculum_gate - 1) % self.num_course_gates
            start_gates[prev_bucket] = prev_gate
            if random_bucket.any():
                start_gates[random_bucket] = torch.randint(
                    0,
                    self.num_course_gates,
                    (int(random_bucket.sum().item()),),
                    device=self.device,
                )
            reset_bucket[prev_bucket] = 1
            reset_bucket[random_bucket] = 2

        # Reset gate progress
        self.gate_indices[env_ids] = start_gates
        self.episode_start_gate[env_ids] = start_gates
        self.episode_reset_bucket[env_ids] = reset_bucket
        self.gate_passed[env_ids] = False
        self.track_completed[env_ids] = False
        self.last_action[env_ids] = 0.0
        self.effort[env_ids] = 0.0
        # Give every spawned env a full grace window so the drone is not immediately
        # killed by the distance-crash check before it has a chance to orient itself.
        self.gate_just_passed_steps[env_ids] = self.crash_grace_steps
        self.stacked_reversal_grace[env_ids] = 0
        self.gates_crossed_this_ep[env_ids] = 0
        self.per_gate_crosses[env_ids] = 0
        self.furthest_gate_this_ep[env_ids] = start_gates
        self.min_altitude_this_ep[env_ids] = float('inf')
        self.mean_altitude_acc[env_ids] = 0.0
        self.mean_action_mag_acc[env_ids] = 0.0

        # Reset gate velocities to prevent drift
        gate_velocities = torch.zeros(n, self.num_gates, 6, device=self.device)
        self.gates.set_velocities(gate_velocities, env_indices=env_ids)

        try:
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()
            gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))
            gate_env_pos_reset = gate_env_pos[env_ids]   # (n, num_gates, 3)
            gate_env_rot_reset = gate_env_rot[env_ids]   # (n, num_gates, 4)

            # Select the start gate per environment
            batch_local = torch.arange(n, device=self.device)
            start_gate_pos = gate_env_pos_reset[batch_local, start_gates]  # (n, 3)
            start_gate_rot = gate_env_rot_reset[batch_local, start_gates]  # (n, 4)

            # Spawn drone 1.5m behind the start gate in gate-local frame
            offset_local_expanded = self.offset_local.unsqueeze(0).expand(n, -1)  # (n, 3)
            offset_world = quat_rotate(start_gate_rot, offset_local_expanded)     # (n, 3)
            drone_start_pos = start_gate_pos + offset_world                        # (n, 3)
            local_rpy_noise = self.init_rpy_dist.sample((*env_ids.shape, 1))
            local_rot_noise = euler_to_quaternion(local_rpy_noise)
            drone_rot = quat_mul(start_gate_rot.unsqueeze(1), local_rot_noise)

            # Store prev_drone_pos for path-projection reward
            self.prev_drone_pos[env_ids] = drone_start_pos

            drone_start_pos_with_agent = drone_start_pos.unsqueeze(1)             # (n, 1, 3)
            env_positions_with_agent = self.envs_positions[env_ids].unsqueeze(1)  # (n, 1, 3)

            self.drone.set_world_poses(
                drone_start_pos_with_agent + env_positions_with_agent,
                drone_rot, env_ids
            )
        except Exception as e:
            import traceback
            print("=" * 80)
            print(f"ERROR: Failed in _reset_idx (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            raise RuntimeError(
                f"DroneRaceEnv._reset_idx failed (num_envs={self.num_envs}, num_gates={self.num_gates})"
            ) from e

        self.drone.set_velocities(
            torch.zeros(n, 1, 6, device=self.device), env_ids
        )
        self.drone.set_joint_positions(torch.zeros(n, 1, 4, device=self.device), env_ids)
        self.drone.set_joint_velocities(torch.zeros(n, 1, 4, device=self.device), env_ids)

        # Compute start gate center for prev_distance_to_gate and crossing detection init
        gate_center_offset_local = torch.tensor([0.0, 0.0, self.gate_height / 2.0], device=self.device)
        gate_center_offset_local_expanded = gate_center_offset_local.unsqueeze(0).expand(n, -1)
        gate_center_offset_world = quat_rotate(start_gate_rot, gate_center_offset_local_expanded)
        start_gate_center = start_gate_pos + gate_center_offset_world

        self.prev_distance_to_gate[env_ids] = torch.norm(
            start_gate_center - drone_start_pos, dim=-1
        )

        # Initialise drone position in gate frame for crossing detection
        drone_to_gate_center = drone_start_pos - start_gate_center
        self.prev_drone_in_gate_frame[env_ids] = quat_rotate_inverse(
            start_gate_rot, drone_to_gate_center
        )

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False
        # Reset all stats on episode start
        self.stats["mean_speed"][env_ids] = 0.
        self.stats["max_speed"][env_ids] = 0.
        self.stats["lap_time_steps"][env_ids] = 0.
        self.stats["reward_progress"][env_ids] = 0.
        self.stats["reward_speed"][env_ids] = 0.
        self.stats["reward_gates"][env_ids] = 0.
        self.stats["reward_stacked_direction"][env_ids] = 0.
        self.stats["reward_stacked_bonus"][env_ids] = 0.
        self.stats["reward_penalties"][env_ids] = 0.
        self.stats["reward_altitude"][env_ids] = 0.
        self.stats["reward_approach"][env_ids] = 0.
        self.stats["reward_centering"][env_ids] = 0.
        self.stats["reward_crash"][env_ids] = 0.
        self.stats["mean_ang_rate"][env_ids] = 0.
        self.stats["ang_penalty_decay_frac"][env_ids] = 0.
        self.stats["curriculum_phase"][env_ids] = float(self.curriculum_phase)
        self.stats["reset_active_gate"][env_ids] = float(self.active_reset_curriculum_gate)
        self.stats["reset_from_gate0"][env_ids] = (reset_bucket == 0).float().unsqueeze(-1)
        self.stats["reset_from_prev_gate"][env_ids] = (reset_bucket == 1).float().unsqueeze(-1)
        self.stats["reset_from_random_gate"][env_ids] = (reset_bucket == 2).float().unsqueeze(-1)
        self.stats["furthest_gate"][env_ids] = 0.
        self.stats["final_dist_to_gate"][env_ids] = 0.
        self.stats["mean_altitude"][env_ids] = 0.
        self.stats["min_altitude"][env_ids] = 0.
        self.stats["mean_action_magnitude"][env_ids] = 0.
        for gi in range(12):
            self.stats[f"reset_gate_{gi}_pass_ema"][env_ids] = float(self.reset_gate_pass_ema[gi].item())
            self.stats[f"gate_{gi}_crosses"][env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        '''
        Input actions are in scaled units 
        '''
        actions = tensordict[("agents", "action")].clone()
        # The IndependentNormal distribution is unbounded, but the controller
        # expects actions in [-1, 1] (matching the Bounded action_spec).
        # Without this clamp, sampled actions can be arbitrarily large,
        # causing runaway body-rate demands and actuator saturation.
        actions = actions.clamp(-1.0, 1.0)
        if self.controller is not None:
            root_state = self.drone.get_state()[..., :13]
            raw_actions = self.controller.scaled_to_raw(actions)
            rotor_cmds = self.controller(root_state, *raw_actions)            
            _ = self.drone.apply_action(rotor_cmds)
        else:
            raise Exception("No controller found. This is not yet supported.")

    def _post_sim_step(self, tensordict: TensorDictBase):
        clamped = tensordict[("agents", "action")].clamp(-1.0, 1.0)
        self.effort = clamped
        self.last_action = clamped

    def _build_robot_state(self) -> torch.Tensor:
        """Builds the custom robot state vector used for observations.

        Calls ``drone.get_state()`` to refresh all cached kinematics, then
        concatenates along the feature dimension:

            [body_frame_linear_velocity (3) | rotation_matrix_flat (9) | angular_velocity (3)]

        Body-frame linear velocity is used instead of world-frame because:
        - Forward speed maps directly to the thrust axis (forward = +x in body frame)
        - Standard in drone racing literature (Swift Nature 2023, Song et al.)
        - World velocity can be reconstructed from body vel + rot_mat if needed

        Returns:
            Tensor of shape (N, 1, 15).
        """
        self.drone.get_state()  # refresh pos, rot, vel_w, vel_b caches
        lin_vel_body = self.drone.vel_b[..., :3]     # (N, 1, 3) — body-frame linear velocity
        rot_mat = self.drone.get_rotation_matrix()   # (N, 1, 9)
        ang_vel = self.drone.get_angular_velocity()  # (N, 1, 3) — world-frame angular velocity
        return torch.cat([lin_vel_body, rot_mat, ang_vel], dim=-1)  # (N, 1, 15)
        
    def get_relative_gate_position(self, gate_indices, gate_env_pos, gate_env_rot, drone_pos, drone_rot):
        """Return gate CENTER position/rotation relative to each drone, both in world and drone-local frames.

        Gate origins are at the bottom of the gate. All relative positions are computed to the gate
        CENTER (origin + gate_height/2 offset along gate local Z), matching the reward and crossing
        detection which also use _get_gate_center(). Previously the obs pointed at the origin, creating
        a 0.75m mismatch between what the policy aimed at and where it needed to fly.

        Args:
            gate_indices: (N,) integer tensor – the target gate index per environment.
            gate_env_pos: (N, num_gates, 3) gate positions in env frame.
            gate_env_rot: (N, num_gates, 4) gate rotations in env frame.
            drone_pos:    (N, 1, 3) drone positions.
            drone_rot:    (N, 1, 4) drone rotations.

        Returns:
            next_gate_pos:        (N, 1, 3) selected gate ORIGIN position in env frame (for rotation queries).
            next_gate_rot:        (N, 1, 4) selected gate rotation in env frame.
            next_gate_rpos_world: (N, 1, 3) gate CENTER relative to drone, in world frame.
            next_gate_rpos_local: (N, 1, 3) gate CENTER relative to drone, in drone-local frame.
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)

        next_gate_pos = gate_env_pos[batch_indices, gate_indices]  # (N, 3) — gate origin
        next_gate_rot = gate_env_rot[batch_indices, gate_indices]  # (N, 4)

        # Compute gate center: origin + height/2 offset along gate local Z axis
        next_gate_center = self._get_gate_center(next_gate_pos, next_gate_rot)  # (N, 3)

        # Expand to match agent dimension for broadcasting
        next_gate_pos = next_gate_pos.unsqueeze(1)    # (N, 1, 3) — origin kept for return (rotation queries)
        next_gate_rot = next_gate_rot.unsqueeze(1)    # (N, 1, 4)
        next_gate_center_1 = next_gate_center.unsqueeze(1)  # (N, 1, 3)

        # Relative position to gate CENTER in world frame
        next_gate_rpos_world = next_gate_center_1 - drone_pos  # (N, 1, 3)

        # Relative position to gate CENTER in drone-local frame
        drone_rot_flat = drone_rot.squeeze(1)                          # (N, 4)
        next_gate_rpos_world_flat = next_gate_rpos_world.squeeze(1)    # (N, 3)
        next_gate_rpos_local_flat = quat_rotate_inverse(drone_rot_flat, next_gate_rpos_world_flat)  # (N, 3)
        next_gate_rpos_local = next_gate_rpos_local_flat.unsqueeze(1)  # (N, 1, 3)

        return next_gate_pos, next_gate_rot, next_gate_rpos_world, next_gate_rpos_local

    def get_next_to_next_gate_position(self, next_to_next_gate_indices, gate_env_pos, gate_env_rot, next_gate_indices):
        """Return the CENTER of the next-to-next gate expressed relative to the CENTER of the next gate,
        rotated into the next gate's local frame.

        Uses gate centers (origin + height/2 offset) for both gates, consistent with how gate crossings
        and the reward are computed.

        Args:
            next_to_next_gate_indices: (N,) index of the gate after the immediate next gate (clamped at last gate).
            gate_env_pos:              (N, num_gates, 3) gate positions in env frame.
            gate_env_rot:              (N, num_gates, 4) gate rotations in env frame.
            next_gate_indices:         (N,) index of the immediate next gate per environment.

        Returns:
            (N, 3) position of the next-to-next gate CENTER relative to the next gate CENTER, in next gate's local frame.
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)

        # Positions/orientations of the immediate next gate in env frame
        next_gate_pos_env = gate_env_pos[batch_indices, next_gate_indices]      # (N, 3)
        next_gate_rot_env = gate_env_rot[batch_indices, next_gate_indices]      # (N, 4)

        # Position of the next-to-next gate in env frame
        n2n_gate_pos_env = gate_env_pos[batch_indices, next_to_next_gate_indices]  # (N, 3)
        n2n_gate_rot_env = gate_env_rot[batch_indices, next_to_next_gate_indices]  # (N, 4)

        # Compute gate centers for both
        next_gate_center = self._get_gate_center(next_gate_pos_env, next_gate_rot_env)    # (N, 3)
        n2n_gate_center = self._get_gate_center(n2n_gate_pos_env, n2n_gate_rot_env)       # (N, 3)

        # Relative position (center-to-center) in env frame
        rpos_env = n2n_gate_center - next_gate_center  # (N, 3)

        # Rotate into the next gate's local frame
        rpos_next_gate_frame = quat_rotate_inverse(next_gate_rot_env, rpos_env)  # (N, 3)

        return rpos_next_gate_frame


    def _compute_state_and_obs(self):
        import traceback
        import sys
        
        try:
            # Build custom state: [lin_vel(3) | rot_mat_flat(9) | ang_vel(3)] -> (N, 1, 15)
            # Also refreshes drone.pos / drone.rot caches used below for gate computations.
            self.drone_state = self._build_robot_state()
            drone_pos = self.drone.pos   # (N, 1, 3) refreshed by _build_robot_state
            drone_rot = self.drone.rot   # (N, 1, 4) refreshed by _build_robot_state

            # Get gate positions from views (similar to fly_through.py)
            # gates.get_world_poses() returns (pos, rot) with shape (num_envs, num_gates, ...)
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()  # (N, num_gates, 3), (N, num_gates, 4)
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed in _compute_state_and_obs (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f"ERROR: Failed in _compute_state_and_obs\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 80 + "\n")
            raise RuntimeError(
                f"DroneRaceEnv._compute_state_and_obs failed (num_envs={self.num_envs}, num_gates={self.num_gates})"
            ) from e

        gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))  # (N, num_gates, 3), (N, num_gates, 4)
        
        # If track is completed, target first gate for landing
        track_completed = self.track_completed  # (N,)
        
        # Get current gate positions for each environment
        target_gate_indices = self.gate_indices  # (N,)

        next_gate_pos, next_gate_rot, next_gate_rpos_world, next_gate_rpos_local = (
            self.get_relative_gate_position(
                target_gate_indices, gate_env_pos, gate_env_rot, drone_pos, drone_rot
            )
        )
        
        # Get next-to-next gate positions. If the immediate next gate is the last gate, clamp so
        # next-to-next points at the same gate (no wrap-around).
        next_to_next_gate_indices = torch.where(
            target_gate_indices == self.num_gates - 1,
            target_gate_indices,
            target_gate_indices + 1,
        )  # (N,)

        next_to_next_gate_pos = self.get_next_to_next_gate_position(
            next_to_next_gate_indices, gate_env_pos, gate_env_rot, target_gate_indices
        )  # (N, 3)

        # Angle between drone->gate vector and gate normal (gate local +x axis in env frame)
        gate_normal_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        gate_normal_local_expanded = gate_normal_local.unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)
        gate_normal_world = quat_rotate(next_gate_rot.squeeze(1), gate_normal_local_expanded)  # (N, 3)
        gate_vec_world = next_gate_rpos_world.squeeze(1)  # (N, 3)
        gate_vec_norm = torch.norm(gate_vec_world, dim=-1).clamp_min(1e-6)
        gate_normal_norm = torch.norm(gate_normal_world, dim=-1).clamp_min(1e-6)
        cos_angle = (gate_vec_world * gate_normal_world).sum(-1) / (gate_vec_norm * gate_normal_norm)
        gate_angle = torch.acos(cos_angle.clamp(-1.0, 1.0))  # (N,)

        # Gate progress: fraction of gates completed in the single lap
        gate_progress = self.gate_indices.float() / self.num_gates  # (N,)
        gate_progress = torch.where(track_completed, torch.ones_like(gate_progress), gate_progress)
        
        # Next gate orientation: first 2 columns of its rotation matrix in world frame.
        # Encodes the gate's local x- and y-axes in world coordinates (6-vector).
        # Avoids quaternion discontinuities and gives the policy an explicit sense of gate facing.
        next_gate_rot_flat = next_gate_rot.squeeze(1)  # (N, 4)
        e_x = torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)
        e_y = torch.tensor([0., 1., 0.], device=self.device).unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)
        gate_col0 = quat_rotate(next_gate_rot_flat, e_x)  # (N, 3) — gate x-axis in world frame
        gate_col1 = quat_rotate(next_gate_rot_flat, e_y)  # (N, 3) — gate y-axis in world frame
        next_gate_rot_mat_2col = torch.cat([gate_col0, gate_col1], dim=-1).unsqueeze(1)  # (N, 1, 6)

        # Gate index normalized to [0, 1]: tells the policy where on the track it is.
        # Sim-only exploit — needed for gate-specific behaviors (climb at gate 7, reverse at gate 8).
        gate_index_norm = (self.gate_indices.float() / (self.num_gates - 1)).unsqueeze(1).unsqueeze(1)  # (N, 1, 1)

        # Scalar distance to gate center (pre-computed to save the policy from learning norm())
        dist_to_gate_center = next_gate_rpos_world.norm(dim=-1, keepdim=True)  # (N, 1, 1)

        # Build observation
        # All components need to have the agent dimension (middle dimension) to match spec (N, 1, obs_dim)
        obs = [
            self.drone_state,                        # (N, 1, 15) body_vel(3) + rot_mat(9) + ang_vel(3)
            self.last_action,                        # (N, 1, 4)  previous action (body rates + thrust)
            dist_to_gate_center,                     # (N, 1, 1)  distance to gate center
            next_gate_rpos_local,                    # (N, 1, 3)  gate CENTER in drone-local frame
            next_to_next_gate_pos.unsqueeze(1),      # (N, 1, 3)  next-next gate CENTER in next gate frame
            next_gate_rot_mat_2col,                  # (N, 1, 6)  gate orientation
            gate_index_norm,                         # (N, 1, 1)  position on track [0,1]
        ]

        # Concatenate along last dimension: (N, 1, 33)
        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "info": {
                    "drone_state": self.drone_state,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _get_gate_center(self, gate_pos, gate_rot):
        """Compute gate center from gate origin (bottom-centre) by adding height/2 offset in gate frame.

        Args:
            gate_pos: (N, 3) gate origin positions in env frame.
            gate_rot: (N, 4) gate orientations (quaternions) in env frame.

        Returns:
            gate_center: (N, 3) gate centre positions in env frame.
        """
        offset_local = torch.tensor([0.0, 0.0, self.gate_height / 2.0], device=self.device)
        offset_local_expanded = offset_local.unsqueeze(0).expand(gate_pos.shape[0], -1)
        offset_world = quat_rotate(gate_rot, offset_local_expanded)
        return gate_pos + offset_world

    def _detect_gate_crossings(self, drone_pos_flat, current_gate_center, current_gate_rot,
                                gate_env_pos, gate_env_rot, batch_indices):
        """Detect whether the drone crossed through the current target gate this step.

        Uses plane-crossing (x: negative -> positive) plus bounding-box check
        (|y|, |z| < gate_width/2) in the gate frame centred at gate_center.

        Args:
            drone_pos_flat: (N, 3) current drone positions in env/world frame.
            current_gate_center: (N, 3) centre position of each env's current target gate
                in env/world frame.
            current_gate_rot: (N, 4) quaternion orientation of each env's current target gate
                in env/world frame.
            gate_env_pos: (N, G, 3) all gate origin positions for each env.
            gate_env_rot: (N, G, 4) all gate orientations (quaternions) for each env.
            batch_indices: (N,) environment indices used to gather the active gate per env.

        Side-effects: updates gate_indices, gate_passed, track_completed,
        prev_drone_in_gate_frame.

        Returns:
            gate_passed_this_step: (N,) bool — True for envs that just passed a gate.
            crossed_gate_idx:      (N,) long — gate index that was just crossed before any
                target advance. The duplicated lap-closure gate remains `num_course_gates`
                so downstream code can exclude it from per-gate accounting.
            gate_index_changed:    (N,) bool — True for envs whose target gate advanced.
            new_gate_center:       (N, 3) — centre of the (possibly new) target gate.
        """
        drone_to_gate = drone_pos_flat - current_gate_center
        curr_in_gate = quat_rotate_inverse(current_gate_rot, drone_to_gate)  # (N, 3)

        # Drone position 
        prev_x = self.prev_drone_in_gate_frame[..., 0]
        curr_x = curr_in_gate[..., 0]
        crossed_plane = (prev_x < 0) & (curr_x > 0)

        # TODO: Build `gates_passed_successfully` as a per-environment binary mask.
        # Shape: (N,) where N == self.num_envs.
        # Type: bool is preferred (True/False); equivalent to 1/0 when cast to int/float.
        # Gate width and height can be accessed by using self.gate_width and self.gate_height.
        # ----- ADD YOUR GATE-CROSSING MASK CODE BELOW (replace the placeholder) -----
        # curr_in_gate: (N, 3) drone position in gate local frame centered at gate_center.
        # x=forward through gate, y=lateral, z=up. Bounding box check in y and z.
        in_bounds_y = curr_in_gate[..., 1].abs() < (self.gate_width / 2.0)
        in_bounds_z = curr_in_gate[..., 2].abs() < (self.gate_height / 2.0)
        gates_passed_successfully = crossed_plane & in_bounds_y & in_bounds_z
        # ----- END STUDENT CODE -----
        
        gate_passed_this_step = gates_passed_successfully & (~self.gate_passed)
        self.gate_passed[gate_passed_this_step] = True

        old_gate_indices = self.gate_indices.clone()
        crossed_gate_idx = old_gate_indices.clone()
        last_gate_passed = gate_passed_this_step & (self.gate_indices + 1 >= self.num_gates)
        self.track_completed[last_gate_passed] = True
        self.gate_indices[gate_passed_this_step] = torch.clamp(
            self.gate_indices[gate_passed_this_step] + 1,
            max=self.num_gates - 1,
        )
        gate_index_changed = (self.gate_indices != old_gate_indices)
        self.gate_passed[gate_index_changed] = False

        # Recompute gate centre for envs whose target changed
        new_gate_pos = gate_env_pos[batch_indices, self.gate_indices]
        new_gate_rot = gate_env_rot[batch_indices, self.gate_indices]
        new_gate_center = self._get_gate_center(new_gate_pos, new_gate_rot)
        new_in_gate = quat_rotate_inverse(new_gate_rot, drone_pos_flat - new_gate_center)

        # old_prev_in_gate_frame = self.prev_drone_in_gate_frame.clone()
        self.prev_drone_in_gate_frame = torch.where(
            gate_index_changed.unsqueeze(-1), new_in_gate, curr_in_gate,
        )

        return gate_passed_this_step, crossed_gate_idx, gate_index_changed, new_gate_center


    def _compute_reward_and_done(self):
        import traceback
        import sys

        track_completed = self.track_completed  # (N,)

        try:
            # Use cached pose tensors directly; self.drone_state does not include position/quaternion.
            drone_pos = self.drone.pos  # (N, 1, 3), env frame
            drone_rot = self.drone.rot  # (N, 1, 4), quaternion

            gate_world_pos, gate_world_rot = self.gates.get_world_poses()
            gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))

            # Print debug positions in the same frame (env frame) for easier sanity checks.
            gate_idx0 = int(self.gate_indices[0].item())
            # print(f"Drone position (env frame): {torch.round(drone_pos[0, 0] * 100) / 100}")
            if self.debug_gate_origins:
                self._draw_gate_origins(gate_world_pos, gate_world_rot, env_idx=0)

            batch_indices = torch.arange(self.num_envs, device=self.device)
            current_gate_pos = gate_env_pos[batch_indices, self.gate_indices]  # (N, 3)
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed in _compute_reward_and_done (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            traceback.print_exc()
            raise

        current_gate_rot = gate_env_rot[batch_indices, self.gate_indices]  # (N, 4)
        current_gate_center = self._get_gate_center(current_gate_pos, current_gate_rot)  # (N, 3)

        drone_pos_flat = drone_pos.squeeze(1)  # (N, 3)
        distance_to_gate = torch.norm(drone_pos_flat - current_gate_center, dim=-1)  # (N,)

        # --- gate crossing detection ---
        # You either _deteect_gate_crossings or _detect_gate_crossings_via_segments
        # This function call updates the gate indexes
        prev_gate_frame = self.prev_drone_in_gate_frame.clone()
        gate_passed_this_step, crossed_gate_idx, gate_index_changed, new_gate_center = self._detect_gate_crossings(
            drone_pos_flat, current_gate_center, current_gate_rot,
            gate_env_pos, gate_env_rot, batch_indices,
        )

        # Increment per-episode gate crossing counter (actual crossings, not init index).
        self.gates_crossed_this_ep[gate_passed_this_step] += 1

        # Grace period after gate crossing: reset counter for envs that just advanced,
        # then decrement all. Dist-crash is disabled during the grace window so the drone
        # isn't immediately killed when the target gate jumps to the next gate (7.07m away).
        self.gate_just_passed_steps[gate_index_changed] = self.crash_grace_steps
        self.gate_just_passed_steps = (self.gate_just_passed_steps - 1).clamp(min=0)
        in_grace = self.gate_just_passed_steps > 0

        # Re-gather active target-gate geometry after possible gate advancement.
        current_gate_rot = gate_env_rot[batch_indices, self.gate_indices]
        current_gate_center = new_gate_center
        current_gate_local = quat_rotate_inverse(current_gate_rot, drone_pos_flat - current_gate_center)
        distance_to_gate = torch.norm(drone_pos_flat - current_gate_center, dim=-1)

        # -----------------------------------------------------------------------
        # STUDENT TODO (2/3): Implement your reward function.
        #
        # Variables available in this scope:
        #   drone_pos_flat        (N, 3)    drone position in env frame
        #   drone_rot             (N, 1, 4) drone orientation quaternion (w, x, y, z)
        #   distance_to_gate      (N,)      Euclidean distance to current gate centre
        #   current_gate_center   (N, 3)    3-D centre of the current target gate
        #   gate_passed_this_step (N,)      bool – True when drone just passed a gate
        #   gate_index_changed    (N,)      bool – True when target gate index advanced
        #   new_gate_center       (N, 3)    centre of the (possibly new) target gate
        #   crashed_collision     (N,)      bool – True when contact forces are detected
        #   self.drone.vel        (N, 1, 6) [lin_vel(3) | ang_vel(3)] in world frame
        #   self.gate_indices     (N,)      index of current target gate (0…num_gates-1)
        #   self.track_completed  (N,)      bool – True when the full lap is done
        #
        # Useful helpers (already imported at top of file):
        #   quat_axis(q, axis)          extract body axis vector (0=x, 1=y, 2=z)
        #   quat_rotate(q, v)           rotate vector v by quaternion q
        #   quat_rotate_inverse(q, v)   rotate vector v by inverse of quaternion q
        #
        # Your reward scalars are loaded in __init__ (STUDENT TODO 1/3) and
        # configured in cfg/task/DroneRace.yaml.
        # -----------------------------------------------------------------------
        # ----- ADD YOUR REWARD CODE BELOW (replace the placeholder) -----

        speed_shaping_enabled = self.curriculum_phase >= 1
        speed_focus_phase = self.curriculum_phase >= 2
        disable_dense_shaping = speed_focus_phase and self.phase2_disable_dense_shaping
        speed_phase_weight = 1.0 if speed_shaping_enabled else 0.0
        current_speed_scale = self.reward_speed_scale_phase2 if speed_focus_phase else self.reward_speed_scale
        current_speed_target = (
            self.reward_speed_target_ms_phase2 if speed_focus_phase else self.reward_speed_target_ms
        )
        sparse_zone_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for gate_idx in self.sparse_only_gate_indices:
            sparse_zone_mask |= self.gate_indices == gate_idx
        hard_turn_target_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for gate_idx in self.hard_turn_target_gates:
            hard_turn_target_mask |= self.gate_indices == gate_idx
        stacked_target_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for gate_idx in self.stacked_reversal_target_gates:
            stacked_target_mask |= self.gate_indices == gate_idx
        prev_target_indices = (self.gate_indices - 1).remainder(max(self.num_gates - 1, 1))
        stacked_entry_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for entry_gate, target_gate in self.stacked_reversal_gate_pairs:
            stacked_entry_mask |= (
                gate_index_changed
                & (self.gate_indices == target_gate)
                & (prev_target_indices == entry_gate)
            )
        if self.stacked_reversal_grace_steps > 0:
            self.stacked_reversal_grace[stacked_entry_mask] = self.stacked_reversal_grace_steps
        stacked_reversal_in_grace = self.stacked_reversal_grace > 0
        self.stacked_reversal_grace = (self.stacked_reversal_grace - 1).clamp(min=0)

        # 1. Path-projection progress reward (Song et al. 2021).
        #    Projects the drone's step displacement onto the unit vector from the previous
        #    gate center to the current target gate center. This is smooth across gate
        #    transitions (no discontinuity when the target switches) and rewards forward
        #    movement along the racing line rather than distance to a point.
        #    This stays on in phase 0 only as light guidance. Gate crossing rewards
        #    should dominate, otherwise the policy can learn to fly past gates.
        #    Placed AFTER _detect_gate_crossings so self.gate_indices is already updated.
        prev_gate_indices = (self.gate_indices - 1) % (self.num_gates - 1)  # (N,) — wraps at track
        prev_gate_pos_env = gate_env_pos[batch_indices, prev_gate_indices]   # (N, 3)
        prev_gate_rot_env = gate_env_rot[batch_indices, prev_gate_indices]   # (N, 4)
        prev_gate_center_proj = self._get_gate_center(prev_gate_pos_env, prev_gate_rot_env)  # (N, 3)
        gate_to_gate_vec = new_gate_center - prev_gate_center_proj           # (N, 3)
        gate_to_gate_norm = gate_to_gate_vec / (gate_to_gate_vec.norm(dim=-1, keepdim=True) + 1e-6)

        displacement = drone_pos_flat - self.prev_drone_pos                  # (N, 3)
        progress = (displacement * gate_to_gate_norm).sum(dim=-1)            # (N,)
        self.prev_drone_pos = drone_pos_flat.clone()

        progress_reward = self.reward_progress_scale * progress
        progress_reward = torch.where(
            sparse_zone_mask,
            torch.zeros_like(progress_reward),
            progress_reward,
        )
        if disable_dense_shaping:
            progress_reward = torch.zeros_like(progress_reward)
        reward = progress_reward.clone()

        # 2. Dense forward-speed shaping.
        #    Only turns on in phase 1 and stays capped so sparse gate rewards remain dominant.
        lin_vel_world = self.drone.vel[:, 0, :3]  # (N, 3) linear velocity in world frame
        speed_along_racing_line = (lin_vel_world * gate_to_gate_norm).sum(dim=-1).clamp(min=0.0)  # (N,)
        speed_reward = (
            speed_phase_weight
            * current_speed_scale
            * speed_along_racing_line.clamp(max=current_speed_target)
            * self.step_dt
        )
        speed_reward = torch.where(
            sparse_zone_mask,
            torch.zeros_like(speed_reward),
            speed_reward,
        )
        reward += speed_reward

        # 3. Sparse gate passage bonus.
        gate_reward = self.reward_gate_passage * gate_passed_this_step.float()
        reward += gate_reward

        real_gate_cross_mask = gate_passed_this_step & (crossed_gate_idx < self.num_course_gates)
        hard_turn_clear_mask = torch.zeros_like(gate_passed_this_step)
        for gate_idx in self.hard_turn_target_gates:
            hard_turn_clear_mask |= crossed_gate_idx == gate_idx
        hard_turn_clear_mask &= real_gate_cross_mask
        hard_turn_bonus_reward = self.reward_hard_turn_clear_bonus * hard_turn_clear_mask.float()
        reward += hard_turn_bonus_reward

        # 3b. Extra sparse bonuses around the hard stacked return gates.
        stacked_clear_mask = torch.zeros_like(gate_passed_this_step)
        for gate_idx in self.stacked_reversal_target_gates:
            stacked_clear_mask |= crossed_gate_idx == gate_idx
        stacked_clear_mask &= real_gate_cross_mask
        stacked_bonus_reward = (
            self.reward_stacked_entry_bonus * stacked_entry_mask.float()
            + self.reward_stacked_clear_bonus * stacked_clear_mask.float()
        )
        reward += stacked_bonus_reward

        # 4. Ordered sequence bonus.
        #    Gate crossings are only counted for the current target gate, so a non-zero
        #    bonus here means the drone is advancing through the course in order. The bonus
        #    grows with the current streak length to emphasize full consecutive laps.
        ordered_streak = self.gates_crossed_this_ep.float()
        sequence_reward = self.reward_gate_sequence_scale * ordered_streak * gate_passed_this_step.float()
        reward += sequence_reward

        # 5. Lap completion bonus with optional time-based speed incentive.
        #    The speed incentive stays off during the accuracy-first phase.
        #    time_fraction = fraction of episode budget remaining (1.0 = instant, 0.0 = at timeout).
        time_fraction = (self.max_episode_length - self.progress_buf).float() / self.max_episode_length
        lap_completion_reward = self.reward_lap_completion * self.track_completed.float()
        lap_speed_reward = speed_phase_weight * self.reward_lap_speed_bonus * time_fraction * self.track_completed.float()
        lap_reward = lap_completion_reward + lap_speed_reward
        reward += lap_reward

        # 5b. Signed local-x progress shaping inside the hard lower stacked gates only.
        # This is the only dense shaping reintroduced in the sparse stacked sections.
        stacked_corridor_mask = (
            stacked_target_mask
            & (~gate_index_changed)
            & (current_gate_local[:, 0] >= -4.0)
            & (current_gate_local[:, 0] <= 2.0)
            & (current_gate_local[:, 1].abs() <= self.gate_width)
            & (current_gate_local[:, 2].abs() <= self.gate_height)
        )
        stacked_delta_x = current_gate_local[:, 0] - prev_gate_frame[:, 0]
        stacked_direction_reward = torch.where(
            stacked_corridor_mask,
            self.reward_stacked_direction_scale * stacked_delta_x,
            torch.zeros_like(stacked_delta_x),
        )
        if disable_dense_shaping:
            stacked_direction_reward = torch.zeros_like(stacked_direction_reward)
        reward += stacked_direction_reward

        hard_turn_corridor_mask = (
            hard_turn_target_mask
            & (~gate_index_changed)
            & (current_gate_local[:, 0] >= self.hard_turn_corridor_x_min)
            & (current_gate_local[:, 0] <= self.hard_turn_corridor_x_max)
            & (current_gate_local[:, 1].abs() <= self.hard_turn_corridor_halfwidth)
            & (current_gate_local[:, 2].abs() <= self.hard_turn_corridor_height)
        )
        hard_turn_delta_x = current_gate_local[:, 0] - prev_gate_frame[:, 0]
        hard_turn_direction_reward = torch.where(
            hard_turn_corridor_mask,
            self.reward_hard_turn_direction_scale * hard_turn_delta_x,
            torch.zeros_like(hard_turn_delta_x),
        )
        if disable_dense_shaping:
            hard_turn_direction_reward = torch.zeros_like(hard_turn_direction_reward)
        reward += hard_turn_direction_reward

        # 6. Altitude mismatch penalty.
        #    Race track has two elevated gates (z=3.0) at gates 7 and 11. The path-projection
        #    reward gives near-zero signal on the purely-vertical 2m segments (gate 6→7, 10→11)
        #    because gate_to_gate_norm points almost entirely in Z. Instead of paying a bonus
        #    for being aligned, subtract a bounded penalty when the drone is at the wrong height.
        current_gate_z = current_gate_center[:, 2]  # (N,)
        # Gate centers: normal gates (origin z=1.0) have center z=1.75m (= 1.0 + gate_height/2 = 1.0 + 0.75);
        # elevated gates (origin z=3.0) have center z=3.75m. Threshold midpoint = 2.75m.
        # Only fires for gates 7 and 11 (origin z=3.0). gate_height=1.5 (original gate design).
        elevated_gate = current_gate_z > 2.75        # True only for elevated gates at origin z=3.0
        drone_z = drone_pos_flat[:, 2]               # (N,)
        altitude_error = (drone_z - current_gate_z).abs()
        altitude_penalty = torch.where(
            elevated_gate,
            self.reward_altitude_scale * (1.0 - torch.exp(-altitude_error)),
            torch.zeros_like(altitude_error)
        )
        altitude_penalty = torch.where(
            sparse_zone_mask,
            torch.zeros_like(altitude_penalty),
            altitude_penalty,
        )
        if disable_dense_shaping:
            altitude_penalty = torch.zeros_like(altitude_penalty)
        reward -= altitude_penalty

        # 7. Gate-centering penalty.
        #    When the drone is near the target gate plane, penalize lateral/vertical
        #    miss relative to the gate center. This keeps progress shaping from
        #    rewarding fly-bys that move along the track but miss the aperture.
        gate_plane_window = (current_gate_local[:, 0] > -6.0) & (current_gate_local[:, 0] < 2.0)
        gate_center_miss = torch.linalg.norm(current_gate_local[:, 1:3], dim=-1)
        centering_penalty = torch.where(
            gate_plane_window,
            self.reward_gate_centering_scale * (1.0 - torch.exp(-gate_center_miss)),
            torch.zeros_like(gate_center_miss),
        )
        centering_penalty = torch.where(
            sparse_zone_mask,
            torch.zeros_like(centering_penalty),
            centering_penalty,
        )
        centering_penalty = torch.where(
            hard_turn_corridor_mask,
            centering_penalty * self.hard_turn_penalty_scale,
            centering_penalty,
        )
        if disable_dense_shaping:
            centering_penalty = torch.zeros_like(centering_penalty)
        reward -= centering_penalty

        # 8. Near-gate retreat penalty.
        #    The 180° reversal at gate 7→8 (identical XY, only Z differs) causes path-projection
        #    progress to break down because the gate-to-gate direction reverses mid-approach.
        #    Instead of giving a bonus for small approach improvements, penalize the drone only
        #    when it retreats from the gate while already in the close-approach region.
        close_approach_mask = distance_to_gate < 6.0
        dist_improvement = self.prev_distance_to_gate - distance_to_gate  # positive = approaching
        retreat_amount = (-dist_improvement).clamp(min=0.0)
        approach_penalty = torch.where(
            close_approach_mask,
            self.reward_approach_scale * retreat_amount,
            torch.zeros_like(retreat_amount)
        )
        approach_penalty = torch.where(
            sparse_zone_mask,
            torch.zeros_like(approach_penalty),
            approach_penalty,
        )
        approach_penalty = torch.where(
            hard_turn_corridor_mask,
            approach_penalty * self.hard_turn_penalty_scale,
            approach_penalty,
        )
        if disable_dense_shaping:
            approach_penalty = torch.zeros_like(approach_penalty)
        reward -= approach_penalty
        # Update prev_distance_to_gate for next step's approach reward computation.
        self.prev_distance_to_gate = distance_to_gate.clone()

        # 9. Angular rate penalty with linear decay (Song et al.: used only in early training).
        #    Decays from reward_angular_penalty → 0 over angular_penalty_decay_frames steps.
        self.total_frames_counter += self.num_envs
        ang_decay_frac = max(0.0, 1.0 - self.total_frames_counter / self.angular_penalty_decay_frames)
        ang_vel = self.drone.vel[:, 0, 3:]  # (N, 3) angular velocity in world frame
        ang_penalty = ang_vel.pow(2).sum(dim=-1)  # (N,)
        angular_penalty = (self.reward_angular_penalty * ang_decay_frac) * ang_penalty
        if self.sparse_disable_smoothness_penalties:
            angular_penalty = torch.where(
                sparse_zone_mask,
                torch.zeros_like(angular_penalty),
                angular_penalty,
            )
        if disable_dense_shaping:
            angular_penalty = torch.zeros_like(angular_penalty)
        reward -= angular_penalty

        # 10. Action smoothness penalty.
        action_diff = self.drone.throttle_difference.squeeze(-1)  # (N,)
        action_smooth_penalty = self.reward_action_smooth_scale * action_diff
        if self.sparse_disable_smoothness_penalties:
            action_smooth_penalty = torch.where(
                sparse_zone_mask,
                torch.zeros_like(action_smooth_penalty),
                action_smooth_penalty,
            )
        if disable_dense_shaping:
            action_smooth_penalty = torch.zeros_like(action_smooth_penalty)
        reward -= action_smooth_penalty

        # ----- END STUDENT CODE -----

        # -----------------------------------------------------------------------
        # STUDENT TODO (3/3): Implement the crash / termination condition.
        # You might need to add geometric out-of-bounds checks.
        # -----------------------------------------------------------------------
        # ----- ADD YOUR CRASH CONDITION BELOW (replace the placeholder) -----

        # Physical collision detected via base_link contact forces.
        # drone.base_link is tracked with track_contact_forces=True (initialized at line 143).
        contact_forces = self.drone.base_link.get_net_contact_forces()  # (N, 1, 3)
        phys_crash = contact_forces.norm(dim=-1).squeeze(-1) > 0.1  # (N,)

        # Ground crash: altitude below minimum threshold.
        ground_crash = drone_pos_flat[:, 2] < self.crash_z_min  # (N,)

        # Out-of-bounds: drone is too far from its target gate (lost / diverged).
        # Grace period suppresses this check for crash_grace_steps steps after a gate crossing,
        # preventing false terminations when the target gate index just advanced (~7m away).
        stacked_distance_window = stacked_target_mask | stacked_reversal_in_grace
        distance_threshold = torch.full_like(distance_to_gate, self.crash_dist_threshold)
        distance_threshold = torch.where(
            stacked_distance_window,
            torch.full_like(distance_to_gate, self.stacked_reversal_dist_threshold),
            distance_threshold,
        )
        dist_crash = distance_to_gate > distance_threshold  # (N,)
        dist_crash &= ~in_grace
        dist_crash &= ~stacked_reversal_in_grace

        crashed = phys_crash | ground_crash | dist_crash

        # Apply crash penalty to reward.
        reward -= self.reward_crash_scale * crashed.float()

        # ----- END STUDENT CODE -----
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        completed_task = self.track_completed
        done = truncated | completed_task.unsqueeze(-1) | crashed.unsqueeze(-1)

        # --- Accuracy-first curriculum EMA update ---
        # On each episode termination, update the rolling averages used to decide when
        # to enable speed shaping. The unlock metric is full ordered-lap completion rate.
        done_flat = done.squeeze(-1)  # (N,)
        n_done = int(done_flat.sum().item())
        if n_done > 0:
            lap_completed_mean = completed_task[done_flat].float().mean().item()
            furthest_done = self.furthest_gate_this_ep[done_flat].float()
            furthest_mean = furthest_done.mean().item()
            alpha = self.curriculum_ema_alpha
            decay = (1.0 - alpha) ** n_done
            self.lap_completion_rate_ema = decay * self.lap_completion_rate_ema + (1.0 - decay) * lap_completed_mean
            self.furthest_gate_ema = decay * self.furthest_gate_ema + (1.0 - decay) * furthest_mean

            gate0_done_mask = done_flat & (self.episode_start_gate == 0)
            n_gate0_done = int(gate0_done_mask.sum().item())
            if n_gate0_done > 0:
                pass_matrix = (
                    self.per_gate_crosses[gate0_done_mask, :self.num_course_gates] > 0
                ).float()
                pass_mean = pass_matrix.mean(dim=0)
                reset_alpha = self.reset_curriculum_metric_alpha
                reset_decay = (1.0 - reset_alpha) ** n_gate0_done
                self.reset_gate_pass_ema[:self.num_course_gates] = (
                    reset_decay * self.reset_gate_pass_ema[:self.num_course_gates]
                    + (1.0 - reset_decay) * pass_mean
                )
                if self.reset_curriculum_enabled:
                    weak_gate_candidates = torch.nonzero(
                        self.reset_gate_pass_ema[:self.num_course_gates]
                        < self.reset_curriculum_gate_pass_threshold
                    ).squeeze(-1)
                    self.active_reset_curriculum_gate = (
                        int(weak_gate_candidates[0].item())
                        if weak_gate_candidates.numel() > 0
                        else -1
                    )
                else:
                    self.active_reset_curriculum_gate = -1

        # --- Curriculum phase transition ---
        # Speed shaping can unlock on the first success, or fall back to the original
        # accuracy-gated rule when phase_speed_unlock_on_first_success is disabled.
        frames_in_phase = self.total_frames_counter - self._phase_started_at_frames
        if self.curriculum_phase == 0:
            if self.phase_speed_unlock_on_first_success and completed_task.any():
                self.curriculum_phase = 1
                self._phase_started_at_frames = self.total_frames_counter
            elif frames_in_phase >= self.curriculum_min_phase_frames and \
               self.reset_gate_pass_ema[self.phase1_unlock_gate_index].item() >= self.phase1_unlock_gate_pass_rate:
                self.curriculum_phase = 1
                self._phase_started_at_frames = self.total_frames_counter
        elif self.curriculum_phase == 1 and \
             self.lap_completion_rate_ema >= self.phase2_speed_focus_accuracy_rate:
            self.curriculum_phase = 2
            self._phase_started_at_frames = self.total_frames_counter

        # --- stats ---
        self.stats["truncated"].add_(truncated.float())
        self.stats["collision"].add_(crashed.float().unsqueeze(-1))
        self.stats["success"].bitwise_or_(completed_task.unsqueeze(-1))
        self.stats["return"].add_(reward.unsqueeze(-1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["gates_passed"][:] = self.gates_crossed_this_ep.float().unsqueeze(1)
        # Granular crash diagnostics
        self.stats["crashed_z"].add_(ground_crash.float().unsqueeze(-1))
        self.stats["crashed_z_gate"].add_(phys_crash.float().unsqueeze(-1))
        self.stats["crashed_distance"].add_(dist_crash.float().unsqueeze(-1))
        # Uprightness: exponential moving average of the drone's up-vector z-component.
        self.stats["drone_uprightness"].lerp_(self.drone.up[..., 2], 1 - self.alpha)

        # --- extended stats for tuning ---
        # Speed: running mean and max of linear speed magnitude
        speed = lin_vel_world.norm(dim=-1)  # (N,)
        ep_len = self.progress_buf.float().clamp_min(1)  # avoid div-by-zero on step 0
        # Incremental mean: mean_n = mean_{n-1} + (x - mean_{n-1}) / n
        self.stats["mean_speed"].add_((speed.unsqueeze(-1) - self.stats["mean_speed"]) / ep_len.unsqueeze(-1))
        self.stats["max_speed"] = torch.maximum(self.stats["max_speed"], speed.unsqueeze(-1))
        # Lap time: record step count when lap is first completed
        just_completed = completed_task & (self.stats["lap_time_steps"].squeeze(-1) == 0)
        self.stats["lap_time_steps"][just_completed] = self.progress_buf[just_completed].float().unsqueeze(-1)
        # Reward component breakdown (cumulative)
        gate_reward_total = gate_reward + sequence_reward + lap_reward + hard_turn_bonus_reward  # (N,) includes ordered streak + lap bonuses
        penalty_reward = angular_penalty + action_smooth_penalty  # (N,)
        crash_reward = self.reward_crash_scale * crashed.float()  # (N,)
        self.stats["reward_progress"].add_((progress_reward + hard_turn_direction_reward).unsqueeze(-1))
        self.stats["reward_speed"].add_(speed_reward.unsqueeze(-1))
        self.stats["reward_gates"].add_(gate_reward_total.unsqueeze(-1))
        self.stats["reward_stacked_direction"].add_(stacked_direction_reward.unsqueeze(-1))
        self.stats["reward_stacked_bonus"].add_(stacked_bonus_reward.unsqueeze(-1))
        self.stats["reward_penalties"].add_(penalty_reward.unsqueeze(-1))
        self.stats["reward_altitude"].add_((-altitude_penalty).unsqueeze(-1))
        self.stats["reward_approach"].add_((-approach_penalty).unsqueeze(-1))
        self.stats["reward_centering"].add_((-centering_penalty).unsqueeze(-1))
        self.stats["reward_crash"].add_(crash_reward.unsqueeze(-1))
        # Angular rate magnitude (incremental mean)
        ang_rate_mag = ang_vel.norm(dim=-1)  # (N,)
        self.stats["mean_ang_rate"].add_((ang_rate_mag.unsqueeze(-1) - self.stats["mean_ang_rate"]) / ep_len.unsqueeze(-1))
        # Angular penalty decay fraction (same scalar for all envs)
        self.stats["ang_penalty_decay_frac"][:] = ang_decay_frac

        # --- new rich diagnostics ---
        # Furthest gate reached this episode (max gate index seen)
        self.furthest_gate_this_ep = torch.maximum(self.furthest_gate_this_ep, self.gate_indices)
        self.stats["furthest_gate"][:] = self.furthest_gate_this_ep.float().unsqueeze(1)

        # Per-gate crossing counts (gates 0–11, clamp index to valid range)
        if gate_passed_this_step.any():
            real_gate_cross_mask = gate_passed_this_step & (crossed_gate_idx < self.num_course_gates)
            safe_crossed_gate_idx = crossed_gate_idx.clamp(0, self.num_course_gates - 1)
            per_gate_update = torch.zeros(self.num_envs, 12, device=self.device)
            per_gate_update.scatter_add_(
                1,
                safe_crossed_gate_idx.unsqueeze(1),
                real_gate_cross_mask.float().unsqueeze(1)
            )
            self.per_gate_crosses += per_gate_update.long()
        for gi in range(12):
            self.stats[f"gate_{gi}_crosses"][:] = self.per_gate_crosses[:, gi].float().unsqueeze(1)

        # Curriculum phase (0=accuracy-first, 1=bridge-speed, 2=speed-focus).
        self.stats["curriculum_phase"][:] = float(self.curriculum_phase)
        self.stats["curriculum_accuracy_ema"][:] = float(self.lap_completion_rate_ema)
        self.stats["curriculum_furthest_ema"][:] = float(self.furthest_gate_ema)
        self.stats["reset_active_gate"][:] = float(self.active_reset_curriculum_gate)
        self.stats["reset_from_gate0"][:] = (self.episode_reset_bucket == 0).float().unsqueeze(-1)
        self.stats["reset_from_prev_gate"][:] = (self.episode_reset_bucket == 1).float().unsqueeze(-1)
        self.stats["reset_from_random_gate"][:] = (self.episode_reset_bucket == 2).float().unsqueeze(-1)
        for gi in range(12):
            self.stats[f"reset_gate_{gi}_pass_ema"][:] = float(self.reset_gate_pass_ema[gi].item())

        # Distance to current gate at end of episode (meaningful when episode ends)
        self.stats["final_dist_to_gate"][:] = distance_to_gate.unsqueeze(1)

        # Altitude stats: min and incremental mean
        self.min_altitude_this_ep = torch.minimum(self.min_altitude_this_ep, drone_z)
        self.mean_altitude_acc.add_((drone_z - self.mean_altitude_acc) / ep_len)
        min_alt_clamped = self.min_altitude_this_ep.clone()
        min_alt_clamped[min_alt_clamped == float('inf')] = 0.0
        self.stats["min_altitude"][:] = min_alt_clamped.unsqueeze(1)
        self.stats["mean_altitude"][:] = self.mean_altitude_acc.unsqueeze(1)

        # Action magnitude (incremental mean of L2 norm of action vector)
        action_mag = self.effort.squeeze(1).norm(dim=-1)  # (N,)
        self.mean_action_mag_acc.add_((action_mag - self.mean_action_mag_acc) / ep_len)
        self.stats["mean_action_magnitude"][:] = self.mean_action_mag_acc.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1).unsqueeze(-1),
                },
                "done": done,
                "terminated": crashed.unsqueeze(-1),
                "truncated": truncated,
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )
