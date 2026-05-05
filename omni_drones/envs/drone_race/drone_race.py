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


import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec, BinaryDiscreteTensorSpec

import isaacsim.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate, quat_rotate_inverse, quat_axis
from omni_drones.utils.drone_race_geometry import (
    detect_gate_crossings_from_frames,
    detect_two_body_gate_completion,
)
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

    - `drone_state` (15): `[lin_vel(3) | rot_mat_flat(9) | ang_vel(3)]`.
    - Current target gate center relative to the drone, current centered gate-frame position,
      current gate axes, and normalized ordered-gate progress.
    - Lookahead features for the next configured gates: relative center positions plus gate axes.
    - Virtual pre/center/post waypoint vectors for the current gate.
    - Previous action, which helps PPO learn smoother body-rate/thrust commands.

    ## Reward

    Dense progress, aperture/centerline shaping, ordered-gate bonuses, full-lap completion,
    time pressure, forward-speed shaping, and crash/wrong-gate/stall penalties.

    ## Episode End

    The episode ends when the drone crashes, crosses the wrong gate, misses the
    target gate aperture, stalls, leaves course bounds, completes the full lap,
    or reaches the maximum episode length.

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
    def __init__(self, cfg, headless):
        self.lookahead_gates = int(cfg.task.get("lookahead_gates", 3))
        self.waypoint_offset = float(cfg.task.get("waypoint_offset", 1.75))
        self.start_distance = float(cfg.task.get("start_distance", 2.0))
        self.start_lateral_noise = float(cfg.task.get("start_lateral_noise", 0.0))
        self.start_vertical_noise = float(cfg.task.get("start_vertical_noise", 0.0))
        self.curriculum_stage = int(cfg.task.get("curriculum_stage", 1))
        self.random_start_prob = float(cfg.task.get("random_start_prob", 0.0))
        self.gate_pass_thresh = float(cfg.task.get("curriculum_gate_pass_threshold", 0.80))
        self.course_pass_thresh = float(cfg.task.get("curriculum_course_pass_threshold", 0.80))
        self.ema_alpha = float(cfg.task.get("curriculum_ema_alpha", 0.05))
        # Phase 0 parameters: uniform random starts until mean gate EMA ≥ threshold or frames exceeded
        self.curriculum_phase0_threshold = float(cfg.task.get("curriculum_phase0_threshold", 0.15))
        self.curriculum_phase0_frames = int(cfg.task.get("curriculum_phase0_frames", 30_000_000))
        payload_cfg = cfg.task.get("payload", {}) or {}
        self.payload_enabled = bool(payload_cfg.get("enabled", False))
        self.payload_bar_length = float(payload_cfg.get("bar_length", 1.0))
        self.payload_mass = float(payload_cfg.get("mass", 0.30))
        self.payload_radius = float(payload_cfg.get("radius", 0.04))

        self.reward_progress_scale = float(cfg.task.get("reward_progress_scale", 4.0))
        self.reward_centerline_scale = float(cfg.task.get("reward_centerline_scale", 0.7))
        self.reward_payload_centerline_scale = float(cfg.task.get("reward_payload_centerline_scale", 0.9))
        self.reward_gate_passage = float(cfg.task.get("reward_gate_passage", 25.0))
        self.reward_drone_gate_entry = float(cfg.task.get("reward_drone_gate_entry", 5.0))
        self.reward_completion = float(cfg.task.get("reward_completion", 250.0))
        self.reward_time_penalty = float(cfg.task.get("reward_time_penalty", 0.015))
        self.reward_speed_scale = float(cfg.task.get("reward_speed_scale", 0.04))
        self.reward_effort_weight = float(cfg.task.get("reward_effort_weight", 0.002))
        self.reward_action_smoothness_weight = float(cfg.task.get("reward_action_smoothness_weight", 0.01))
        self.reward_payload_swing_weight = float(cfg.task.get("reward_payload_swing_weight", 0.08))
        self.reward_payload_lateral_velocity_weight = float(
            cfg.task.get("reward_payload_lateral_velocity_weight", 0.02)
        )
        self.reward_crash = float(cfg.task.get("reward_crash", 80.0))
        self.reward_wrong_gate = float(cfg.task.get("reward_wrong_gate", 120.0))
        self.reward_payload_miss = float(cfg.task.get("reward_payload_miss", 120.0))
        self.reward_stall = float(cfg.task.get("reward_stall", 30.0))

        self.progress_clamp = float(cfg.task.get("progress_clamp", 3.0))
        self.speed_reward_cap = float(cfg.task.get("speed_reward_cap", 20.0))
        self.bounds_margin_xy = float(cfg.task.get("bounds_margin_xy", 8.0))
        self.bounds_min_z = float(cfg.task.get("bounds_min_z", 0.15))
        self.bounds_max_z = float(cfg.task.get("bounds_max_z", 7.0))
        self.contact_force_threshold = float(cfg.task.get("contact_force_threshold", 1e-4))
        self.stall_speed = float(cfg.task.get("stall_speed", 0.2))
        self.stall_grace_steps = int(cfg.task.get("stall_grace_steps", 150))
        self.stall_distance = float(cfg.task.get("stall_distance", 2.0))
        self.gate_width = float(cfg.task.get("gate_width", 1.0))

        self.gate_scale = cfg.task.gate_scale
        
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

        if self.payload_enabled:
            try:
                self.payload = RigidPrimView(
                    f"/World/envs/env_*/{self.drone.name}_*/payload",
                    reset_xform_properties=False,
                    track_contact_forces=True,
                )
                self.payload.initialize()
                print("[DroneRaceEnv] payload.initialize() completed")
            except Exception as e:
                print("=" * 80)
                print("ERROR: Failed to initialize slung payload")
                print("=" * 80)
                traceback.print_exc()
                print("=" * 80)
                raise
        else:
            self.payload = None

        self.hover_cmd_thrust = None

        print(f"[DroneRaceEnv] num_envs={self.num_envs}, num_gates={self.num_gates}")
        # Track gate progress for each environment
        self.gate_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.start_gate_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gate_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.gate_completion_flags = torch.zeros(
            self.num_envs, self.num_gates, device=self.device, dtype=torch.bool
        )
        self.track_completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.prev_distance_to_gate = torch.zeros(self.num_envs, device=self.device)
        self.prev_payload_distance_to_gate = torch.zeros(self.num_envs, device=self.device)
        self.prev_drone_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_payload_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.speed_sum = torch.zeros(self.num_envs, device=self.device)
        self.max_speed_buf = torch.zeros(self.num_envs, device=self.device)
        self.payload_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self.payload_max_speed_buf = torch.zeros(self.num_envs, device=self.device)
        self.payload_swing_sum = torch.zeros(self.num_envs, device=self.device)
        self.payload_max_swing_buf = torch.zeros(self.num_envs, device=self.device)
        self.action_smoothness = torch.zeros(self.num_envs, 1, device=self.device)
        # Gate crossing detection: drone position in centered gate frame from previous step
        self.prev_drone_in_gate_frame = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_payload_in_gate_frame = torch.zeros(self.num_envs, 3, device=self.device)
        # Anti-cheat: drone must approach from behind gate (x < -waypoint_offset) before crossing counts
        self.gate_approach_confirmed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Per-gate EMA pass rates for curriculum tracking (shape: [num_gates])
        self.gate_ema_pass_rate = torch.zeros(self.num_gates, device=self.device)
        self.last_action = torch.zeros(self.num_envs, 1, self.drone.action_spec.shape[-1], device=self.device)
        self.effort = torch.zeros(self.num_envs, 1, self.drone.action_spec.shape[-1], device=self.device) 
        self.current_gate_frame_pos = torch.zeros(self.num_envs, 3, device=self.device)

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
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([.2, .2, 0.], device=self.device) * torch.pi
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

        # Gate positions and orientations will be retrieved from views at runtime
        # No need to store them manually since gates are static

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

        if self.payload_enabled:
            from omni_drones.envs.payload.utils import attach_payload

            attach_payload(
                f"/World/envs/env_0/{self.drone.name}_0",
                self.payload_bar_length,
                payload_radius=self.payload_radius,
                payload_mass=self.payload_mass,
            )

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # Custom robot state: linear_vel(3) + rotation_matrix_flat(9) + angular_vel(3)
        robot_state_dim = 3 + 9 + 3
        action_dim = self.drone.action_spec.shape[-1]
        lookahead_dim = self.lookahead_gates * (3 + 6)
        waypoint_dim = (self.lookahead_gates + 1) * 3 * 3
        progress_dim = 2
        payload_dim = 20 if self.payload_enabled else 0
        observation_dim = (
            robot_state_dim
            + 3  # current gate center relative to drone, drone frame
            + 3  # drone position in centered current-gate frame
            + 6  # current gate x/y axes in world frame
            + lookahead_dim
            + waypoint_dim
            + progress_dim
            + payload_dim
            + action_dim
        )
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
        stats_entries = {
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "gates_passed": Unbounded(1),
            "lap_time_sec": Unbounded(1),
            "mean_speed": Unbounded(1),
            "max_speed": Unbounded(1),
            "progress_reward": Unbounded(1),
            "collision": Unbounded(1),
            "wrong_gate": Unbounded(1),
            "crashed_bounds": Unbounded(1),
            "stalled": Unbounded(1),
            "payload_collision": Unbounded(1),
            "payload_miss": Unbounded(1),
            "payload_mean_speed": Unbounded(1),
            "payload_max_speed": Unbounded(1),
            "payload_mean_swing_angle": Unbounded(1),
            "payload_max_swing_angle": Unbounded(1),
            "drone_gate_entry": Unbounded(1),
            "start_gate_index": Unbounded(1),
            "full_course_start": Unbounded(1),
            "success": BinaryDiscreteTensorSpec(1, dtype=bool),
            "truncated":  Unbounded(1),
        }
        for gate_idx in range(self.num_gates):
            stats_entries[f"gate_{gate_idx + 1:02d}_passed"] = Unbounded(1)
        stats_spec = Composite(stats_entries).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _sample_start_gate_indices(self, env_ids: torch.Tensor) -> torch.Tensor:
        count = len(env_ids)
        if self.curriculum_stage >= 2:
            return torch.zeros(count, dtype=torch.long, device=self.device)

        # Stage 0: pure uniform random across all gates — no bias toward gate 0
        if self.curriculum_stage == 0:
            return torch.randint(0, self.num_gates, (count,), device=self.device)

        # Stage 1: 60% gate 0 / 30% weakest-gate weighted / 10% uniform random
        r = torch.rand(count, device=self.device)
        result = torch.zeros(count, dtype=torch.long, device=self.device)

        # 30% band: sample from weak gates weighted by shortfall
        mid_band = (r >= 0.60) & (r < 0.90)
        if mid_band.any():
            weak_mask = self.gate_ema_pass_rate < self.gate_pass_thresh
            if weak_mask.any():
                shortfall = (self.gate_pass_thresh - self.gate_ema_pass_rate).clamp(min=0)
                shortfall = shortfall * weak_mask.float()
                n_mid = int(mid_band.sum().item())
                weak_indices = torch.multinomial(
                    shortfall.unsqueeze(0).expand(n_mid, -1),
                    num_samples=1,
                ).squeeze(1)
                result[mid_band] = weak_indices
            # else: all gates mastered — keep gate 0 for this band

        # 10% band: uniform random across all gates
        high_band = r >= 0.90
        if high_band.any():
            result[high_band] = torch.randint(
                0, self.num_gates, (int(high_band.sum().item()),), device=self.device
            )

        return result

    def update_curriculum(self, gate_pass_rates: dict, full_course_rate, env_frames: int = None) -> dict:
        """Update per-gate EMA pass rates and auto-advance curriculum stage.

        Args:
            gate_pass_rates: {gate_index (int): pass_rate (float)} for gates 1..N (1-indexed keys from stats).
            full_course_rate: fraction of envs completing a full lap this batch, or None.
            env_frames: total environment frames processed so far, used for phase 0 time-based fallback.
        Returns:
            Dict of curriculum metrics to merge into the W&B log dict.
        """
        alpha = self.ema_alpha
        for gate_idx, rate in gate_pass_rates.items():
            i = int(gate_idx) - 1  # stats keys are 1-indexed (gate01..gate13)
            if 0 <= i < self.num_gates:
                self.gate_ema_pass_rate[i] = (
                    (1 - alpha) * self.gate_ema_pass_rate[i] + alpha * float(rate)
                )

        mean_ema = self.gate_ema_pass_rate.mean().item()
        transitioned = False
        if self.curriculum_stage == 0:
            # Advance to stage 1 when mean gate EMA hits threshold OR frame budget exhausted
            frames_exceeded = env_frames is not None and env_frames >= self.curriculum_phase0_frames
            if mean_ema >= self.curriculum_phase0_threshold or frames_exceeded:
                self.curriculum_stage = 1
                transitioned = True
        elif self.curriculum_stage == 1:
            if (self.gate_ema_pass_rate >= self.gate_pass_thresh).all():
                self.curriculum_stage = 2
                transitioned = True
        elif self.curriculum_stage == 2:
            if full_course_rate is not None and float(full_course_rate) >= self.course_pass_thresh:
                self.curriculum_stage = 3
                transitioned = True

        metrics = {
            "curriculum/stage": self.curriculum_stage,
            "curriculum/transition": int(transitioned),
            "curriculum/mean_ema_rate": mean_ema,
            "curriculum/num_weak_gates": int(
                (self.gate_ema_pass_rate < self.gate_pass_thresh).sum().item()
            ),
        }
        for i in range(self.num_gates):
            metrics[f"curriculum/gate{i + 1:02d}_ema_rate"] = self.gate_ema_pass_rate[i].item()
        return metrics

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        
        # Reset gate progress
        start_gate_indices = self._sample_start_gate_indices(env_ids)
        self.gate_indices[env_ids] = start_gate_indices
        self.start_gate_indices[env_ids] = start_gate_indices
        self.gate_passed[env_ids] = False
        self.gate_completion_flags[env_ids] = False
        self.track_completed[env_ids] = False
        self.gate_approach_confirmed[env_ids] = False
        self.last_action[env_ids] = 0.0
        self.effort[env_ids] = 0.0
        self.action_smoothness[env_ids] = 0.0
        self.speed_sum[env_ids] = 0.0
        self.max_speed_buf[env_ids] = 0.0
        self.payload_speed_sum[env_ids] = 0.0
        self.payload_max_speed_buf[env_ids] = 0.0
        self.payload_swing_sum[env_ids] = 0.0
        self.payload_max_swing_buf[env_ids] = 0.0

        # Reset gate velocities to prevent drift (gates are static, so we just zero velocities)
        # Set velocities to zero for all gates in reset environments
        num_gates_to_reset = len(env_ids) * self.num_gates
        gate_velocities = torch.zeros(num_gates_to_reset, 6, device=self.device)
        
        # Reshape to match gate view shape: (num_envs, num_gates, 6)
        gate_velocities = gate_velocities.reshape(len(env_ids), self.num_gates, 6)
        self.gates.set_velocities(gate_velocities, env_indices=env_ids)

        try:
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()  # (num_envs, num_gates, 3), (num_envs, num_gates, 4)
            gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))  # (N, num_gates, 3), (N, num_gates, 4)
            gate_env_pos = gate_env_pos[env_ids]  # (len(env_ids), num_gates, 3)
            gate_env_rot = gate_env_rot[env_ids]  # (len(env_ids), num_gates, 4)
            gate_centers = self._get_all_gate_centers(gate_env_pos, gate_env_rot)
            local_batch = torch.arange(len(env_ids), device=self.device)
            target_gate_center = gate_centers[local_batch, start_gate_indices]
            target_gate_rot = gate_env_rot[local_batch, start_gate_indices]

            lateral_noise = (
                torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0
            ) * self.start_lateral_noise
            vertical_noise = (
                torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0
            ) * self.start_vertical_noise
            offset_local = torch.stack(
                [
                    torch.full((len(env_ids),), -self.start_distance, device=self.device),
                    lateral_noise,
                    vertical_noise,
                ],
                dim=-1,
            )
            offset_world = quat_rotate(target_gate_rot, offset_local)
            drone_start_pos = target_gate_center + offset_world
            
            drone_start_pos_with_agent = drone_start_pos.unsqueeze(1)  # (len(env_ids), 1, 3)
            env_positions_with_agent = self.envs_positions[env_ids].unsqueeze(1)  # (len(env_ids), 1, 3)

            self.drone.set_world_poses(
                drone_start_pos_with_agent + env_positions_with_agent,
                target_gate_rot.unsqueeze(1), env_ids
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
            torch.zeros_like(self.init_vels[env_ids]), env_ids
        )

        self.drone.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.drone.set_joint_velocities(self.init_joint_vels[env_ids], env_ids)
        if self.payload_enabled:
            self.payload.set_masses(
                torch.full((len(env_ids),), self.payload_mass, device=self.device),
                env_indices=env_ids,
            )

        self.prev_distance_to_gate[env_ids] = torch.norm(
            target_gate_center - drone_start_pos,
            dim=-1
        )
        self.prev_drone_pos[env_ids] = drone_start_pos
        if self.payload_enabled:
            payload_start_pos = drone_start_pos + quat_rotate(
                target_gate_rot,
                torch.tensor(
                    [0.0, 0.0, -self.payload_bar_length],
                    device=self.device,
                ).unsqueeze(0).expand(len(env_ids), -1),
            )
            self.prev_payload_pos[env_ids] = payload_start_pos
            self.prev_payload_distance_to_gate[env_ids] = torch.norm(
                target_gate_center - payload_start_pos,
                dim=-1,
            )
        else:
            self.prev_payload_pos[env_ids] = drone_start_pos
            self.prev_payload_distance_to_gate[env_ids] = self.prev_distance_to_gate[env_ids]

        # Initialise drone position in gate frame for crossing detection
        drone_to_first_gate_center = drone_start_pos - target_gate_center  # (len(env_ids), 3)
        self.prev_drone_in_gate_frame[env_ids] = quat_rotate_inverse(
            target_gate_rot, drone_to_first_gate_center
        )  # (len(env_ids), 3)
        self.prev_payload_in_gate_frame[env_ids] = quat_rotate_inverse(
            target_gate_rot,
            self.prev_payload_pos[env_ids] - target_gate_center,
        )

        for key in self.stats.keys():
            self.stats[key][env_ids] = torch.zeros_like(self.stats[key][env_ids])

    def _pre_sim_step(self, tensordict: TensorDictBase):
        '''
        Input actions are in scaled units 
        '''
        actions = tensordict[("agents", "action")].clone()
        if self.controller is not None:
            root_state = self.drone.get_state()[..., :13]
            raw_actions = self.controller.scaled_to_raw(actions)
            rotor_cmds = self.controller(root_state, *raw_actions)            
            _ = self.drone.apply_action(rotor_cmds)
        else:
            raise Exception("No controller found. This is not yet supported.")

    def _post_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")].clone()
        self.action_smoothness = torch.norm(actions - self.last_action, dim=-1)
        self.effort = actions
        self.last_action = actions

    def _build_robot_state(self) -> torch.Tensor:
        """Builds the custom robot state vector used for observations.

        Calls ``drone.get_state()`` to refresh all cached kinematics, then concatenates:

            [linear_velocity (3) | rotation_matrix_flat (9) | angular_velocity (3)]

        Returns:
            Tensor of shape (N, 1, 15).
        """
        self.drone.get_state()  # refresh pos, rot, vel_w, vel_b caches
        lin_vel = self.drone.get_linear_velocity()   # (N, 1, 3)
        rot_mat = self.drone.get_rotation_matrix()   # (N, 1, 9)
        ang_vel = self.drone.get_angular_velocity()  # (N, 1, 3)
        return torch.cat([lin_vel, rot_mat, ang_vel], dim=-1)  # (N, 1, 15)
        
    def get_relative_gate_position(self, gate_indices, gate_env_pos, gate_env_rot, drone_pos, drone_rot):
        """Return gate position/rotation relative to each drone, both in world and drone-local frames.

        Args:
            gate_indices: (N,) integer tensor – the target gate index per environment.
            gate_env_pos: (N, num_gates, 3) gate positions in env frame.
            gate_env_rot: (N, num_gates, 4) gate rotations in env frame.
            drone_pos:    (N, 1, 3) drone positions.
            drone_rot:    (N, 1, 4) drone rotations.

        Returns:
            next_gate_pos:        (N, 1, 3) selected gate position in env frame.
            next_gate_rot:        (N, 1, 4) selected gate rotation in env frame.
            next_gate_rpos_world: (N, 1, 3) gate position relative to drone, in world frame.
            next_gate_rpos_local: (N, 1, 3) gate position relative to drone, in drone-local frame.
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)

        next_gate_pos = gate_env_pos[batch_indices, gate_indices]  # (N, 3)
        next_gate_rot = gate_env_rot[batch_indices, gate_indices]  # (N, 4)

        # Expand to match agent dimension for broadcasting
        next_gate_pos = next_gate_pos.unsqueeze(1)  # (N, 1, 3)
        next_gate_rot = next_gate_rot.unsqueeze(1)  # (N, 1, 4)

        # Relative position in world frame
        next_gate_rpos_world = next_gate_pos - drone_pos  # (N, 1, 3)

        # Relative position in drone-local frame
        drone_rot_flat = drone_rot.squeeze(1)                          # (N, 4)
        next_gate_rpos_world_flat = next_gate_rpos_world.squeeze(1)    # (N, 3)
        next_gate_rpos_local_flat = quat_rotate_inverse(drone_rot_flat, next_gate_rpos_world_flat)  # (N, 3)
        next_gate_rpos_local = next_gate_rpos_local_flat.unsqueeze(1)  # (N, 1, 3)

        return next_gate_pos, next_gate_rot, next_gate_rpos_world, next_gate_rpos_local

    def get_next_to_next_gate_position(self, next_to_next_gate_indices, gate_env_pos, gate_env_rot, next_gate_indices):
        """Return the position of the next-to-next gate expressed in the next gate's local frame.

        Args:
            next_to_next_gate_indices: (N,) index of the gate after the immediate next gate (clamped at last gate).
            gate_env_pos:              (N, num_gates, 3) gate positions in env frame.
            gate_env_rot:              (N, num_gates, 4) gate rotations in env frame.
            next_gate_indices:         (N,) index of the immediate next gate per environment.

        Returns:
            (N, 3) position of the next-to-next gate relative to the next gate, in the next gate's local frame.
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)

        # Positions/orientations of the immediate next gate in env frame
        next_gate_pos_env = gate_env_pos[batch_indices, next_gate_indices]      # (N, 3)
        next_gate_rot_env = gate_env_rot[batch_indices, next_gate_indices]      # (N, 4)

        # Position of the next-to-next gate in env frame
        n2n_gate_pos_env = gate_env_pos[batch_indices, next_to_next_gate_indices]  # (N, 3)

        # Relative position in env frame
        rpos_env = n2n_gate_pos_env - next_gate_pos_env  # (N, 3)

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
        gate_centers = self._get_all_gate_centers(gate_env_pos, gate_env_rot)

        batch_indices = torch.arange(self.num_envs, device=self.device)
        target_gate_indices = self.gate_indices
        drone_pos_flat = drone_pos.squeeze(1)
        drone_rot_flat = drone_rot.squeeze(1)

        current_gate_center = gate_centers[batch_indices, target_gate_indices]
        current_gate_rot = gate_env_rot[batch_indices, target_gate_indices]
        current_gate_x, current_gate_y = self._gate_axes(current_gate_rot)

        current_rpos_world = current_gate_center - drone_pos_flat
        current_rpos_local = quat_rotate_inverse(drone_rot_flat, current_rpos_world)
        current_gate_frame_pos = quat_rotate_inverse(
            current_gate_rot,
            drone_pos_flat - current_gate_center,
        )
        self.current_gate_frame_pos = current_gate_frame_pos

        payload_features = []
        if self.payload_enabled:
            payload_pos = self.get_env_poses(self.payload.get_world_poses())[0]
            payload_vel = self.payload.get_velocities()
            payload_rpos_world = payload_pos - drone_pos_flat
            payload_rpos_local = quat_rotate_inverse(drone_rot_flat, payload_rpos_world)
            payload_gate_rpos_local = quat_rotate_inverse(
                drone_rot_flat,
                current_gate_center - payload_pos,
            )
            payload_gate_frame_pos = quat_rotate_inverse(
                current_gate_rot,
                payload_pos - current_gate_center,
            )
            tether_norm = payload_rpos_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            tether_dir = payload_rpos_world / tether_norm
            swing_sin = (
                payload_rpos_world[..., :2].norm(dim=-1, keepdim=True)
                / max(self.payload_bar_length, 1e-6)
            ).clamp(0.0, 2.0)
            pending_flag = self.gate_passed.float().unsqueeze(-1)
            payload_features = [
                payload_rpos_local,
                payload_vel,
                payload_gate_frame_pos,
                payload_gate_rpos_local,
                tether_dir,
                torch.cat([swing_sin, pending_flag], dim=-1),
            ]

        offsets = torch.arange(1, self.lookahead_gates + 1, device=self.device)
        lookahead_indices = torch.clamp(
            target_gate_indices.unsqueeze(-1) + offsets.unsqueeze(0),
            max=self.num_gates - 1,
        )
        lookahead_batch = batch_indices.unsqueeze(-1).expand(-1, self.lookahead_gates)
        lookahead_centers = gate_centers[lookahead_batch, lookahead_indices]
        lookahead_rots = gate_env_rot[lookahead_batch, lookahead_indices]
        lookahead_rpos_world = lookahead_centers - drone_pos_flat.unsqueeze(1)
        lookahead_rpos_local = quat_rotate_inverse(
            drone_rot_flat.unsqueeze(1).expand(-1, self.lookahead_gates, -1).reshape(-1, 4),
            lookahead_rpos_world.reshape(-1, 3),
        ).reshape(self.num_envs, self.lookahead_gates, 3)
        lookahead_x, lookahead_y = self._gate_axes(lookahead_rots)
        lookahead_features = torch.cat(
            [lookahead_rpos_local, lookahead_x, lookahead_y],
            dim=-1,
        ).flatten(1)

        waypoint_indices = torch.clamp(
            target_gate_indices.unsqueeze(-1)
            + torch.arange(0, self.lookahead_gates + 1, device=self.device).unsqueeze(0),
            max=self.num_gates - 1,
        )
        waypoint_batch = batch_indices.unsqueeze(-1).expand(-1, self.lookahead_gates + 1)
        waypoint_centers = gate_centers[waypoint_batch, waypoint_indices]
        waypoint_rots = gate_env_rot[waypoint_batch, waypoint_indices]
        waypoint_x, _ = self._gate_axes(waypoint_rots)
        pre_gate = waypoint_centers - waypoint_x * self.waypoint_offset
        post_gate = waypoint_centers + waypoint_x * self.waypoint_offset
        waypoints_world = torch.stack([pre_gate, waypoint_centers, post_gate], dim=2)
        waypoint_rpos_world = waypoints_world.reshape(self.num_envs, -1, 3) - drone_pos_flat.unsqueeze(1)
        waypoint_rpos_local = quat_rotate_inverse(
            drone_rot_flat.unsqueeze(1)
            .expand(-1, waypoint_rpos_world.shape[1], -1)
            .reshape(-1, 4),
            waypoint_rpos_world.reshape(-1, 3),
        ).reshape(self.num_envs, -1)

        gate_progress = self.gate_indices.float() / max(self.num_gates - 1, 1)
        gate_progress = torch.where(self.track_completed, torch.ones_like(gate_progress), gate_progress)
        remaining_progress = 1.0 - gate_progress
        progress_features = torch.stack([gate_progress, remaining_progress], dim=-1)

        obs = torch.cat(
            [
                self.drone_state.squeeze(1),
                current_rpos_local,
                current_gate_frame_pos,
                current_gate_x,
                current_gate_y,
                lookahead_features,
                waypoint_rpos_local,
                progress_features,
                *payload_features,
                self.last_action.squeeze(1),
            ],
            dim=-1,
        ).unsqueeze(1)

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

    def _get_all_gate_centers(self, gate_pos, gate_rot):
        """Compute centered gate positions for tensors shaped (N, G, 3)/(N, G, 4)."""
        offset_local = torch.tensor([0.0, 0.0, self.gate_height / 2.0], device=self.device)
        flat_pos = gate_pos.reshape(-1, 3)
        flat_rot = gate_rot.reshape(-1, 4)
        flat_offset = offset_local.unsqueeze(0).expand(flat_pos.shape[0], -1)
        flat_centers = flat_pos + quat_rotate(flat_rot, flat_offset)
        return flat_centers.reshape_as(gate_pos)

    def _gate_axes(self, gate_rot):
        flat_rot = gate_rot.reshape(-1, 4)
        e_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(flat_rot.shape[0], -1)
        e_y = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(flat_rot.shape[0], -1)
        gate_x = quat_rotate(flat_rot, e_x).reshape(*gate_rot.shape[:-1], 3)
        gate_y = quat_rotate(flat_rot, e_y).reshape(*gate_rot.shape[:-1], 3)
        return gate_x, gate_y

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
            gate_index_changed:    (N,) bool — True for envs whose target gate advanced.
            new_gate_center:       (N, 3) — centre of the (possibly new) target gate.
        """
        prev_in_gate = quat_rotate_inverse(
            current_gate_rot,
            self.prev_drone_pos - current_gate_center,
        )
        curr_in_gate = quat_rotate_inverse(
            current_gate_rot,
            drone_pos_flat - current_gate_center,
        )
        crossed_target, target_inside, crossing_in_gate, gates_passed_successfully = (
            detect_gate_crossings_from_frames(
                prev_in_gate,
                curr_in_gate,
                self.gate_width,
                self.gate_height,
            )
        )
        target_missed = crossed_target & (~target_inside)

        gate_centers = self._get_all_gate_centers(gate_env_pos, gate_env_rot)
        prev_to_all = self.prev_drone_pos.unsqueeze(1) - gate_centers
        curr_to_all = drone_pos_flat.unsqueeze(1) - gate_centers
        prev_all_gate = quat_rotate_inverse(
            gate_env_rot.reshape(-1, 4),
            prev_to_all.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_gates, 3)
        curr_all_gate = quat_rotate_inverse(
            gate_env_rot.reshape(-1, 4),
            curr_to_all.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_gates, 3)
        _, _, _, valid_all_gate_pass = detect_gate_crossings_from_frames(
            prev_all_gate,
            curr_all_gate,
            self.gate_width,
            self.gate_height,
        )

        same_physical_gate = (
            torch.norm(gate_centers - current_gate_center.unsqueeze(1), dim=-1) < 1e-3
        ) & (
            torch.abs((gate_env_rot * current_gate_rot.unsqueeze(1)).sum(-1)) > 0.999
        )
        target_mask = torch.zeros_like(valid_all_gate_pass)
        target_mask[batch_indices, self.gate_indices] = True
        wrong_gate_passed = (
            valid_all_gate_pass
            & (~target_mask)
            & (~same_physical_gate)
        ).any(dim=-1)
        
        gate_passed_this_step = gates_passed_successfully & (~self.gate_passed)
        self.gate_passed[gate_passed_this_step] = True

        old_gate_indices = self.gate_indices.clone()
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
        self.prev_drone_pos = drone_pos_flat.clone()

        return (
            gate_passed_this_step,
            gate_index_changed,
            new_gate_center,
            curr_in_gate,
            crossing_in_gate,
            target_missed,
            wrong_gate_passed,
        )

    def _detect_gate_crossings(self, drone_pos_flat, payload_pos_flat, current_gate_center,
                                current_gate_rot, gate_env_pos, gate_env_rot, batch_indices):
        """Payload-aware ordered gate crossing.

        This override keeps the original drone-only behavior when payloads are
        disabled. With a payload, the drone must enter the target aperture first
        and the payload must then cross the same aperture before the gate index
        advances.
        """
        prev_in_gate = quat_rotate_inverse(
            current_gate_rot,
            self.prev_drone_pos - current_gate_center,
        )
        curr_in_gate = quat_rotate_inverse(
            current_gate_rot,
            drone_pos_flat - current_gate_center,
        )
        prev_payload_in_gate = quat_rotate_inverse(
            current_gate_rot,
            self.prev_payload_pos - current_gate_center,
        )
        curr_payload_in_gate = quat_rotate_inverse(
            current_gate_rot,
            payload_pos_flat - current_gate_center,
        )

        (
            drone_crossed,
            drone_inside,
            crossing_in_gate,
            drone_valid_target,
        ) = detect_gate_crossings_from_frames(
            prev_in_gate,
            curr_in_gate,
            self.gate_width,
            self.gate_height,
        )
        target_missed = drone_crossed & (~drone_inside)
        payload_missed = torch.zeros_like(target_missed)
        payload_first = torch.zeros_like(target_missed)

        # Approach confirmed when drone is behind the gate (x < -waypoint_offset)
        approach_zone = curr_in_gate[..., 0] < -self.waypoint_offset
        self.gate_approach_confirmed = self.gate_approach_confirmed | approach_zone

        if self.payload_enabled:
            two_body = detect_two_body_gate_completion(
                prev_in_gate,
                curr_in_gate,
                prev_payload_in_gate,
                curr_payload_in_gate,
                self.gate_width,
                self.gate_height,
                self.gate_passed,
            )
            drone_entry_this_step = two_body["drone_valid"] & (~self.gate_passed) & self.gate_approach_confirmed
            gate_passed_this_step = two_body["gate_completed"] & self.gate_approach_confirmed
            target_missed = two_body["aperture_missed"]
            payload_missed = two_body["payload_crossed"] & (~two_body["payload_inside"])
            payload_first = two_body["payload_first"]
            self.gate_passed[two_body["drone_entered"] & self.gate_approach_confirmed] = True
        else:
            gate_passed_this_step = drone_valid_target & (~self.gate_passed) & self.gate_approach_confirmed
            drone_entry_this_step = gate_passed_this_step
            self.gate_passed[gate_passed_this_step] = True

        gate_centers = self._get_all_gate_centers(gate_env_pos, gate_env_rot)
        prev_to_all = self.prev_drone_pos.unsqueeze(1) - gate_centers
        curr_to_all = drone_pos_flat.unsqueeze(1) - gate_centers
        prev_all_gate = quat_rotate_inverse(
            gate_env_rot.reshape(-1, 4),
            prev_to_all.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_gates, 3)
        curr_all_gate = quat_rotate_inverse(
            gate_env_rot.reshape(-1, 4),
            curr_to_all.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_gates, 3)
        _, _, _, valid_all_gate_pass = detect_gate_crossings_from_frames(
            prev_all_gate,
            curr_all_gate,
            self.gate_width,
            self.gate_height,
        )
        if self.payload_enabled:
            prev_payload_to_all = self.prev_payload_pos.unsqueeze(1) - gate_centers
            curr_payload_to_all = payload_pos_flat.unsqueeze(1) - gate_centers
            prev_payload_all_gate = quat_rotate_inverse(
                gate_env_rot.reshape(-1, 4),
                prev_payload_to_all.reshape(-1, 3),
            ).reshape(self.num_envs, self.num_gates, 3)
            curr_payload_all_gate = quat_rotate_inverse(
                gate_env_rot.reshape(-1, 4),
                curr_payload_to_all.reshape(-1, 3),
            ).reshape(self.num_envs, self.num_gates, 3)
            _, _, _, payload_valid_all_gate_pass = detect_gate_crossings_from_frames(
                prev_payload_all_gate,
                curr_payload_all_gate,
                self.gate_width,
                self.gate_height,
            )
            valid_all_gate_pass = valid_all_gate_pass | payload_valid_all_gate_pass

        same_physical_gate = (
            torch.norm(gate_centers - current_gate_center.unsqueeze(1), dim=-1) < 1e-3
        ) & (
            torch.abs((gate_env_rot * current_gate_rot.unsqueeze(1)).sum(-1)) > 0.999
        )
        target_mask = torch.zeros_like(valid_all_gate_pass)
        target_mask[batch_indices, self.gate_indices] = True
        wrong_gate_passed = (
            valid_all_gate_pass
            & (~target_mask)
            & (~same_physical_gate)
        ).any(dim=-1) | payload_first

        completed_gate_indices = self.gate_indices[gate_passed_this_step].clone()
        if gate_passed_this_step.any():
            self.gate_completion_flags[
                batch_indices[gate_passed_this_step],
                completed_gate_indices,
            ] = True

        old_gate_indices = self.gate_indices.clone()
        last_gate_passed = gate_passed_this_step & (self.gate_indices + 1 >= self.num_gates)
        self.track_completed[last_gate_passed] = True
        self.gate_indices[gate_passed_this_step] = torch.clamp(
            self.gate_indices[gate_passed_this_step] + 1,
            max=self.num_gates - 1,
        )
        gate_index_changed = self.gate_indices != old_gate_indices
        self.gate_passed[gate_index_changed] = False
        # Reset approach flag for the new gate
        self.gate_approach_confirmed[gate_index_changed] = False

        new_gate_pos = gate_env_pos[batch_indices, self.gate_indices]
        new_gate_rot = gate_env_rot[batch_indices, self.gate_indices]
        new_gate_center = self._get_gate_center(new_gate_pos, new_gate_rot)
        new_in_gate = quat_rotate_inverse(new_gate_rot, drone_pos_flat - new_gate_center)
        new_payload_in_gate = quat_rotate_inverse(new_gate_rot, payload_pos_flat - new_gate_center)

        self.prev_drone_in_gate_frame = torch.where(
            gate_index_changed.unsqueeze(-1), new_in_gate, curr_in_gate,
        )
        self.prev_payload_in_gate_frame = torch.where(
            gate_index_changed.unsqueeze(-1), new_payload_in_gate, curr_payload_in_gate,
        )
        self.prev_drone_pos = drone_pos_flat.clone()
        self.prev_payload_pos = payload_pos_flat.clone()

        return (
            gate_passed_this_step,
            drone_entry_this_step,
            gate_index_changed,
            new_gate_center,
            curr_in_gate,
            curr_payload_in_gate,
            crossing_in_gate,
            target_missed,
            payload_missed,
            wrong_gate_passed,
        )


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
        if self.payload_enabled:
            payload_pos_flat = self.get_env_poses(self.payload.get_world_poses())[0]
            payload_vels = self.payload.get_velocities()
        else:
            payload_pos_flat = drone_pos_flat
            payload_vels = torch.zeros(self.num_envs, 6, device=self.device)
        drone_distance_to_gate = torch.norm(drone_pos_flat - current_gate_center, dim=-1)
        payload_distance_to_gate = torch.norm(payload_pos_flat - current_gate_center, dim=-1)
        distance_to_gate = (
            0.4 * drone_distance_to_gate + 0.6 * payload_distance_to_gate
            if self.payload_enabled
            else drone_distance_to_gate
        )

        # --- gate crossing detection ---
        # You either _deteect_gate_crossings or _detect_gate_crossings_via_segments
        # This function call updates the gate indexes
        (
            gate_passed_this_step,
            drone_entry_this_step,
            gate_index_changed,
            new_gate_center,
            current_gate_frame_pos,
            payload_gate_frame_pos,
            crossing_in_gate,
            target_missed,
            payload_missed,
            wrong_gate_passed,
        ) = self._detect_gate_crossings(
            drone_pos_flat,
            payload_pos_flat,
            current_gate_center,
            current_gate_rot,
            gate_env_pos,
            gate_env_rot,
            batch_indices,
        )

        # -----------------------------------------------------------------------
        # Reward components:
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
        #   self.drone.vel        (N, 1, 6) [lin_vel(3) | ang_vel(3)] in body frame
        #   self.gate_indices     (N,)      index of current target gate (0…num_gates-1)
        #   self.track_completed  (N,)      bool – True when the full lap is done
        #
        # Useful helpers (already imported at top of file):
        #   quat_axis(q, axis)          extract body axis vector (0=x, 1=y, 2=z)
        #   quat_rotate(q, v)           rotate vector v by quaternion q
        #   quat_rotate_inverse(q, v)   rotate vector v by inverse of quaternion q
        #
        # Reward scalars are loaded in __init__ and
        # configured in cfg/task/DroneRace.yaml.
        # -----------------------------------------------------------------------
        # Ordered-gate racing reward.

        prev_distance_to_gate = (
            0.4 * self.prev_distance_to_gate + 0.6 * self.prev_payload_distance_to_gate
            if self.payload_enabled
            else self.prev_distance_to_gate
        )
        progress_delta = (prev_distance_to_gate - distance_to_gate).clamp(
            -self.progress_clamp,
            self.progress_clamp,
        )
        reward_progress = self.reward_progress_scale * progress_delta

        lateral_error = current_gate_frame_pos[:, 1] / max(self.gate_width * 0.5, 1e-6)
        vertical_error = current_gate_frame_pos[:, 2] / max(self.gate_height * 0.5, 1e-6)
        aperture_error_sq = lateral_error.square() + vertical_error.square()
        plane_proximity = torch.exp(
            -torch.abs(current_gate_frame_pos[:, 0]) / max(self.waypoint_offset, 1e-6)
        )
        reward_centerline = (
            self.reward_centerline_scale
            * torch.exp(-aperture_error_sq)
            * plane_proximity
        )
        if self.payload_enabled:
            payload_lateral_error = payload_gate_frame_pos[:, 1] / max(self.gate_width * 0.5, 1e-6)
            payload_vertical_error = payload_gate_frame_pos[:, 2] / max(self.gate_height * 0.5, 1e-6)
            payload_aperture_error_sq = payload_lateral_error.square() + payload_vertical_error.square()
            payload_plane_proximity = torch.exp(
                -torch.abs(payload_gate_frame_pos[:, 0]) / max(self.waypoint_offset, 1e-6)
            )
            reward_centerline = reward_centerline + (
                self.reward_payload_centerline_scale
                * torch.exp(-payload_aperture_error_sq)
                * payload_plane_proximity
            )

        current_gate_x, _ = self._gate_axes(current_gate_rot)
        lin_vel_world = self.drone.vel_w[..., :3].squeeze(1)
        payload_lin_vel_world = payload_vels[..., :3]
        speed = torch.norm(lin_vel_world, dim=-1)
        payload_speed = torch.norm(payload_lin_vel_world, dim=-1)
        drone_forward_speed = (lin_vel_world * current_gate_x).sum(-1)
        payload_forward_speed = (payload_lin_vel_world * current_gate_x).sum(-1)
        forward_speed = (
            torch.minimum(drone_forward_speed, payload_forward_speed)
            if self.payload_enabled
            else drone_forward_speed
        )
        reward_speed = self.reward_speed_scale * forward_speed.clamp(0.0, self.speed_reward_cap)
        reward_effort = -self.reward_effort_weight * self.effort.squeeze(1).abs().mean(-1)
        reward_action_smoothness = (
            -self.reward_action_smoothness_weight
            * self.action_smoothness.squeeze(-1)
        )
        tether = payload_pos_flat - drone_pos_flat
        tether_len = tether.norm(dim=-1).clamp_min(1e-6)
        swing_angle = torch.acos((-tether[:, 2] / tether_len).clamp(-1.0, 1.0))
        payload_lateral_speed = torch.norm(payload_lin_vel_world[..., :2], dim=-1)
        reward_payload_swing = (
            -self.reward_payload_swing_weight * swing_angle.square()
            -self.reward_payload_lateral_velocity_weight * payload_lateral_speed
            if self.payload_enabled
            else torch.zeros_like(speed)
        )

        completed_this_step = gate_passed_this_step & self.track_completed
        reward = (
            reward_progress
            + reward_centerline
            + reward_speed
            + reward_payload_swing
            + reward_effort
            + reward_action_smoothness
            - self.reward_time_penalty
            + self.reward_drone_gate_entry * drone_entry_this_step.float()
            + self.reward_gate_passage * gate_passed_this_step.float()
            + self.reward_completion * completed_this_step.float()
        )

        # -----------------------------------------------------------------------
        # Termination components:
        # Collision, target miss, wrong gate, out-of-bounds, NaN, and stall.
        # -----------------------------------------------------------------------
        # Ordered-gate racing termination.

        contact_forces = self.drone.base_link.get_net_contact_forces()
        contact_magnitudes = torch.norm(contact_forces, dim=-1)
        if contact_magnitudes.ndim > 1:
            crashed_collision = contact_magnitudes.gt(self.contact_force_threshold).any(-1)
        else:
            crashed_collision = contact_magnitudes.gt(self.contact_force_threshold)
        if self.payload_enabled:
            payload_forces = self.payload.get_net_contact_forces()
            payload_contact_magnitudes = torch.norm(payload_forces, dim=-1)
            if payload_contact_magnitudes.ndim > 1:
                payload_collision = payload_contact_magnitudes.gt(self.contact_force_threshold).any(-1)
            else:
                payload_collision = payload_contact_magnitudes.gt(self.contact_force_threshold)
        else:
            payload_collision = torch.zeros_like(crashed_collision)

        gate_centers = self._get_all_gate_centers(gate_env_pos, gate_env_rot)
        min_xy = gate_centers[..., :2].amin(dim=1) - self.bounds_margin_xy
        max_xy = gate_centers[..., :2].amax(dim=1) + self.bounds_margin_xy
        crashed_bounds = (
            (drone_pos_flat[:, 0] < min_xy[:, 0])
            | (drone_pos_flat[:, 0] > max_xy[:, 0])
            | (drone_pos_flat[:, 1] < min_xy[:, 1])
            | (drone_pos_flat[:, 1] > max_xy[:, 1])
            | (drone_pos_flat[:, 2] < self.bounds_min_z)
            | (drone_pos_flat[:, 2] > self.bounds_max_z)
            | (payload_pos_flat[:, 0] < min_xy[:, 0])
            | (payload_pos_flat[:, 0] > max_xy[:, 0])
            | (payload_pos_flat[:, 1] < min_xy[:, 1])
            | (payload_pos_flat[:, 1] > max_xy[:, 1])
            | (payload_pos_flat[:, 2] < self.bounds_min_z)
            | (payload_pos_flat[:, 2] > self.bounds_max_z)
        )
        wrong_gate = target_missed | wrong_gate_passed
        hasnan = torch.isnan(self.drone_state).any(-1).squeeze(-1)
        stall_speed_value = torch.minimum(speed, payload_speed) if self.payload_enabled else speed
        stalled = (
            (stall_speed_value < self.stall_speed)
            & (self.progress_buf > self.stall_grace_steps)
            & (distance_to_gate > self.stall_distance)
        )

        crash_not_wrong = crashed_collision | payload_collision | crashed_bounds | hasnan
        crashed = crash_not_wrong | wrong_gate | stalled
        reward = reward - self.reward_crash * crash_not_wrong.float()
        reward = reward - self.reward_wrong_gate * wrong_gate.float()
        reward = reward - self.reward_payload_miss * payload_missed.float()
        reward = reward - self.reward_stall * stalled.float()

        next_distance_to_gate = torch.norm(drone_pos_flat - new_gate_center, dim=-1)
        next_payload_distance_to_gate = torch.norm(payload_pos_flat - new_gate_center, dim=-1)
        self.prev_distance_to_gate = torch.where(
            self.track_completed,
            torch.zeros_like(next_distance_to_gate),
            next_distance_to_gate,
        )
        self.prev_payload_distance_to_gate = torch.where(
            self.track_completed,
            torch.zeros_like(next_payload_distance_to_gate),
            next_payload_distance_to_gate,
        )

        self.speed_sum += speed
        self.max_speed_buf = torch.maximum(self.max_speed_buf, speed)
        self.payload_speed_sum += payload_speed
        self.payload_max_speed_buf = torch.maximum(self.payload_max_speed_buf, payload_speed)
        self.payload_swing_sum += swing_angle
        self.payload_max_swing_buf = torch.maximum(self.payload_max_swing_buf, swing_angle)

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        completed_task = self.track_completed
        done = truncated | completed_task.unsqueeze(-1) | crashed.unsqueeze(-1)

        # --- stats ---
        policy_dt = self.dt * self.substeps
        episode_steps = self.progress_buf.clamp_min(1.0)
        gates_passed = (
            self.gate_indices
            - self.start_gate_indices
            + self.track_completed.long()
        ).clamp_min(0)
        self.stats["truncated"].add_(truncated.float())
        self.stats["collision"].add_(crashed_collision.float().unsqueeze(-1))
        self.stats["wrong_gate"].add_(wrong_gate.float().unsqueeze(-1))
        self.stats["crashed_bounds"].add_(crashed_bounds.float().unsqueeze(-1))
        self.stats["stalled"].add_(stalled.float().unsqueeze(-1))
        self.stats["payload_collision"].add_(payload_collision.float().unsqueeze(-1))
        self.stats["payload_miss"].add_(payload_missed.float().unsqueeze(-1))
        self.stats["drone_gate_entry"].add_(drone_entry_this_step.float().unsqueeze(-1))
        self.stats["success"].bitwise_or_(completed_task.unsqueeze(-1))
        self.stats["return"].add_(reward.unsqueeze(-1))
        self.stats["progress_reward"].add_(reward_progress.unsqueeze(-1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["gates_passed"][:] = gates_passed.float().unsqueeze(1)
        self.stats["lap_time_sec"][:] = (self.progress_buf * policy_dt).unsqueeze(1)
        self.stats["mean_speed"][:] = (self.speed_sum / episode_steps).unsqueeze(1)
        self.stats["max_speed"][:] = self.max_speed_buf.unsqueeze(1)
        self.stats["payload_mean_speed"][:] = (self.payload_speed_sum / episode_steps).unsqueeze(1)
        self.stats["payload_max_speed"][:] = self.payload_max_speed_buf.unsqueeze(1)
        self.stats["payload_mean_swing_angle"][:] = (self.payload_swing_sum / episode_steps).unsqueeze(1)
        self.stats["payload_max_swing_angle"][:] = self.payload_max_swing_buf.unsqueeze(1)
        self.stats["start_gate_index"][:] = self.start_gate_indices.float().unsqueeze(1)
        self.stats["full_course_start"][:] = (self.start_gate_indices == 0).float().unsqueeze(1)
        for gate_idx in range(self.num_gates):
            self.stats[f"gate_{gate_idx + 1:02d}_passed"][:] = (
                self.gate_completion_flags[:, gate_idx].float().unsqueeze(1)
            )

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
