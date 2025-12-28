"""
QuadrupedCorridor: A training environment with narrow passageways.
Agents start on one side (blocked off from each other) and must navigate
through narrow passages to reach goals on the other side (mixed up to force collision avoidance).
"""
import functools as ft
import pathlib
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from typing import NamedTuple, Tuple, Optional

from .base import MultiAgentEnv, RolloutResult
from ..trainer.data import Rollout
from ..utils.typing import State, Array, AgentState, Action, Reward, Cost, Done, Info, Pos2d
from ..utils.graph import GraphsTuple, EdgeBlock, GetGraph
from .obstacle import Obstacle, Rectangle
from ..utils.utils import merge01, jax_vmap, tree_index, tree_concat_at_front
from .plot import render_video as render_quadruped_video
from .utils import get_node_goal_rng, inside_obstacles, get_lidar


class QuadrupedCorridor(MultiAgentEnv):

    AGENT = 0
    GOAL = 1
    OBS = 2
    ENV_SIZE = 48.0

    class EnvState(NamedTuple):
        agent: State
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.55 / ENV_SIZE,
        "comm_radius": 1.0 / 4.0,
        "n_rays": 32,
        "top_k_rays": 8,
        "action_lower": np.array([-1.50 / ENV_SIZE, -1.50 / ENV_SIZE, -2.0]),
        "action_upper": np.array([1.50 / ENV_SIZE, 1.50 / ENV_SIZE, 2.0]),
        "default_area_size": 1.0,
        # Corridor-specific parameters
        "passageway_width": 0.05,  # Width of narrow passages (normalized)
        "num_passages": 3,  # Number of vertical passageways
        "start_y": 0.02,  # Y position where agents start (bottom)
        "goal_y": 0.50,  # Y position where goals are placed (top, past tunnels)
        "wall_thickness": 0.02,  # Thickness of walls between passages
        "start_separation": 0.15,  # Horizontal spacing between agent start positions
        "tunnel_end_y": 0.40,  # Y position where tunnels end (goals are past this)
        "goal_area_y_min": 0.45,  # Minimum Y for goal placement in open space
        "goal_area_y_max": 0.95,  # Maximum Y for goal placement in open space
        "random_obstacle_prob": 0.3,  # Probability of adding a random obstacle in a passageway
        "random_obstacle_size_range": [0.02, 0.08],  # Size range for random obstacles (normalized)
        "randomize_orientation": True,  # Whether to randomize the direction of tunnels
        "fixed_orientation": None,  # Fixed orientation angle in radians (None = randomize)
        "randomize_position": True,  # Whether to randomize where the corridor system is placed
        "position_offset_range": 0.15,  # Maximum offset for corridor position (normalized, as fraction of area_size)
        "open_space_obstacle_count": 5,  # Number of random obstacles in open space (goal area)
        "open_space_obstacle_size_range": [0.03, 0.10],  # Size range for open space obstacles (normalized)
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = QuadrupedCorridor.PARAMS["default_area_size"] if area_size is None else area_size
        super(QuadrupedCorridor, self).__init__(num_agents, area_size, max_step, dt, params)
        self.trajs = []
        self.create_obstacles = jax_vmap(Rectangle.create)
        self.max_travel = self._params.get("max_travel", None)

    @property
    def state_dim(self) -> int:
        return 5  # x, y, yaw, vx, vy

    @property
    def node_dim(self) -> int:
        return self.state_dim + 3  # state plus indicator bits

    @property
    def edge_dim(self) -> int:
        return self.state_dim  # use full state difference

    @property
    def action_dim(self) -> int:
        return 3  # ax, ay, yawrate

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    def _get_rotation_and_offset(self, key: Array) -> Tuple[float, jnp.ndarray]:
        """Generate rotation angle and position offset for the corridor system."""
        area_size = self.area_size
        
        # Determine rotation angle
        randomize_orientation = self._params.get("randomize_orientation", True)
        fixed_orientation = self._params.get("fixed_orientation")
        
        if not randomize_orientation:
            rotation_angle = 0.0
        elif fixed_orientation is not None:
            rotation_angle = float(fixed_orientation)
        else:
            angle_key, key = jr.split(key, 2)
            rotation_angle = jr.uniform(angle_key, (), minval=0.0, maxval=2 * jnp.pi)
        
        # Determine position offset
        randomize_position = self._params.get("randomize_position", True)
        position_offset_range = self._params.get("position_offset_range", 0.15) * area_size
        
        if randomize_position:
            pos_key, key = jr.split(key, 2)
            position_offset = jr.uniform(
                pos_key, 
                (2,), 
                minval=-position_offset_range, 
                maxval=position_offset_range
            )
        else:
            position_offset = jnp.array([0.0, 0.0])
        
        return rotation_angle, position_offset

    def _rotate_point(self, point: jnp.ndarray, angle: float, center: jnp.ndarray) -> jnp.ndarray:
        """Rotate a point around a center by given angle."""
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        rel_point = point - center
        rotated_rel = jnp.array([
            cos_a * rel_point[0] - sin_a * rel_point[1],
            sin_a * rel_point[0] + cos_a * rel_point[1]
        ])
        return rotated_rel + center

    def _generate_corridor_obstacles(self, key: Array, rotation_angle: float, position_offset: jnp.ndarray, goal_positions_unrotated: Optional[jnp.ndarray] = None) -> Tuple[Rectangle, list]:
        """
        Generate obstacles that create narrow passageways at a given orientation and position.
        Creates walls that form num_passages corridors.
        Also adds random obstacles that can block some passageways.
        Returns: (obstacles, agent_start_xs)
        """
        area_size = self.area_size
        passageway_width = self._params["passageway_width"] * area_size
        num_passages = int(self._params["num_passages"])  # Ensure concrete value
        wall_thickness = self._params["wall_thickness"] * area_size
        start_y = self._params["start_y"] * area_size
        tunnel_end_y = self._params.get("tunnel_end_y", 0.40) * area_size
        
        # Base center of rotation (center of area) + position offset
        base_center = jnp.array([area_size / 2, area_size / 2])
        center = base_center + position_offset
        
        # Calculate total width needed
        total_passage_width = num_passages * passageway_width
        total_wall_width = (num_passages - 1) * wall_thickness
        total_width = total_passage_width + total_wall_width
        
        # Center the corridor system
        x_start = (area_size - total_width) / 2
        
        obstacles_list = []
        
        # Create vertical walls between passages (only up to tunnel_end_y)
        for i in range(num_passages - 1):
            wall_x = x_start + (i + 1) * passageway_width + i * wall_thickness + wall_thickness / 2
            wall_center = jnp.array([wall_x, (start_y + tunnel_end_y) / 2])
            wall_width = wall_thickness
            wall_height = tunnel_end_y - start_y + 0.05 * area_size  # Extend slightly beyond
            obstacles_list.append((wall_center, wall_width, wall_height, 0.0))
        
        # Create horizontal blocking walls at start (to separate agents initially)
        # These create individual starting chambers for each agent
        agent_separation = self._params["start_separation"] * area_size
        num_agents = self.num_agents
        
        # Calculate agent start positions with proper clearance from walls
        car_radius = self._params["car_radius"] * area_size
        agent_start_xs = []
        for i in range(num_agents):
            # Distribute agents across the passages
            passage_idx = i % num_passages
            passage_center_x = x_start + passage_idx * (passageway_width + wall_thickness) + passageway_width / 2
            
            # Calculate safe offset range to avoid walls
            # Left wall of passage is at: passage_center_x - passageway_width / 2
            # Right wall of passage is at: passage_center_x + passageway_width / 2
            # Need to keep agent at least car_radius away from each wall
            safe_offset_max = passageway_width / 2 - car_radius * 1.5  # Clearance from right wall
            safe_offset_min = -passageway_width / 2 + car_radius * 1.5  # Clearance from left wall
            
            # Add small random offset within safe range
            offset_key, key = jr.split(key, 2)
            offset = jr.uniform(offset_key, (), minval=safe_offset_min, maxval=safe_offset_max)
            agent_start_xs.append(passage_center_x + offset)
        
        # Create blocking walls at start (small horizontal walls to separate agents)
        blocking_wall_height = 0.05 * area_size
        for i in range(num_agents - 1):
            if i < num_passages - 1:
                # Place wall between adjacent agents in same passage
                wall_x = (agent_start_xs[i] + agent_start_xs[i + 1]) / 2
                wall_center = jnp.array([wall_x, start_y + blocking_wall_height / 2])
                wall_width = wall_thickness
                wall_height = blocking_wall_height
                obstacles_list.append((wall_center, wall_width, wall_height, 0.0))
        
        # Block off the entrance side of tunnels (prevent agents from going backwards)
        # Place wall far enough below start_y to allow safe agent placement
        car_radius = self._params["car_radius"] * area_size
        entrance_wall_y = start_y - car_radius * 1.5 - 0.02 * area_size  # Well below start_y to allow safe agent placement
        entrance_wall_height = wall_thickness
        entrance_wall_width = total_width + 2 * wall_thickness  # Span entire width of corridor system
        
        # Entrance wall center
        entrance_wall_center = jnp.array([area_size / 2, entrance_wall_y])
        obstacles_list.append((
            entrance_wall_center,
            entrance_wall_width,
            entrance_wall_height,
            0.0
        ))
        
        # Create side walls to contain the corridor system (only up to tunnel_end_y)
        left_wall_x = x_start - wall_thickness / 2
        right_wall_x = x_start + total_width + wall_thickness / 2
        wall_height = tunnel_end_y - start_y + 0.05 * area_size
        
        # Left wall
        obstacles_list.append((
            jnp.array([left_wall_x, (start_y + tunnel_end_y) / 2]),
            wall_thickness,
            wall_height,
            0.0
        ))
        
        # Right wall
        obstacles_list.append((
            jnp.array([right_wall_x, (start_y + tunnel_end_y) / 2]),
            wall_thickness,
            wall_height,
            0.0
        ))
        
        # Add random obstacles that can block some passageways
        random_obstacle_prob = self._params.get("random_obstacle_prob", 0.3)
        obstacle_size_range = self._params.get("random_obstacle_size_range", [0.02, 0.08])
        
        # Generate random obstacles for each passageway using JAX operations
        # We'll generate obstacles for all passages, but some will be placed outside the area
        # to effectively "not exist" when should_add_obstacle is False
        for passage_idx in range(num_passages):
            passage_center_x = x_start + passage_idx * (passageway_width + wall_thickness) + passageway_width / 2
            
            # Split keys for this passage
            prob_key, size_key, pos_key, x_offset_key, theta_key, key = jr.split(key, 6)
            
            # Decide if this passage gets a random obstacle (JAX boolean, not Python)
            should_add_obstacle = jr.uniform(prob_key, ()) < random_obstacle_prob
            
            # Random size within passageway constraints - small enough to allow agents to go around
            # Ensure at least car_radius * 2.5 clearance on at least one side for navigation
            car_radius = self._params["car_radius"] * area_size
            min_clearance = car_radius * 2.5  # Minimum clearance needed for agent to pass
            max_size_for_navigation = passageway_width - min_clearance  # Leave clearance for agent to pass
            max_size = min(obstacle_size_range[1] * area_size, max_size_for_navigation)
            max_size = jnp.maximum(max_size, obstacle_size_range[0] * area_size)  # At least minimum size
            obstacle_size = jr.uniform(
                size_key, 
                (), 
                minval=obstacle_size_range[0] * area_size,
                maxval=max_size
            )
            
            # Random position along the passageway (between start_y and tunnel_end_y)
            obstacle_y = jr.uniform(
                pos_key,
                (),
                minval=start_y + 0.1 * area_size,
                maxval=tunnel_end_y - 0.1 * area_size
            )
            
            # Random x position within passageway, but ensure at least one side has enough clearance
            # Left wall is at: passage_center_x - passageway_width / 2
            # Right wall is at: passage_center_x + passageway_width / 2
            obstacle_half_size = obstacle_size / 2
            left_wall_x = passage_center_x - passageway_width / 2
            right_wall_x = passage_center_x + passageway_width / 2
            
            # Calculate clearance on each side for a given obstacle_x
            # We want to ensure that for any valid obstacle_x, at least one side has >= min_clearance
            
            # Strategy: Constrain obstacle_x such that:
            # - If placed at left bound: right side has min_clearance
            # - If placed at right bound: left side has min_clearance
            # - Anywhere in between: at least one side has min_clearance
            
            # Left bound: obstacle_x such that right clearance = min_clearance
            #   right_clearance = right_wall_x - (obstacle_x + obstacle_half_size) = min_clearance
            #   obstacle_x = right_wall_x - obstacle_half_size - min_clearance
            left_bound = right_wall_x - obstacle_half_size - min_clearance
            
            # Right bound: obstacle_x such that left clearance = min_clearance
            #   left_clearance = (obstacle_x - obstacle_half_size) - left_wall_x = min_clearance
            #   obstacle_x = left_wall_x + obstacle_half_size + min_clearance
            right_bound = left_wall_x + obstacle_half_size + min_clearance
            
            # Also ensure obstacle doesn't go outside passageway bounds
            absolute_min = left_wall_x + obstacle_half_size
            absolute_max = right_wall_x - obstacle_half_size
            
            # Safe range: intersection of all constraints
            safe_x_min = jnp.maximum(absolute_min, right_bound)
            safe_x_max = jnp.minimum(absolute_max, left_bound)
            
            # Ensure valid range
            safe_x_min = jnp.minimum(safe_x_min, safe_x_max - 0.001 * area_size)
            
            obstacle_x = jr.uniform(
                x_offset_key,
                (),
                minval=safe_x_min,
                maxval=safe_x_max
            )
            
            # Random orientation
            obstacle_theta = jr.uniform(theta_key, (), minval=0.0, maxval=2 * jnp.pi)
            
            # Use JAX conditional: if should_add_obstacle is False, place obstacle far outside area
            # This effectively makes it "not exist" without using Python if
            obstacle_x = jnp.where(should_add_obstacle, obstacle_x, -10.0 * area_size)
            obstacle_y = jnp.where(should_add_obstacle, obstacle_y, -10.0 * area_size)
            obstacle_size = jnp.where(should_add_obstacle, obstacle_size, 0.001 * area_size)
            
            obstacles_list.append((
                jnp.array([obstacle_x, obstacle_y]),
                obstacle_size,
                obstacle_size,
                obstacle_theta
            ))
        
        # Add random obstacles in the open space (goal area)
        open_space_obstacle_count = int(self._params.get("open_space_obstacle_count", 3))
        open_space_size_range = self._params.get("open_space_obstacle_size_range", [0.03, 0.10])
        goal_area_y_min = self._params.get("goal_area_y_min", 0.45) * area_size
        goal_area_y_max = self._params.get("goal_area_y_max", 0.95) * area_size
        car_radius = self._params["car_radius"] * area_size
        
        # Generate open space obstacles, ensuring they don't intersect with goals
        # Use rejection sampling with JAX-compatible approach
        min_goal_clearance = car_radius * 2.5  # Minimum distance from goals
        
        def attempt_obstacle(attempt_key):
            """Attempt to generate one obstacle."""
            size_key, x_key, y_key, theta_key, next_key = jr.split(attempt_key, 5)
            
            # Random size
            obstacle_size = jr.uniform(
                size_key,
                (),
                minval=open_space_size_range[0] * area_size,
                maxval=open_space_size_range[1] * area_size
            )
            
            # Random position in open space (goal area)
            obstacle_x = jr.uniform(x_key, (), minval=0.1 * area_size, maxval=0.9 * area_size)
            obstacle_y = jr.uniform(y_key, (), minval=goal_area_y_min, maxval=goal_area_y_max)
            obstacle_pos = jnp.array([obstacle_x, obstacle_y])
            
            # Random orientation
            obstacle_theta = jr.uniform(theta_key, (), minval=0.0, maxval=2 * jnp.pi)
            
            # Check if this obstacle intersects with any goals
            # Create a temporary obstacle in unrotated coordinates to check collision
            # Use proper Rectangle collision detection with expanded size for clearance
            if goal_positions_unrotated is not None and goal_positions_unrotated.shape[0] > 0:
                # Create temporary obstacle for collision checking
                # Expand obstacle size to account for goal clearance (goals are points, so we need clearance around them)
                # The clearance should account for the fact that goals will be checked with radius later
                temp_obstacle_size = obstacle_size + min_goal_clearance * 2
                temp_obstacle = Rectangle.create(
                    obstacle_pos,
                    temp_obstacle_size,
                    temp_obstacle_size,
                    obstacle_theta
                )
                # Check if any goal is inside this expanded obstacle (with additional margin)
                # Use the obstacle's inside method directly for each goal (since it's a single obstacle)
                clearance_radius = car_radius * 0.5
                goal_intersections = jax.vmap(
                    lambda goal: temp_obstacle.inside(goal, r=clearance_radius)
                )(goal_positions_unrotated)
                has_intersection = jnp.any(goal_intersections)
            else:
                has_intersection = jnp.array(False)
            
            return obstacle_pos, obstacle_size, obstacle_theta, has_intersection, next_key
        
        # Vectorize open space obstacle generation: generate all obstacles at once
        if open_space_obstacle_count > 0:
            # Generate all obstacles in parallel using vmap
            max_attempts = 10  # Reduced from 20 for better performance
            all_obs_keys = jr.split(key, open_space_obstacle_count)
            
            def generate_one_obstacle(obs_key):
                # Try multiple times for this obstacle
                attempt_keys = jr.split(obs_key, max_attempts)
                results = jax.vmap(attempt_obstacle)(attempt_keys)
                
                # Find first valid (non-intersecting) obstacle
                valid_mask = ~results[3]  # has_intersection is False
                valid_indices = jnp.where(valid_mask, jnp.arange(max_attempts), max_attempts)
                first_valid_idx = jnp.argmin(valid_indices)
                
                # Use first valid obstacle, or last one if all intersect (should be rare)
                return (
                    results[0][first_valid_idx],
                    results[1][first_valid_idx],
                    results[2][first_valid_idx]
                )
            
            # Generate all obstacles in parallel
            all_obstacles = jax.vmap(generate_one_obstacle)(all_obs_keys)
            
            # Add to obstacles_list (still need Python loop for appending, but generation is vectorized)
            for i in range(open_space_obstacle_count):
                obstacles_list.append((
                    all_obstacles[0][i],
                    all_obstacles[1][i],
                    all_obstacles[1][i],  # height = width
                    all_obstacles[2][i]
                ))
            
            # Update key
            key = all_obs_keys[-1]
        
        # Convert to arrays for batch creation
        if len(obstacles_list) == 0:
            centers = jnp.zeros((0, 2), dtype=jnp.float32)
            widths = jnp.zeros((0,), dtype=jnp.float32)
            heights = jnp.zeros((0,), dtype=jnp.float32)
            thetas = jnp.zeros((0,), dtype=jnp.float32)
        else:
            centers = jnp.stack([obs[0] for obs in obstacles_list], axis=0)
            widths = jnp.array([obs[1] for obs in obstacles_list], dtype=jnp.float32)
            heights = jnp.array([obs[2] for obs in obstacles_list], dtype=jnp.float32)
            thetas = jnp.array([obs[3] for obs in obstacles_list], dtype=jnp.float32)
        
        # Rotate all obstacle positions by rotation_angle around center
        cos_a = jnp.cos(rotation_angle)
        sin_a = jnp.sin(rotation_angle)
        
        # Rotate centers
        rel_centers = centers - center[None, :]
        rotated_rel_centers = jnp.stack([
            cos_a * rel_centers[:, 0] - sin_a * rel_centers[:, 1],
            sin_a * rel_centers[:, 0] + cos_a * rel_centers[:, 1]
        ], axis=1)
        rotated_centers = rotated_rel_centers + center[None, :]
        
        # Rotate obstacle orientations
        rotated_thetas = thetas + rotation_angle
        
        obstacles = self.create_obstacles(rotated_centers, widths, heights, rotated_thetas)
        return obstacles, agent_start_xs

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        area_size = self.area_size
        start_y = self._params["start_y"] * area_size
        goal_y = self._params["goal_y"] * area_size
        passageway_width = self._params["passageway_width"] * area_size
        num_passages = int(self._params["num_passages"])
        wall_thickness = self._params["wall_thickness"] * area_size
        tunnel_end_y = self._params.get("tunnel_end_y", 0.40) * area_size
        goal_area_y_min = self._params.get("goal_area_y_min", 0.45) * area_size
        goal_area_y_max = self._params.get("goal_area_y_max", 0.95) * area_size
        
        # First, determine rotation and position offset (needed for goal placement)
        # We'll generate these first, then use them for both obstacles and goals
        rot_offset_key, goal_pre_key, obstacle_key, key = jr.split(key, 4)
        rotation_angle, position_offset = self._get_rotation_and_offset(rot_offset_key)
        
        # Center of rotation (with position offset applied)
        base_center = jnp.array([area_size / 2, area_size / 2])
        center = base_center + position_offset
        
        # Calculate passage centers (in unrotated coordinate system)
        x_start = (area_size - (num_passages * passageway_width + (num_passages - 1) * wall_thickness)) / 2
        passage_centers_unrotated = []
        for i in range(num_passages):
            passage_x = x_start + i * (passageway_width + wall_thickness) + passageway_width / 2
            passage_centers_unrotated.append(jnp.array([passage_x, (start_y + tunnel_end_y) / 2]))
        
        # Rotate passage centers and convert to JAX array for indexing
        cos_a = jnp.cos(rotation_angle)
        sin_a = jnp.sin(rotation_angle)
        passage_centers_list = []
        for passage_center_unrotated in passage_centers_unrotated:
            rel_pos = passage_center_unrotated - center
            rotated_rel = jnp.array([
                cos_a * rel_pos[0] - sin_a * rel_pos[1],
                sin_a * rel_pos[0] + cos_a * rel_pos[1]
            ])
            passage_centers_list.append(rotated_rel + center)
        
        # Calculate passage centers (in unrotated coordinate system) - needed for goal placement
        x_start = (area_size - (num_passages * passageway_width + (num_passages - 1) * wall_thickness)) / 2
        passage_centers_unrotated = []
        for i in range(num_passages):
            passage_x = x_start + i * (passageway_width + wall_thickness) + passageway_width / 2
            passage_centers_unrotated.append(jnp.array([passage_x, (start_y + tunnel_end_y) / 2]))
        passage_centers_arr = jnp.stack(passage_centers_unrotated, axis=0)
        
        # Generate goals FIRST (in unrotated coordinates) so we can pass them to obstacle generation
        goal_key, agent_key, key = jr.split(key, 3)
        
        # Shuffle goal assignments so agents must navigate to different passages
        shuffle_key, goal_key = jr.split(goal_key, 2)
        goal_indices = jr.permutation(shuffle_key, jnp.arange(self.num_agents))
        
        # Generate goal positions in unrotated coordinates
        goal_positions_unrotated = []
        goal_yaws = []
        goal_keys_split = jr.split(goal_key, self.num_agents)
        
        for i in range(self.num_agents):
            # Use JAX modulo operation instead of int() - convert to int32 for indexing
            goal_passage_idx = (goal_indices[i] % num_passages).astype(jnp.int32)
            # Use JAX array indexing instead of Python list indexing
            goal_passage_center = passage_centers_arr[goal_passage_idx]
            
            # Place goal in open space past the tunnel exit (in unrotated coordinates)
            x_key, y_key, yaw_key = jr.split(goal_keys_split[i], 3)
            
            # In unrotated system: X near passage center, Y past tunnel_end_y
            # Ensure goals stay within bounds (accounting for rotation)
            # After rotation, goals should be within [car_radius, area_size - car_radius]
            # To ensure this, we need to constrain unrotated positions more conservatively
            car_radius = self._params["car_radius"] * area_size
            margin = car_radius * 1.5  # Extra margin for rotation
            goal_x_unrotated = jr.uniform(x_key, (), minval=margin, maxval=area_size - margin)
            goal_y_unrotated = jr.uniform(y_key, (), minval=goal_area_y_min, maxval=goal_area_y_max)
            goal_pos_unrotated = jnp.array([goal_x_unrotated, goal_y_unrotated])
            goal_positions_unrotated.append(goal_pos_unrotated)
            
            # Random orientation at goal
            yaw = jr.uniform(yaw_key, (), minval=-jnp.pi, maxval=jnp.pi)
            goal_yaws.append(yaw)
        
        goal_positions_unrotated = jnp.stack(goal_positions_unrotated)
        goal_yaws = jnp.stack(goal_yaws)[:, None]
        
        # Now generate obstacles, passing goal positions so they can avoid them
        obstacles, agent_start_xs = self._generate_corridor_obstacles(
            obstacle_key, rotation_angle, position_offset, goal_positions_unrotated=goal_positions_unrotated
        )
        
        # Rotate passage centers for agent placement
        cos_a = jnp.cos(rotation_angle)
        sin_a = jnp.sin(rotation_angle)
        passage_centers_list = []
        for passage_center_unrotated in passage_centers_unrotated:
            rel_pos = passage_center_unrotated - center
            rotated_rel = jnp.array([
                cos_a * rel_pos[0] - sin_a * rel_pos[1],
                sin_a * rel_pos[0] + cos_a * rel_pos[1]
            ])
            passage_centers_list.append(rotated_rel + center)
        passage_centers_arr = jnp.stack(passage_centers_list, axis=0)

        # Place agents at start positions (rotated coordinate system)
        agent_positions = []
        agent_yaws = []
        pos_key, agent_key = jr.split(agent_key, 2)
        
        # Get agent radius for safe placement
        car_radius = self._params["car_radius"] * area_size
        
        # Calculate start position in unrotated system (at start_y, along passage centers)
        for i in range(self.num_agents):
            # Use pre-calculated start x positions (unrotated) - these already have proper clearance
            start_x_unrotated = agent_start_xs[i]
            # Add very small random offset perpendicular to tunnel direction (already constrained in agent_start_xs)
            offset_key, pos_key = jr.split(pos_key, 2)
            # Minimal additional offset since agent_start_xs already accounts for wall clearance
            perp_offset = jr.uniform(offset_key, (), minval=-car_radius * 0.3, maxval=car_radius * 0.3)
            
            # Start position in unrotated system
            # Move agents further from entrance wall to avoid intersection
            safe_start_y = start_y + car_radius * 2.0 + 0.02 * area_size  # Safe distance from entrance wall
            
            # Final x position - ensure it stays within safe bounds
            final_x = start_x_unrotated + perp_offset
            # Clip to ensure we don't go outside passage bounds (agent_start_xs already accounts for this, but be safe)
            final_x = jnp.clip(final_x, car_radius + 0.01 * area_size, area_size - car_radius - 0.01 * area_size)
            
            agent_pos_unrotated = jnp.array([final_x, safe_start_y])
            
            # Rotate agent position
            rel_pos = agent_pos_unrotated - center
            rotated_rel = jnp.array([
                cos_a * rel_pos[0] - sin_a * rel_pos[1],
                sin_a * rel_pos[0] + cos_a * rel_pos[1]
            ])
            agent_pos_rotated = rotated_rel + center
            agent_positions.append(agent_pos_rotated)
            
            # Random orientation (not necessarily facing along tunnel)
            yaw_key, pos_key = jr.split(pos_key, 2)
            # Can face any direction, with slight bias toward tunnel direction
            tunnel_direction = rotation_angle + jnp.pi / 2  # Direction along tunnel
            # 50% chance to face roughly toward tunnel, 50% completely random
            bias_key, yaw_key = jr.split(yaw_key, 2)
            use_bias = jr.uniform(bias_key, ()) < 0.5
            random_yaw = jr.uniform(yaw_key, (), minval=-jnp.pi, maxval=jnp.pi)
            biased_yaw = tunnel_direction + jr.uniform(yaw_key, (), minval=-jnp.pi / 3, maxval=jnp.pi / 3)
            yaw = jnp.where(use_bias, biased_yaw, random_yaw)
            agent_yaws.append(yaw)

        agent_positions = jnp.stack(agent_positions)
        agent_yaws = jnp.stack(agent_yaws)[:, None]

        # Rotate goal positions and verify they're valid (in bounds and not intersecting obstacles)
        goal_positions_rotated = []
        car_radius = self._params["car_radius"] * area_size
        
        for i in range(self.num_agents):
            goal_pos_unrotated = goal_positions_unrotated[i]
            
            # Rotate goal position
            rel_pos = goal_pos_unrotated - center
            rotated_rel = jnp.array([
                cos_a * rel_pos[0] - sin_a * rel_pos[1],
                sin_a * rel_pos[0] + cos_a * rel_pos[1]
            ])
            goal_pos_rotated = rotated_rel + center
            
            # Ensure goal stays in bounds after rotation
            goal_pos_rotated = jnp.clip(
                goal_pos_rotated,
                car_radius + 0.01 * area_size,
                area_size - car_radius - 0.01 * area_size
            )
            
            # Check if goal intersects with obstacles (with sufficient clearance)
            goal_clearance = car_radius * 2.0  # Clearance around goal
            goal_intersects = inside_obstacles(goal_pos_rotated, obstacles, r=goal_clearance)
            
            # If goal intersects, try multiple times to find a nearby valid position
            # Use rejection sampling with small random offsets
            max_attempts = 5  # Reduced from 10 for better performance
            offset_key, key = jr.split(key, 2)
            offset_keys = jr.split(offset_key, max_attempts)
            
            def try_offset(offset_key):
                max_offset = car_radius * 3.0
                offset = jr.uniform(offset_key, (2,), minval=-max_offset, maxval=max_offset)
                candidate_pos = goal_pos_rotated + offset
                candidate_pos = jnp.clip(
                    candidate_pos,
                    car_radius + 0.01 * area_size,
                    area_size - car_radius - 0.01 * area_size
                )
                candidate_intersects = inside_obstacles(candidate_pos, obstacles, r=goal_clearance)
                return candidate_pos, candidate_intersects
            
            # Try multiple offsets
            candidates = jax.vmap(try_offset)(offset_keys)
            valid_candidates = ~candidates[1]  # candidates that don't intersect
            
            # Find first valid candidate
            valid_indices = jnp.where(valid_candidates, jnp.arange(max_attempts), max_attempts)
            first_valid_idx = jnp.argmin(valid_indices)
            
            # Use first valid candidate if goal intersects, otherwise use original
            candidate_pos = candidates[0][first_valid_idx]
            goal_pos_rotated = jnp.where(
                goal_intersects,
                candidate_pos,
                goal_pos_rotated
            )
            
            goal_positions_rotated.append(goal_pos_rotated)
        
        goal_positions = jnp.stack(goal_positions_rotated)

        # Zero velocities
        agent_v_x = jnp.zeros((self.num_agents, 1))
        agent_v_y = jnp.zeros((self.num_agents, 1))

        # Construct states
        states = jnp.concatenate([agent_positions, agent_yaws, agent_v_x, agent_v_y], axis=1)
        goals = jnp.concatenate([goal_positions, goal_yaws, agent_v_x, agent_v_y], axis=1)
        env_states = self.EnvState(states, goals, obstacles)

        return self.get_graph(env_states)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        # Scale normalized actions to actual physical ranges
        scaled_action = 0.5 * ((action + 1.0) * (self.PARAMS["action_upper"] - self.PARAMS["action_lower"])) + self.PARAMS["action_lower"]

        # Unpack
        ax = scaled_action[:, 0]
        ay = scaled_action[:, 1]
        yaw_rate = scaled_action[:, 2]
        vx = agent_states[:, 3]
        vy = agent_states[:, 4]
        yaw = agent_states[:, 2]

        # Compute global velocity derivatives
        x_dot = vx * jnp.cos(yaw) - vy * jnp.sin(yaw)
        y_dot = vx * jnp.sin(yaw) + vy * jnp.cos(yaw)

        x_dot_dot = ax
        y_dot_dot = ay

        yaw_dot = yaw_rate

        return jnp.stack([x_dot, y_dot, yaw_dot, x_dot_dot, y_dot_dot], axis=1)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        x_dot = self.agent_xdot(agent_states, action)
        n_state_agent_new = agent_states + x_dot * self.dt
        n_state_agent_new = n_state_agent_new.at[:, 2].set(
            jnp.arctan2(jnp.sin(n_state_agent_new[:, 2]), jnp.cos(n_state_agent_new[:, 2]))
        )
        n_state_agent_new = self.clip_state(n_state_agent_new)
        return n_state_agent_new

    def step(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # Calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # Clip action to reasonable values
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)

        # Episode ends when reaching max episode steps
        done = jnp.array(False)

        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == (self.num_agents, self.n_cost)
        assert done.shape == tuple()
        
        next_state = self.EnvState(next_agent_states, goals, obstacles)

        info = {}
        if get_eval_info:
            agent_pos = agent_states[:, :2]
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])

        return self.get_graph(next_state), reward, cost, done, info

    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)

        reward = jnp.zeros(()).astype(jnp.float32)

        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(goal_pos - agent_pos, axis=-1)
        reward -= dist2goal.mean() * 0.01

        goal_threshold = self._params["car_radius"]
        reward -= jnp.where(dist2goal > goal_threshold, 1.0, 0.0).mean() * 0.001

        agent_yaw = agent_states[:, 2]
        goal_yaw = goals[:, 2]
        yaw_err = jnp.arctan2(jnp.sin(agent_yaw - goal_yaw), jnp.cos(agent_yaw - goal_yaw))
        reward -= (1.0 - jnp.cos(yaw_err)).mean() * 0.001

        reward -= (jnp.linalg.norm(action[:,:3], axis=1) ** 2).mean() * 0.0001
        return reward

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = dist.min(axis=1)
        agent_cost = self._params["car_radius"] * 2 - min_dist
        obs_cost: Array
        if self._params["top_k_rays"] > 0:
            obs_states = graph.type_states(
                type_idx=QuadrupedCorridor.OBS,
                n_type=self._params["top_k_rays"] * self.num_agents
            )[:, :2]
            obs_states = jnp.reshape(obs_states, (self.num_agents, self._params["top_k_rays"], 2))
            dist_obs = jnp.linalg.norm(obs_states - agent_pos[:, None, :], axis=-1)
            obs_cost = self.params["car_radius"] - dist_obs.min(axis=1)
        else:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)

        cost = jnp.stack([agent_cost, obs_cost], axis=1)
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)
        return cost

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        if viz_opts is None:
            viz_opts = {}

        viz_opts.setdefault('show_orientation', True)
        viz_opts.setdefault('arrow_length', self.params["car_radius"] * 2.0)

        new_trajs = np.array(self.trajs)
        if new_trajs.size > 0:
            new_trajs = new_trajs[::2]
        else:
            new_trajs = None

        rollout_result = self._to_rollout_result(rollout)

        render_quadruped_video(
            rollout=rollout_result,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["top_k_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            trajs=new_trajs,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: Pos2d) -> list[EdgeBlock]:
        n_hits = self._params["top_k_rays"] * self.num_agents

        agent_feats = state.agent
        agent_pos = agent_feats[:, :2]
        pos_diff = agent_feats[:, None, :] - agent_feats[None, :, :]
        dist = jnp.linalg.norm(agent_pos[:, None, :] - agent_pos[None, :, :], axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1/self.ENV_SIZE)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(pos_diff, agent_agent_mask, id_agent, id_agent)

        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = agent_feats[:, None, :] - state.goal[None, :, :]
        agent_goal_edges = EdgeBlock(agent_goal_feats, agent_goal_mask, id_agent, id_goal)

        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["top_k_rays"], (i + 1) * self._params["top_k_rays"])
            lidar_feats = agent_pos[i, :2] - lidar_data[id_hits, :2]
            lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1/self.ENV_SIZE)
            agent_obs_mask = jnp.ones((1, self._params["top_k_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            lidar_feats_padded = jnp.pad(
                lidar_feats,
                ((0, 0), (0, self.state_dim - lidar_feats.shape[-1])),
                mode="constant"
            )
            agent_obs_edges.append(
                EdgeBlock(lidar_feats_padded[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def control_affine_dyn(self, state: State) -> Tuple[Array, Array]:
        assert state.ndim == 2

        vx = state[:, 3]
        vy = state[:, 4]
        yaw = state[:, 2]

        f = jnp.zeros_like(state)
        f = f.at[:, 0].set(vx * jnp.cos(yaw) - vy * jnp.sin(yaw))
        f = f.at[:, 1].set(vx * jnp.sin(yaw) + vy * jnp.cos(yaw))

        g = jnp.zeros((state.shape[0], self.state_dim, self.action_dim))
        g = g.at[:, 3, 0].set(1.0)
        g = g.at[:, 4, 1].set(1.0)
        g = g.at[:, 2, 2].set(1.0)

        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_feats = state[graph.receivers, :self.state_dim] - state[graph.senders, :self.state_dim]

        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState) -> GraphsTuple:
        n_hits = self._params["top_k_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, self.node_dim))
        # agent nodes
        node_feats = node_feats.at[: self.num_agents, : self.state_dim].set(state.agent)
        node_feats = node_feats.at[: self.num_agents, self.state_dim + 2].set(1.0)
        # goal nodes
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, : self.state_dim].set(state.goal)
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, self.state_dim + 1].set(1.0)
        # lidar nodes will be filled after padding data below

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(QuadrupedCorridor.GOAL)
        if n_hits > 0:
            node_type = node_type.at[-n_hits:].set(QuadrupedCorridor.OBS)

        get_lidar_vmap = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
                max_returns=self._params["top_k_rays"],
            )
        )

        lidar_data = merge01(get_lidar_vmap(state.agent[:, :2]))
        lidar_data_padded = jnp.concatenate(
            [
                lidar_data,
                jnp.zeros((lidar_data.shape[0], 1)),
                jnp.zeros((lidar_data.shape[0], 1)),
                jnp.zeros((lidar_data.shape[0], 1)),
            ],
            axis=1,
        )
        if n_hits > 0:
            node_feats = node_feats.at[-n_hits:, : self.state_dim].set(lidar_data_padded)
            node_feats = node_feats.at[-n_hits:, self.state_dim].set(1.0)

        edge_blocks = self.edge_blocks(state, lidar_data)

        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data_padded], axis=0),
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0.0, 0.0, -jnp.pi, -2.0/self.ENV_SIZE, -1.5/self.ENV_SIZE])
        upper_lim = jnp.array([self.area_size, self.area_size, jnp.pi, 3.0/self.ENV_SIZE, 1.5/self.ENV_SIZE])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(3) * -1.0
        upper_lim = jnp.ones(3)
        return lower_lim, upper_lim

    def add_traj(self, x_ref):
        x_ref_new = jax.device_get(x_ref)
        self.trajs.append(x_ref_new)

    def _to_rollout_result(self, rollout: Rollout) -> RolloutResult:
        graph0 = tree_index(rollout.graph, 0)
        Tp1_graph = tree_concat_at_front(graph0, rollout.next_graph, axis=0)

        T_cost = rollout.costs
        while T_cost.ndim > 1:
            T_cost = T_cost.mean(axis=-1)

        return RolloutResult(
            Tp1_graph=Tp1_graph,
            T_action=rollout.actions,
            T_reward=rollout.rewards,
            T_cost=T_cost,
            T_done=rollout.dones,
            T_info={},
            T_rnn_state=rollout.rnn_states,
        )

