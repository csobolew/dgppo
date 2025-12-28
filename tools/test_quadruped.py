#!/usr/bin/env python3
"""
Test script for QuadrupedAccel environment with custom obstacles, agent positions, and waypoints.
Runs without live IsaacSim updates.

Example usage:
    # Basic run with custom agent position and waypoints
    python tools/test_quadruped.py \\
        --log-dir logs/QuadrupedAccel/dgppo/seed0_1120114512_WCJZ \\
        --agent-pos '[0.1,0.1,0.0]' \\
        --waypoints '[[0.5,0.5,0.0],[0.9,0.9,1.57]]' \\
        --obstacles '[[0.3,0.3,0.1,0.1,0.0],[0.7,0.7,0.15,0.15,0.785]]' \\
        --visualize --save-trajectory --output-dir test_results

    # Using obstacle file
    python tools/test_quadruped.py \\
        --log-dir logs/QuadrupedAccel/dgppo/seed0_1120114512_WCJZ \\
        --agent-pos '[0.1,0.1]' \\
        --waypoints '[[0.5,0.5],[0.9,0.9]]' \\
        --obstacles-file obstacles.json \\
        --visualize
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection

from dgppo.algo import make_algo
from dgppo.env import ENV, DEFAULT_MAX_STEP
from dgppo.env.obstacle import Rectangle
from dgppo.utils.graph import GraphsTuple
from dgppo.env.quadruped_accel import QuadrupedAccel

# Import helper functions from policy_server
sys.path.insert(0, str(Path(__file__).parent))
from policy_server import (
    load_training_config,
    build_env_from_config,
    resolve_checkpoint_step,
    empty_rectangles,
    scale_states,
    scale_positions,
    rectangles_from_obstacles,
    normalize_states,
    normalize_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test QuadrupedAccel environment with custom configuration."
    )
    parser.add_argument(
        "--log-dir", type=Path, required=True,
        help="Training run directory containing config.yaml and models/."
    )
    parser.add_argument(
        "--step", type=int,
        help="Checkpoint step to load (defaults to the latest numeric folder)."
    )
    parser.add_argument(
        "--algo", default="dgppo",
        help="Algorithm name (default: dgppo)."
    )
    parser.add_argument(
        "--env", help="Override environment id from the config."
    )
    parser.add_argument(
        "--num-agents", type=int, default=1,
        help="Number of agents (default: 1)."
    )
    # Keep these names consistent with tools/policy_server.py so we can
    # re-use its build_env_from_config helper without AttributeError.
    parser.add_argument(
        "--num-obs", type=int,
        help="Override obstacle count (passed through to build_env_from_config)."
    )
    parser.add_argument(
        "--n-rays", type=int,
        help="Override LiDAR ray count (passed through to build_env_from_config)."
    )
    parser.add_argument(
        "--max-step", type=int, help="Override environment max episode length."
    )
    parser.add_argument(
        "--area-size", type=float,
        help="Override the physical area size used by the environment."
    )
    parser.add_argument(
        "--agent-pos", type=str,
        help="Agent initial position as '[x,y,yaw]' or '[x,y]' (normalized 0-1). "
             "Example: '[0.1,0.1,0.0]' or '[0.1,0.1]'"
    )
    parser.add_argument(
        "--waypoints", type=str,
        help="Waypoints as JSON array of [x,y,yaw] or [x,y] (normalized 0-1). "
             "Example: '[[0.5,0.5,0.0],[0.9,0.9,1.57]]' or '[[0.5,0.5],[0.9,0.9]]'"
    )
    parser.add_argument(
        "--obstacles", type=str,
        help="Obstacles as JSON array of [x,y,width,height,theta] (normalized 0-1). "
             "Example: '[[0.3,0.3,0.1,0.1,0.0],[0.7,0.7,0.15,0.15,0.785]]'"
    )
    parser.add_argument(
        "--obstacles-file", type=Path,
        help="JSON file containing obstacles array (same format as --obstacles)."
    )
    parser.add_argument(
        "--waypoint-threshold", type=float, default=0.05,
        help="Distance threshold to consider a waypoint reached (normalized, default: 0.05)."
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Directory to save results (trajectory data, visualization)."
    )
    parser.add_argument(
        "--save-trajectory", action="store_true",
        help="Save trajectory data to JSON file."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show matplotlib visualization of the run."
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force JAX to run on CPU."
    )
    parser.add_argument(
        "--no-jit", action="store_true",
        help="Disable JAX JIT compilation."
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO)."
    )
    return parser.parse_args()


def parse_state_string(state_str: str, state_dim: int = 5) -> np.ndarray:
    """Parse a state string like '[x,y,yaw]' or '[x,y]' into a full state array."""
    try:
        state_list = json.loads(state_str)
        state_array = np.array(state_list, dtype=np.float32)
        
        if state_array.ndim == 0:
            raise ValueError("State must be a list/array")
        if state_array.ndim == 1:
            state_array = state_array.reshape(1, -1)
        
        # Pad or truncate to state_dim
        if state_array.shape[1] < state_dim:
            # Pad with zeros for missing dimensions
            padding = np.zeros((state_array.shape[0], state_dim - state_array.shape[1]), dtype=np.float32)
            state_array = np.concatenate([state_array, padding], axis=1)
        elif state_array.shape[1] > state_dim:
            state_array = state_array[:, :state_dim]
        
        return state_array
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def create_custom_env_state(
    env: QuadrupedAccel,
    agent_pos: Optional[np.ndarray],
    waypoints: Optional[List[np.ndarray]],
    obstacles: Optional[np.ndarray],
) -> QuadrupedAccel.EnvState:
    """Create a custom environment state with specified obstacles, agent position, and waypoints."""
    num_agents = env.num_agents
    area_size = float(env.area_size)
    state_dim = env.state_dim
    
    # Default agent position if not provided
    if agent_pos is None:
        agent_pos = np.array([[0.1, 0.1, 0.0, 0.0, 0.0]], dtype=np.float32)
        if num_agents > 1:
            # Spread agents horizontally
            agent_pos = np.tile(agent_pos, (num_agents, 1))
            for i in range(num_agents):
                agent_pos[i, 0] = 0.1 + i * 0.1
    
    # Ensure correct shape
    if agent_pos.shape[0] != num_agents:
        if agent_pos.shape[0] == 1:
            agent_pos = np.tile(agent_pos, (num_agents, 1))
        else:
            raise ValueError(f"Expected {num_agents} agents, got {agent_pos.shape[0]}")
    
    # Scale agent positions to environment coordinates
    agents_scaled = scale_states(agent_pos, area_size)
    
    # Use first waypoint as initial goal, or default goal position
    if waypoints and len(waypoints) > 0:
        goal_pos = waypoints[0].copy()
    else:
        # Default goal position
        goal_pos = np.array([[0.9, 0.9, 0.0, 0.0, 0.0]], dtype=np.float32)
        if num_agents > 1:
            goal_pos = np.tile(goal_pos, (num_agents, 1))
            for i in range(num_agents):
                goal_pos[i, 0] = 0.9 - i * 0.1
    
    if goal_pos.shape[0] != num_agents:
        if goal_pos.shape[0] == 1:
            goal_pos = np.tile(goal_pos, (num_agents, 1))
        else:
            raise ValueError(f"Expected {num_agents} goals, got {goal_pos.shape[0]}")
    
    # Scale goal positions
    goals_scaled = scale_states(goal_pos, area_size)
    
    # Create obstacles
    if obstacles is not None and obstacles.size > 0:
        rectangles = rectangles_from_obstacles(env, obstacles)
    else:
        rectangles = empty_rectangles()
    
    return env.EnvState(
        agent=jnp.array(agents_scaled, dtype=jnp.float32),
        goal=jnp.array(goals_scaled, dtype=jnp.float32),
        obstacle=rectangles,
    )


def run_episode(
    env: QuadrupedAccel,
    act_fn,
    init_rnn_state,
    initial_state: QuadrupedAccel.EnvState,
    waypoints: Optional[List[np.ndarray]],
    waypoint_threshold: float,
    max_steps: int,
    logger: logging.Logger,
) -> Tuple[List[GraphsTuple], List[np.ndarray], List[float], List[np.ndarray]]:
    """Run an episode with waypoint following."""
    graphs = []
    actions = []
    rewards = []
    costs = []
    
    # Initialize
    graph = env.get_graph(initial_state)
    graphs.append(graph)
    rnn_state = init_rnn_state
    current_waypoint_idx = 0
    area_size = float(env.area_size)
    
    logger.info(f"Starting episode with {len(waypoints) if waypoints else 0} waypoints")
    
    for step in range(max_steps):
        # Get action from policy
        if rnn_state is None:
            action_eval = act_fn(graph._replace(env_states=None))
            if isinstance(action_eval, tuple):
                action_out = action_eval[0]
            else:
                action_out = action_eval
            action = np.asarray(jax.device_get(action_out))
        else:
            action_out, rnn_state = act_fn(graph._replace(env_states=None), rnn_state)
            action = np.asarray(jax.device_get(action_out))
        
        actions.append(action)
        
        # Step environment
        next_graph, reward, cost, done, info = env.step(graph, action, get_eval_info=False)
        graphs.append(next_graph)
        rewards.append(float(reward))
        costs.append(np.asarray(jax.device_get(cost)))
        
        # Check if waypoint reached and update goal
        if waypoints and current_waypoint_idx < len(waypoints):
            agent_states = next_graph.type_states(type_idx=0, n_type=env.num_agents)
            agent_pos = np.asarray(jax.device_get(agent_states[:, :2])) / area_size
            current_waypoint = waypoints[current_waypoint_idx][:, :2]
            
            # Check distance to waypoint
            distances = np.linalg.norm(agent_pos - current_waypoint, axis=1)
            if np.all(distances < waypoint_threshold):
                logger.info(f"Waypoint {current_waypoint_idx + 1}/{len(waypoints)} reached at step {step}")
                current_waypoint_idx += 1
                
                # Update goal to next waypoint
                if current_waypoint_idx < len(waypoints):
                    next_waypoint = waypoints[current_waypoint_idx].copy()
                    if next_waypoint.shape[0] != env.num_agents:
                        if next_waypoint.shape[0] == 1:
                            next_waypoint = np.tile(next_waypoint, (env.num_agents, 1))
                    goals_scaled = scale_states(next_waypoint, area_size)
                    next_graph = next_graph._replace(
                        env_states=next_graph.env_states._replace(
                            goal=jnp.array(goals_scaled, dtype=jnp.float32)
                        )
                    )
                    # Recompute graph with new goal
                    next_graph = env.get_graph(next_graph.env_states)
        
        graph = next_graph
        
        if done:
            logger.info(f"Episode ended at step {step}")
            break
    
    logger.info(f"Episode completed after {len(actions)} steps")
    return graphs, actions, rewards, costs


def visualize_trajectory(
    env: QuadrupedAccel,
    graphs: List[GraphsTuple],
    waypoints: Optional[List[np.ndarray]],
    output_path: Optional[Path] = None,
):
    """Visualize the trajectory."""
    area_size = float(env.area_size)
    radius = float(env.params.get("car_radius", 0.5))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(0.0, area_size)
    ax.set_ylim(0.0, area_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Agent Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Plot obstacles
    if graphs[0].env_states.obstacle.center.shape[0] > 0:
        obstacle = graphs[0].env_states.obstacle
        obstacle_points = np.asarray(jax.device_get(obstacle.points))
        patches = [Polygon(obstacle_points[i], closed=True) for i in range(obstacle_points.shape[0])]
        obstacle_collection = PatchCollection(patches, color="#8a0000", alpha=0.8, zorder=1)
        ax.add_collection(obstacle_collection)
    
    # Plot waypoints
    if waypoints:
        for i, wp in enumerate(waypoints):
            wp_scaled = wp[:, :2] * area_size
            for j, pos in enumerate(wp_scaled):
                ax.plot(pos[0], pos[1], 'go', markersize=10, markeredgecolor='k', markeredgewidth=2, zorder=3)
                ax.text(pos[0] + radius, pos[1] + radius, f'W{i+1}', fontsize=8, zorder=4)
    
    # Plot trajectory
    agent_trajectories = [[] for _ in range(env.num_agents)]
    for graph in graphs:
        states = np.asarray(jax.device_get(graph.states))
        agent_pos = states[:env.num_agents, :2]
        for i in range(env.num_agents):
            agent_trajectories[i].append(agent_pos[i])
    
    for i, traj in enumerate(agent_trajectories):
        traj_array = np.array(traj)
        ax.plot(traj_array[:, 0], traj_array[:, 1], 'b-', alpha=0.5, linewidth=1.5, label=f'Agent {i+1}' if env.num_agents > 1 else 'Agent')
    
    # Plot start and end positions
    if graphs:
        start_states = np.asarray(jax.device_get(graphs[0].states))
        end_states = np.asarray(jax.device_get(graphs[-1].states))
        start_pos = start_states[:env.num_agents, :2]
        end_pos = end_states[:env.num_agents, :2]
        
        for i in range(env.num_agents):
            start_circle = Circle((start_pos[i, 0], start_pos[i, 1]), radius=radius,
                                 facecolor="#0068ff", edgecolor="k", linewidth=1.5, zorder=5)
            ax.add_patch(start_circle)
            end_circle = Circle((end_pos[i, 0], end_pos[i, 1]), radius=radius * 0.8,
                               facecolor="#ff0068", edgecolor="k", linewidth=1.5, zorder=5)
            ax.add_patch(end_circle)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


def save_trajectory_data(
    graphs: List[GraphsTuple],
    actions: List[np.ndarray],
    rewards: List[float],
    costs: List[np.ndarray],
    env: QuadrupedAccel,
    output_path: Path,
):
    """Save trajectory data to JSON file."""
    area_size = float(env.area_size)
    
    trajectory_data = {
        "num_agents": env.num_agents,
        "area_size": area_size,
        "num_steps": len(actions),
        "states": [],
        "actions": [],
        "rewards": rewards,
        "costs": [],
    }
    
    for graph in graphs:
        states = np.asarray(jax.device_get(graph.states))
        agent_states = states[:env.num_agents]
        goals = states[env.num_agents:env.num_agents * 2]
        
        # Normalize states
        agent_states_norm = normalize_states(agent_states, area_size)
        goals_norm = normalize_states(goals, area_size)
        
        trajectory_data["states"].append({
            "agents": agent_states_norm.tolist(),
            "goals": goals_norm.tolist(),
        })
    
    for action in actions:
        trajectory_data["actions"].append(action.tolist())
    
    for cost in costs:
        trajectory_data["costs"].append(cost.tolist())
    
    with output_path.open("w") as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f"Trajectory data saved to {output_path}")


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), None)
    if log_level is None:
        raise ValueError(f"Unknown log level {args.log_level}")
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("test-quadruped")
    
    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")
    if args.no_jit:
        jax.config.update("jax_disable_jit", True)
    
    # Load config and build environment
    log_dir = args.log_dir.resolve()
    config = load_training_config(log_dir)
    env = build_env_from_config(config, args)
    
    # Override num_agents if specified
    if args.num_agents:
        env = QuadrupedAccel(
            num_agents=int(args.num_agents),
            area_size=float(env.area_size),
            max_step=int(args.max_step or getattr(config, "max_step", None) or DEFAULT_MAX_STEP),
            dt=0.03,
            params=env.params,
        )
    
    # Load algorithm and model
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=config.cost_weight,
        actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=getattr(config, "Vh_gnn_layers", 1),
        lr_actor=config.lr_actor,
        lr_Vl=config.lr_Vl,
        max_grad_norm=2.0,
        seed=config.seed,
        use_rnn=False,
        rnn_layers=config.rnn_layers,
        use_lstm=config.use_lstm
    )
    
    model_dir = log_dir / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing models/ directory in {log_dir}")
    
    step = resolve_checkpoint_step(model_dir, args.step)
    algo.load(str(model_dir), step)
    logger.info(f"Loaded checkpoint at step {step}")
    
    act_fn = algo.act if args.no_jit else jax.jit(algo.act)
    init_rnn_state = getattr(algo, "init_rnn_state", None)
    
    # Parse inputs
    agent_pos = None
    if args.agent_pos:
        agent_pos = parse_state_string(args.agent_pos, env.state_dim)
        if agent_pos.shape[0] == 1 and env.num_agents > 1:
            agent_pos = np.tile(agent_pos, (env.num_agents, 1))
    
    waypoints = None
    if args.waypoints:
        waypoint_list = json.loads(args.waypoints)
        waypoints = [parse_state_string(json.dumps(wp), env.state_dim) for wp in waypoint_list]
        # Ensure all waypoints have correct number of agents
        for i, wp in enumerate(waypoints):
            if wp.shape[0] == 1 and env.num_agents > 1:
                waypoints[i] = np.tile(wp, (env.num_agents, 1))
    
    obstacles = None
    if args.obstacles:
        obstacles = np.array(json.loads(args.obstacles), dtype=np.float32)
    elif args.obstacles_file:
        with args.obstacles_file.open("r") as f:
            obstacles = np.array(json.load(f), dtype=np.float32)
    
    # Create custom environment state
    initial_state = create_custom_env_state(env, agent_pos, waypoints, obstacles)
    
    # Run episode
    max_steps = args.max_step or getattr(config, "max_step", None) or DEFAULT_MAX_STEP
    graphs, actions, rewards, costs = run_episode(
        env, act_fn, init_rnn_state, initial_state, waypoints,
        args.waypoint_threshold, max_steps, logger
    )
    
    # Save results
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_trajectory:
        output_path = (args.output_dir / "trajectory.json") if args.output_dir else Path("trajectory.json")
        save_trajectory_data(graphs, actions, rewards, costs, env, output_path)
    
    if args.visualize:
        viz_path = (args.output_dir / "trajectory.png") if args.output_dir else None
        visualize_trajectory(env, graphs, waypoints, viz_path)
    
    logger.info("Test completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.getLogger("test-quadruped").exception("Fatal error: %s", exc)
        sys.exit(1)

