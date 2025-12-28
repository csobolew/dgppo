#!/usr/bin/env python3
"""
Custom test script for the QuadrupedAccel environment that:
- Loads a trained model (like test.py)
- Lets you specify custom obstacle coordinates and agent / goal start coordinates
- Generates videos using the same rendering pipeline as the original test script

Coordinates are in environment units (same scale as training, typically [0, area_size] with
area_size≈1.0 for QuadrupedAccel).

Example:
    # Single goal
    python tools/test_quadruped_custom.py \
        --path logs/QuadrupedAccel/dgppo/seed0_1120114512_WCJZ \
        --agent-pos "[0.1, 0.1, 0.0]" \
        --goal-pos  "[0.9, 0.9, 0.0]" \
        --obstacles "[[0.3, 0.3, 0.1, 0.1, 0.0], [0.7, 0.7, 0.15, 0.15, 0.785]]" \
        --epi 1
    
    # Multiple waypoints (agent visits each sequentially)
    python tools/test_quadruped_custom.py \
        --path logs/QuadrupedAccel/dgppo/seed0_1120114512_WCJZ \
        --agent-pos "[0.1, 0.1, 0.0]" \
        --waypoints "[[0.3, 0.3, 0.0], [0.7, 0.5, 1.57], [0.9, 0.9, 0.0]]" \
        --waypoint-threshold 0.05 \
        --obstacles "[[0.5, 0.5, 0.1, 0.1, 0.0]]" \
        --epi 1
"""

import argparse
import datetime
import json
import os
import pathlib

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.env.obstacle import Rectangle
from dgppo.env.quadruped_accel import QuadrupedAccel
from dgppo.trainer.data import Rollout


def _parse_state_array(s: str, state_dim: int) -> np.ndarray:
    """Parse '[x,y,...]' or '[[...],[...]]' into (n, state_dim) numpy array."""
    arr = np.array(json.loads(s), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < state_dim:
        pad = np.zeros((arr.shape[0], state_dim - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > state_dim:
        arr = arr[:, :state_dim]
    return arr


def _parse_obstacles(s: str) -> np.ndarray:
    """Parse obstacle array string into (n, 5) [x, y, width, height, theta]."""
    obs = np.array(json.loads(s), dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[1] != 5:
        raise ValueError("Obstacles must be shape (n, 5): [x, y, width, height, theta].")
    return obs


def _make_rectangles_from_obstacles(env: QuadrupedAccel, obstacle_entries: np.ndarray) -> Rectangle:
    """Create Rectangle obstacle struct from absolute obstacle params."""
    if obstacle_entries.size == 0:
        # No obstacles
        return Rectangle(
            type=jnp.zeros((0, 1), dtype=jnp.float32),
            center=jnp.zeros((0, 2), dtype=jnp.float32),
            width=jnp.zeros((0,), dtype=jnp.float32),
            height=jnp.zeros((0,), dtype=jnp.float32),
            theta=jnp.zeros((0,), dtype=jnp.float32),
            points=jnp.zeros((0, 4, 2), dtype=jnp.float32),
        )

    centers = jnp.array(obstacle_entries[:, :2], dtype=jnp.float32)
    widths = jnp.array(obstacle_entries[:, 2], dtype=jnp.float32)
    heights = jnp.array(obstacle_entries[:, 3], dtype=jnp.float32)
    thetas = jnp.array(obstacle_entries[:, 4], dtype=jnp.float32)
    rectangles = env.create_obstacles(centers, widths, heights, thetas)
    return rectangles


def _build_custom_env_state(
    env: QuadrupedAccel,
    base_state: QuadrupedAccel.EnvState,
    agent_pos: np.ndarray | None,
    goal_pos: np.ndarray | None,
    obstacles: np.ndarray | None,
) -> QuadrupedAccel.EnvState:
    """Return a new EnvState with optional custom agent/goal/obstacle values."""
    num_agents = env.num_agents
    state_dim = env.state_dim

    # Agents
    if agent_pos is None:
        agents_np = np.asarray(base_state.agent, dtype=np.float32)
    else:
        agents_np = np.asarray(agent_pos, dtype=np.float32)

    if agents_np.shape[0] == 1 and num_agents > 1:
        agents_np = np.tile(agents_np, (num_agents, 1))
    if agents_np.shape[0] != num_agents:
        raise ValueError(f"Expected {num_agents} agents, got {agents_np.shape[0]}.")
    if agents_np.shape[1] != state_dim:
        raise ValueError(f"Agent state_dim mismatch: expected {state_dim}, got {agents_np.shape[1]}.")

    # Goals
    if goal_pos is None:
        goals_np = np.asarray(base_state.goal, dtype=np.float32)
    else:
        goals_np = np.asarray(goal_pos, dtype=np.float32)

    if goals_np.shape[0] == 1 and num_agents > 1:
        goals_np = np.tile(goals_np, (num_agents, 1))
    if goals_np.shape[0] != num_agents:
        raise ValueError(f"Expected {num_agents} goals, got {goals_np.shape[0]}.")
    if goals_np.shape[1] != state_dim:
        raise ValueError(f"Goal state_dim mismatch: expected {state_dim}, got {goals_np.shape[1]}.")

    # Obstacles
    if obstacles is None:
        rects = base_state.obstacle
    else:
        rects = _make_rectangles_from_obstacles(env, obstacles)

    return env.EnvState(
        agent=jnp.array(agents_np, dtype=jnp.float32),
        goal=jnp.array(goals_np, dtype=jnp.float32),
        obstacle=rects,
    )


def _rollout_from_custom_state(
    env: QuadrupedAccel,
    act_fn,
    init_rnn_state,
    init_state: QuadrupedAccel.EnvState,
    max_steps: int,
    waypoints: list[np.ndarray] | None = None,
    waypoint_threshold: float = 0.05,
) -> Rollout:
    """
    Run a rollout starting from a given EnvState and return a Rollout object
    that can be passed directly to env.render_video.
    
    If waypoints are provided, the goal will be updated to the next waypoint
    when the agent gets within waypoint_threshold distance.
    """
    graph = env.get_graph(init_state)
    rnn_state = init_rnn_state
    
    # Initialize waypoint tracking
    current_waypoint_idx = 0
    waypoints_completed = []
    
    if waypoints is not None and len(waypoints) > 0:
        # Ensure waypoints have correct shape
        for i, wp in enumerate(waypoints):
            if wp.shape[0] == 1 and env.num_agents > 1:
                waypoints[i] = np.tile(wp, (env.num_agents, 1))
            if waypoints[i].shape[0] != env.num_agents:
                raise ValueError(f"Waypoint {i} has {waypoints[i].shape[0]} agents, expected {env.num_agents}")
        
        # Set initial goal to first waypoint
        if waypoints[0].shape[1] < env.state_dim:
            pad = np.zeros((waypoints[0].shape[0], env.state_dim - waypoints[0].shape[1]), dtype=np.float32)
            first_wp = np.concatenate([waypoints[0], pad], axis=1)
        else:
            first_wp = waypoints[0][:, :env.state_dim]
        
        # Update goal in graph to first waypoint
        new_env_state = graph.env_states._replace(goal=jnp.array(first_wp, dtype=jnp.float32))
        graph = env.get_graph(new_env_state)
        print(f"Starting with waypoint 1/{len(waypoints)} at position {first_wp[0, :2]}")

    graphs = []
    actions = []
    rnn_states = []
    rewards = []
    costs = []
    dones = []
    next_graphs = []

    for step in range(max_steps):
        graphs.append(graph)
        rnn_states.append(rnn_state if rnn_state is not None else jnp.zeros((1,)))

        action_eval = act_fn(graph, rnn_state) if rnn_state is not None else act_fn(graph)
        if isinstance(action_eval, tuple):
            action_out, rnn_state = action_eval
        else:
            action_out = action_eval
        action = action_out

        next_graph, reward, cost, done, _ = env.step(graph, action)
        
        # Handle waypoint tracking and goal updates
        if waypoints is not None and current_waypoint_idx < len(waypoints):
            # Check if current waypoint is reached
            agent_states = next_graph.type_states(type_idx=0, n_type=env.num_agents)
            agent_pos = np.asarray(jax.device_get(agent_states[:, :2]))
            current_waypoint = waypoints[current_waypoint_idx][:, :2]
            
            # Check distance to waypoint (for all agents if multi-agent)
            distances = np.linalg.norm(agent_pos - current_waypoint, axis=1)
            min_distance = np.min(distances)
            waypoint_reached = min_distance < waypoint_threshold
            
            # If any agent is close enough, mark waypoint as complete and advance
            if waypoint_reached:
                waypoints_completed.append((current_waypoint_idx, step, min_distance))
                print(f"Step {step}: Waypoint {current_waypoint_idx + 1}/{len(waypoints)} reached "
                      f"(distance: {min_distance:.4f})")
                current_waypoint_idx += 1
            
            # Always update goal to match current waypoint (ensures correct visualization with position and orientation)
            if current_waypoint_idx < len(waypoints):
                current_wp = waypoints[current_waypoint_idx]
                if current_wp.shape[1] < env.state_dim:
                    pad = np.zeros((current_wp.shape[0], env.state_dim - current_wp.shape[1]), dtype=np.float32)
                    current_wp_full = np.concatenate([current_wp, pad], axis=1)
                else:
                    current_wp_full = current_wp[:, :env.state_dim]
                
                # Update goal in next_graph to match current waypoint (position and orientation)
                new_env_state = next_graph.env_states._replace(
                    goal=jnp.array(current_wp_full, dtype=jnp.float32)
                )
                next_graph = env.get_graph(new_env_state)
                
                if waypoint_reached:
                    print(f"Switching to waypoint {current_waypoint_idx + 1}/{len(waypoints)} "
                          f"at position {current_wp_full[0, :2]}, orientation {current_wp_full[0, 2]:.3f}")

        actions.append(action)
        rewards.append(reward)
        costs.append(cost)
        dones.append(done)
        next_graphs.append(next_graph)

        graph = next_graph
    
    # Print summary
    if waypoints is not None:
        print(f"\nWaypoint summary:")
        print(f"  Total waypoints: {len(waypoints)}")
        print(f"  Completed: {len(waypoints_completed)}")
        for wp_idx, step, dist in waypoints_completed:
            print(f"    Waypoint {wp_idx + 1}: reached at step {step} (distance: {dist:.4f})")
        if current_waypoint_idx < len(waypoints):
            print(f"  Final waypoint {current_waypoint_idx + 1} not reached")

    # Stack along time dimension
    graphs_arr = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *graphs)
    next_graphs_arr = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *next_graphs)
    actions_arr = jnp.stack(actions, axis=0)

    # rnn_states may be dummy if None; still keep shape consistent
    if rnn_state is None:
        rnn_states_arr = jnp.zeros((len(actions), 1))
    else:
        rnn_states_arr = jnp.stack(rnn_states, axis=0)

    rewards_arr = jnp.stack(rewards, axis=0)
    costs_arr = jnp.stack(costs, axis=0)
    dones_arr = jnp.stack(dones, axis=0)
    log_pis_arr = None

    return Rollout(
        graph=graphs_arr,
        actions=actions_arr,
        rnn_states=rnn_states_arr,
        rewards=rewards_arr,
        costs=costs_arr,
        dones=dones_arr,
        log_pis=log_pis_arr,
        next_graph=next_graphs_arr,
    )


def run(args):
    print(f"> Running test_quadruped_custom.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # load config
    with open(os.path.join(args.path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # create environment (QuadrupedAccel)
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=config.obs if args.obs is None else args.obs,
        max_step=args.max_step,
        full_observation=args.full_observation,
    )
    assert isinstance(env, QuadrupedAccel), "This script is intended for the QuadrupedAccel environment."

    # create algorithm
    path = args.path
    model_path = os.path.join(path, "models")
    if args.step is None:
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
    else:
        step = args.step
    print("step:", step)

    algo = make_algo(
        algo=config.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=config.cost_weight,
        actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
        lr_actor=config.lr_actor,
        lr_Vl=config.lr_Vl,
        max_grad_norm=2.0,
        seed=config.seed,
        use_rnn=config.use_rnn,
        rnn_layers=config.rnn_layers,
        use_lstm=config.use_lstm,
    )
    algo.load(model_path, step)

    if args.stochastic:
        def act_fn(graph, rnn_state, key):
            action, _, new_rnn_state = algo.step(graph, rnn_state, key)
            return action, new_rnn_state
        act_fn = jax.jit(act_fn)
    else:
        act_fn = algo.act
        act_fn = jax.jit(act_fn)
    init_rnn_state = algo.init_rnn_state

    # Parse custom inputs (agent/goal/obstacles/waypoints)
    agent_pos = _parse_state_array(args.agent_pos, env.state_dim) if args.agent_pos else None
    goal_pos = _parse_state_array(args.goal_pos, env.state_dim) if args.goal_pos else None
    obstacles = _parse_obstacles(args.obstacles) if args.obstacles else None
    
    # Parse waypoints (list of goals to visit sequentially)
    waypoints = None
    if args.waypoints:
        waypoint_list = json.loads(args.waypoints)
        waypoints = [_parse_state_array(json.dumps(wp), env.state_dim) for wp in waypoint_list]
        print(f"Loaded {len(waypoints)} waypoints")
        if goal_pos is not None:
            print("Warning: Both --goal-pos and --waypoints specified. --waypoints will be used.")

    # Roll out episodes and optionally save videos
    videos_dir = pathlib.Path(path) / "videos_custom" / f"{step}"
    videos_dir.mkdir(exist_ok=True, parents=True)

    for i_epi in range(args.epi):
        key = jr.PRNGKey(args.seed + i_epi)
        # Get a base state to use its (possibly random) obstacles and/or goals if not overridden
        base_graph = env.reset(key)
        base_state = base_graph.env_states

        # If waypoints are provided, use first waypoint as initial goal
        # Otherwise use the provided goal_pos or random goal from base_state
        initial_goal = None
        if waypoints is not None and len(waypoints) > 0:
            # First waypoint will be set in rollout function
            pass
        else:
            initial_goal = goal_pos
        
        custom_state = _build_custom_env_state(env, base_state, agent_pos, initial_goal, obstacles)

        max_steps = args.max_step if args.max_step is not None else env.max_episode_steps
        rollout = _rollout_from_custom_state(
            env, act_fn, init_rnn_state, custom_state, max_steps,
            waypoints=waypoints,
            waypoint_threshold=args.waypoint_threshold
        )

        # Make video if requested
        if not args.no_video:
            video_name = f"custom_n{num_agents}_epi{i_epi:02}"
            video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
            viz_opts = {}
            env.render_video(rollout, video_path, Ta_is_unsafe=None, viz_opts=viz_opts, dpi=args.dpi)
            print(f"Saved video to {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Custom QuadrupedAccel tester with manual start/obstacles.")

    # required arguments
    parser.add_argument("--path", type=str, required=True, help="Training run directory (same as in test.py).")

    # custom arguments
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--epi", type=int, default=1)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)

    # default arguments (mirroring test.py)
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=100)

    # NEW: custom initial conditions
    parser.add_argument("--agent-pos", type=str, default=None,
                        help="Agent start state(s) as JSON list; e.g. '[x,y,yaw]' or '[[...],[...]]'.")
    parser.add_argument("--goal-pos", type=str, default=None,
                        help="Goal state(s) as JSON list; same format as --agent-pos. Ignored if --waypoints is used.")
    parser.add_argument("--waypoints", type=str, default=None,
                        help="Multiple goals to visit sequentially as JSON array; e.g. '[[x1,y1,yaw1],[x2,y2,yaw2]]'. "
                              "Agent will switch to next waypoint when within --waypoint-threshold distance.")
    parser.add_argument("--waypoint-threshold", type=float, default=0.05,
                        help="Distance threshold to consider a waypoint reached (default: 0.05).")
    parser.add_argument("--obstacles", type=str, default=None,
                        help="Obstacles as JSON list of [x,y,width,height,theta].")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()


