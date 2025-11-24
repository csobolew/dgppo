#!/usr/bin/env python3
import argparse
import copy
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import matplotlib.pyplot as plt
import queue
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle

from dgppo.algo import make_algo
from dgppo.env import ENV, DEFAULT_MAX_STEP
from dgppo.env.obstacle import Rectangle
from dgppo.utils.graph import GraphsTuple
from dgppo.env.quadruped_accel import QuadrupedAccel

OBSTACLE_FEATURES = 5
_GRAPH_VISUALIZER = None


class GraphVisualizer(threading.Thread):
    """Minimal live plotter to inspect agent/goal states inside the policy server."""

    def __init__(self, env: QuadrupedAccel):
        super().__init__(daemon=True)
        self.env = env
        self.area_size = float(env.area_size)
        self.n_agents = int(env.num_agents)
        self.radius = float(env.params.get("car_radius", 0.5))
        self._queue: "queue.Queue[tuple[np.ndarray, Optional[np.ndarray]]]" = queue.Queue(maxsize=2)
        self._reset_event = threading.Event()
        self._stop_event = threading.Event()
        self.start()

    def reset(self) -> None:
        self._reset_event.set()

    def update(self, graph: GraphsTuple) -> None:
        graph_np = jax.device_get(graph)
        states = np.asarray(graph_np.states)
        if states.shape[0] < self.n_agents * 2:
            raise ValueError("Graph does not contain enough nodes to extract agent/goal states.")

        obstacle_points = None
        obstacle = getattr(graph_np.env_states, "obstacle", None)
        if isinstance(obstacle, Rectangle):
            obstacle_points = np.asarray(obstacle.points)

        try:
            self._queue.put_nowait((states, obstacle_points))
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait((states, obstacle_points))

    def run(self) -> None:
        plt.ion()
        fig, ax = None, None
        agent_scatter = None
        goal_scatter = None
        obstacle_collection: Optional[PatchCollection] = None

        def init_fig():
            nonlocal fig, ax, agent_scatter, goal_scatter, obstacle_collection
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception:
                    pass
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=80)
            manager = getattr(fig.canvas, "manager", None)
            if manager and hasattr(manager, "set_window_title"):
                manager.set_window_title("Policy Server Preview")
            ax.set_xlim(0.0, self.area_size)
            ax.set_ylim(0.0, self.area_size)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Agents (blue), Goals (green)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            agent_scatter = []
            goal_scatter = []
            obstacle_collection = None
            fig.show()

        init_fig()

        while not self._stop_event.is_set():
            if self._reset_event.is_set():
                init_fig()
                self._reset_event.clear()

            try:
                states, obstacle_points = self._queue.get(timeout=0.1)
            except queue.Empty:
                plt.pause(0.01)
                continue

            agent_pos = states[:self.n_agents, :2]
            goal_pos = states[self.n_agents:self.n_agents * 2, :2]

            if not agent_scatter:
                agent_scatter = []
                for ii in range(self.n_agents):
                    circ = Circle(
                        (agent_pos[ii, 0], agent_pos[ii, 1]),
                        radius=self.radius,
                        facecolor="#0068ff",
                        edgecolor="k",
                        linewidth=1.5,
                        zorder=5,
                    )
                    ax.add_patch(circ)
                    agent_scatter.append(circ)
            else:
                for circ, pos in zip(agent_scatter, agent_pos):
                    circ.center = (pos[0], pos[1])

            if not goal_scatter:
                goal_scatter = []
                for ii in range(self.n_agents):
                    circ = Circle(
                        (goal_pos[ii, 0], goal_pos[ii, 1]),
                        radius=self.radius * 0.8,
                        facecolor="#2fdd00",
                        edgecolor="k",
                        linewidth=1.0,
                        zorder=4,
                    )
                    ax.add_patch(circ)
                    goal_scatter.append(circ)
            else:
                for circ, pos in zip(goal_scatter, goal_pos):
                    circ.center = (pos[0], pos[1])

            if obstacle_points is None or obstacle_points.size == 0:
                if obstacle_collection is not None:
                    obstacle_collection.remove()
                    obstacle_collection = None
            else:
                patches = [Polygon(obstacle_points[ii], closed=True) for ii in range(obstacle_points.shape[0])]
                new_collection = PatchCollection(patches, color="#8a0000", alpha=0.8, zorder=1)
                if obstacle_collection is not None:
                    obstacle_collection.remove()
                obstacle_collection = new_collection
                ax.add_collection(new_collection)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

    def stop(self) -> None:
        self._stop_event.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a trained DGPPO policy over HTTP.")
    parser.add_argument("--log-dir", type=Path, required=True,
                        help="Training run directory containing config.yaml and models/.")
    parser.add_argument("--step", type=int,
                        help="Checkpoint step to load (defaults to the latest numeric folder).")
    parser.add_argument("--algo", default="dgppo",
                        help="Algorithm name (default: dgppo).")
    parser.add_argument("--env", help="Override environment id from the config.")
    parser.add_argument("--num-agents", type=int, help="Override agent count from the config.")
    parser.add_argument("--max-step", type=int, help="Override environment max episode length.")
    parser.add_argument("--max-travel", type=float, help="Override travel budget.")
    parser.add_argument("--num-obs", type=int, help="Override obstacle count.")
    parser.add_argument("--n-rays", type=int, help="Override LiDAR ray count.")
    parser.add_argument("--area-size", type=float,
                        help="Override the physical area size used by the environment.")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Server bind port (default: 8000).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--cpu", action="store_true", help="Force JAX to run on CPU.")
    parser.add_argument("--no-jit", action="store_true", help="Disable JAX JIT compilation.")
    parser.add_argument("--dump-sample", type=str,
                        help="Write a sample world-state JSON to this path and exit. Use '-' for stdout.")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="PRNG seed used with --dump-sample (default: 0).")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable the live matplotlib visualization window.")
    return parser.parse_args()

def visualize_graph_once(env, graph):
    global _GRAPH_VISUALIZER
    if _GRAPH_VISUALIZER is None:
        _GRAPH_VISUALIZER = GraphVisualizer(env)
    _GRAPH_VISUALIZER.update(graph)


def warmup_policy(env, act_fn, init_rnn_state, logger: logging.Logger, seed: int) -> None:
    """Run a dummy inference to trigger JIT compilation before serving."""
    try:
        logger.info("Running warm-up inference to precompile the policy.")
        warmup_graph = env.reset(jax.random.PRNGKey(seed))._replace(env_states=None)
        if init_rnn_state is None:
            action_eval = act_fn(warmup_graph)
        else:
            action_eval = act_fn(warmup_graph, init_rnn_state)
        if isinstance(action_eval, tuple):
            action_out = action_eval[0]
        else:
            action_out = action_eval
        _ = np.asarray(jax.device_get(action_out))
        logger.info("Warm-up inference finished.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up inference failed: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))


def load_training_config(log_dir: Path) -> argparse.Namespace:
    cfg_path = log_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {log_dir}")
    with cfg_path.open("r") as handle:
        return yaml.load(handle, Loader=yaml.UnsafeLoader)


def build_env_from_config(config: argparse.Namespace, args: argparse.Namespace):
    cfg = vars(config)
    env_id = args.env or cfg.get("env") or "QuadrupedAccel"
    if env_id not in ENV:
        raise ValueError(f"Environment {env_id} is not registered.")
    env_cls = ENV[env_id]

    num_agents = args.num_agents or cfg.get("num_agents")
    if num_agents is None:
        raise ValueError("num_agents not found; pass --num-agents.")

    params = copy.deepcopy(env_cls.PARAMS)
    if args.num_obs is not None:
        params["n_obs"] = args.num_obs
    elif cfg.get("n_obs") is not None:
        params["n_obs"] = cfg.get("n_obs")

    if args.n_rays is not None:
        params["n_rays"] = args.n_rays
    elif cfg.get("n_rays") is not None:
        params["n_rays"] = cfg.get("n_rays")

    max_step = args.max_step or cfg.get("max_step") or DEFAULT_MAX_STEP

    area_size = args.area_size if args.area_size is not None else cfg.get("default_area_size")
    if area_size is None:
        raise ValueError("area_size missing in config and not provided via --area-size.")

    env = env_cls(
        num_agents=int(num_agents),
        area_size=float(area_size),
        max_step=int(max_step),
        dt=0.03,
        params=params,
    )

    return env


def resolve_checkpoint_step(model_dir: Path, requested: Optional[int]) -> int:
    if requested is not None:
        step_dir = model_dir / str(requested)
        if not step_dir.exists():
            raise FileNotFoundError(f"Checkpoint step {requested} not found in {model_dir}")
        return requested
    steps = [int(p.name) for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        raise RuntimeError(f"No numeric checkpoints found in {model_dir}")
    return max(steps)


def empty_rectangles() -> Rectangle:
    dtype = jnp.float32
    return Rectangle(
        type=jnp.zeros((0, 1), dtype=dtype),
        center=jnp.zeros((0, 2), dtype=dtype),
        width=jnp.zeros((0,), dtype=dtype),
        height=jnp.zeros((0,), dtype=dtype),
        theta=jnp.zeros((0,), dtype=dtype),
        points=jnp.zeros((0, 4, 2), dtype=dtype),
    )

def normalize_positions(arr: np.ndarray, divisor: float) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] < 2:
        raise ValueError("Expected a 2D array with at least two columns for positions.")
    out = out.copy()
    out[:, :2] /= divisor
    return out


def scale_positions(arr: np.ndarray, factor: float) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] < 2:
        raise ValueError("Expected a 2D array with at least two columns for positions.")
    out = out.copy()
    out[:, :2] *= factor
    return out

def normalize_states(arr: np.ndarray, divisor: float) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] < 5:
        raise ValueError("Expected arr to have shape (N, 5) for [x,y,yaw,vx,vy].")

    out[:, 0] /= divisor
    out[:, 1] /= divisor
    out[:, 3] /= divisor
    out[:, 4] /= divisor

    return out


def scale_states(arr: np.ndarray, factor: float) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] < 5:
        raise ValueError("Expected arr to have shape (N, 5).")

    out[:, 0] *= factor
    out[:, 1] *= factor
    out[:, 3] *= factor
    out[:, 4] *= factor

    return out


def rectangles_from_obstacles(env, obstacle_entries: np.ndarray) -> Rectangle:
    if obstacle_entries.size == 0:
        return empty_rectangles()
    if obstacle_entries.ndim != 2 or obstacle_entries.shape[1] != OBSTACLE_FEATURES:
        raise ValueError(
            f"Obstacles must be an array of shape (n, {OBSTACLE_FEATURES}) with [x, y, width, height, yaw]."
        )
    area_size = float(env.area_size)
    centers = scale_positions(obstacle_entries[:, :2], area_size)
    widths = np.asarray(obstacle_entries[:, 2], dtype=np.float32) * area_size
    heights = np.asarray(obstacle_entries[:, 3], dtype=np.float32) * area_size
    thetas = np.asarray(obstacle_entries[:, 4], dtype=np.float32)
    rectangles = env.create_obstacles(
        jnp.array(centers, dtype=jnp.float32),
        jnp.array(widths, dtype=jnp.float32),
        jnp.array(heights, dtype=jnp.float32),
        jnp.array(thetas, dtype=jnp.float32),
    )
    return rectangles


def world_state_to_graph(payload: Dict[str, Any], env) -> GraphsTuple:
    if "agents" not in payload:
        raise KeyError("Request missing 'agents' field.")
    if "goals" not in payload:
        raise KeyError("Request missing 'goals' field.")

    agents_np = np.asarray(payload["agents"], dtype=np.float32)
    goals_np = np.asarray(payload["goals"], dtype=np.float32)

    if agents_np.ndim != 2:
        raise ValueError("'agents' must be a 2D array-like structure.")
    if goals_np.ndim != 2:
        raise ValueError("'goals' must be a 2D array-like structure.")
    if agents_np.shape != goals_np.shape:
        raise ValueError("'agents' and 'goals' must have matching shapes.")
    if agents_np.shape[0] != env.num_agents:
        raise ValueError(f"Expected {env.num_agents} agents, received {agents_np.shape[0]}.")
    if agents_np.shape[1] != env.state_dim:
        raise ValueError(f"Agents should have dimension {env.state_dim}; received {agents_np.shape[1]}.")

    area_size = float(env.area_size)
    agents_scaled = scale_states(agents_np, area_size)
    goals_scaled = scale_states(goals_np, area_size)

    obstacle_entries = payload.get("obstacles", [])
    obstacle_array = np.asarray(obstacle_entries, dtype=np.float32)
    rectangles = rectangles_from_obstacles(env, obstacle_array)

    env_state = env.EnvState(jnp.array(agents_scaled, dtype=jnp.float32),
                             jnp.array(goals_scaled, dtype=jnp.float32),
                             rectangles)
    graph = env.get_graph(env_state)
    return graph


def env_state_to_payload(env_state, area_size: float) -> Dict[str, Any]:
    agents = normalize_states(env_state.agent, area_size).tolist()
    goals = normalize_states(env_state.goal, area_size).tolist()

    obstacle = env_state.obstacle
    if getattr(obstacle.center, "shape", (0,))[0] == 0:
        obstacles: list[list[float]] = []
    else:
        centers = normalize_positions(obstacle.center, area_size)
        widths = (np.asarray(obstacle.width, dtype=np.float32) / area_size)[:, None]
        heights = (np.asarray(obstacle.height, dtype=np.float32) / area_size)[:, None]
        thetas = np.asarray(obstacle.theta, dtype=np.float32)[:, None]
        obstacles = np.concatenate([centers, widths, heights, thetas], axis=1).tolist()

    return {
        "agents": agents,
        "goals": goals,
        "obstacles": obstacles,
    }


def dump_sample_observation(env, destination: str, seed: int, logger: logging.Logger) -> None:
    key = jax.random.PRNGKey(seed)
    graph = jax.device_get(env.reset(key))
    payload = env_state_to_payload(graph.env_states, float(env.area_size))
    text = json.dumps(payload, indent=2)
    if destination in {"-", "", "stdout"}:
        print(text)
    else:
        out_path = Path(destination)
        out_path.write_text(text)
        logger.info("Sample world state written to %s", out_path.resolve())


def make_request_handler(env, act_fn, step: int, logger: logging.Logger, init_rnn_state, viz_fn):
    rnn_state_lock = threading.Lock()
    rnn_state = init_rnn_state

    class _Handler(BaseHTTPRequestHandler):
        server_version = "DGPPOPlusPolicyServer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.client_address[0], fmt % args)

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in {"/", "/healthz"}:
                self._send_json(200, {"status": "ok", "step": step})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            nonlocal rnn_state
            if self.path != "/act":
                self._send_json(404, {"error": "not found"})
                return

            length_header = self.headers.get("Content-Length")
            if length_header is None:
                self._send_json(411, {"error": "Content-Length header missing"})
                return
            try:
                length = int(length_header)
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return

            raw = self.rfile.read(length)
            try:
                request = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            try:
                graph = world_state_to_graph(request, env)
            except (KeyError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to construct graph from request")
                self._send_json(500, {"error": f"Graph construction failed: {exc.__class__.__name__}"})
                return
            
            try:
                viz_fn(graph)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Visualization failed: %s", exc)

            try:
                if rnn_state is None:
                    action_eval = act_fn(graph._replace(env_states=None))
                    if isinstance(action_eval, tuple):
                        action_out = action_eval[0]
                    else:
                        action_out = action_eval
                    action = np.asarray(jax.device_get(action_out))
                else:
                    with rnn_state_lock:
                        action_out, new_rnn_state = act_fn(graph._replace(env_states=None), rnn_state)
                        action = np.asarray(jax.device_get(action_out))
                        rnn_state = new_rnn_state
            except BrokenPipeError:
                logger.warning("Client disconnected (broken pipe); resetting visualization state.")
                try:
                    viz_fn(reset=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Visualizer reset failed: %s", exc)
                self._send_json(499, {"error": "client disconnected"})
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Policy evaluation failed")
                self._send_json(500, {"error": f"Policy evaluation failed: {exc.__class__.__name__}"})
                return

            self._send_json(200, {"action": action.tolist(), "step": step})

    return _Handler


def start_server(env, act_fn, host: str, port: int, step: int, logger: logging.Logger, init_rnn_state, viz_fn) -> None:
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(
        (host, port),
        make_request_handler(env, act_fn, step, logger, init_rnn_state, viz_fn)
    )
    server.daemon_threads = True
    try:
        logger.info("Serving checkpoint %s on http://%s:%s", step, host, port)
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down policy server")
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), None)
    if log_level is None:
        raise ValueError(f"Unknown log level {args.log_level}")
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("policy-server")

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")
    if args.no_jit:
        jax.config.update("jax_disable_jit", True)

    log_dir = args.log_dir.resolve()
    config = load_training_config(log_dir)
    env = build_env_from_config(config, args)

    if args.dump_sample:
        dump_sample_observation(env, args.dump_sample, args.sample_seed, logger)
        return

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

    act_fn = algo.act if args.no_jit else jax.jit(algo.act)
    init_rnn_state = getattr(algo, "init_rnn_state", None)

    warmup_policy(env, act_fn, init_rnn_state, logger, args.sample_seed)

    if args.no_viz:
        def viz_fn(graph: GraphsTuple = None, reset: bool = False) -> None:  # noqa: ANN001
            return
    else:
        def viz_fn(graph: GraphsTuple = None, reset: bool = False) -> None:  # noqa: ANN001
            if reset or graph is None:
                global _GRAPH_VISUALIZER
                if _GRAPH_VISUALIZER is not None:
                    _GRAPH_VISUALIZER.reset()
                return
            visualize_graph_once(env, graph)

    start_server(env, act_fn, args.host, args.port, step, logger, init_rnn_state, viz_fn)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("policy-server").exception("Fatal error: %s", exc)
        sys.exit(1)
