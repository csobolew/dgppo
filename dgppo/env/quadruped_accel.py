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


class QuadrupedAccel(MultiAgentEnv):

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
        "comm_radius": 1.0 / 3.0,
        "n_rays": 32,
        "top_k_rays": 8,
        "obs_len_range": [0.5/15.0, 0.5/5.0],
        "n_obs": 3,
        "action_lower": np.array([-1.50 / ENV_SIZE, -1.50 / ENV_SIZE, -2.0]),
        "action_upper": np.array([1.50 / ENV_SIZE, 1.50 / ENV_SIZE, 2.0]),
        "default_area_size": 1.0,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = QuadrupedAccel.PARAMS["default_area_size"] if area_size is None else area_size
        super(QuadrupedAccel, self).__init__(num_agents, area_size, max_step, dt, params)
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

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * jnp.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # Randomly generate agent and goal position (x, y)
        pos_key, key = jr.split(key, 2)
        states_pos, goals_pos = get_node_goal_rng(
            key=pos_key,
            side_length=self.area_size,
            dim=2,
            n=self.num_agents,
            min_dist=4 * self.params["car_radius"],
            obstacles=obstacles,
            max_travel=self.max_travel,
        )

        # Randomly generate agent and goal orientation (yaw)
        agent_yaw_key, key = jr.split(key, 2)
        agent_orientations = jr.uniform(agent_yaw_key, (self.num_agents, 1), minval=-jnp.pi, maxval=jnp.pi)

        goal_yaw_key, key = jr.split(key, 2)
        goal_orientations = jr.uniform(goal_yaw_key, (self.num_agents, 1), minval=-jnp.pi, maxval=jnp.pi)

        agent_v_x = jnp.zeros((self.num_agents, 1))
        agent_v_y = jnp.zeros((self.num_agents, 1))

        states = jnp.concatenate([states_pos, agent_orientations, agent_v_x, agent_v_y], axis=1)
        goals = jnp.concatenate([goals_pos, goal_orientations, agent_v_x, agent_v_y], axis=1)
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

        # speed = jnp.linalg.norm(agent_states[:, 3:5], axis=-1)
        # reward -= (speed ** 2).mean() * 0.001  # Match action penalty weight

        # Encourage (but do not require) agents to face the direction of motion.
        #velocity = agent_states[:, 3:5]
        #speed = jnp.linalg.norm(velocity, axis=-1)
        #heading = jnp.stack([jnp.cos(agent_yaw), jnp.sin(agent_yaw)], axis=-1)
        #normalized_vel = jnp.where(speed[:, None] > 1e-6, velocity / speed[:, None], jnp.zeros_like(velocity))
        #heading_alignment = (heading * normalized_vel).sum(axis=-1)
        #reward += (heading_alignment * jnp.where(speed > 1e-3, 1.0, 0.0)).mean() * 0.0001

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
                type_idx=QuadrupedAccel.OBS,
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
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(QuadrupedAccel.GOAL)
        if n_hits > 0:
            node_type = node_type.at[-n_hits:].set(QuadrupedAccel.OBS)

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
