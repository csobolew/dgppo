import argparse
import datetime
import functools as ft
import os
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.utils import test_rollout
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.utils import jax_jit_np, jax_vmap
from dgppo.utils.typing import Array


def test(args):
    print(f"> Running test.py {args}")

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

    # Prepare custom parameters for QuadrupedCorridor
    corridor_kwargs = {}
    env_id = config.env if args.env is None else args.env
    if env_id == 'QuadrupedCorridor':
        if hasattr(args, 'num_passages') and args.num_passages is not None:
            corridor_kwargs['num_passages'] = args.num_passages
        if hasattr(args, 'passageway_width') and args.passageway_width is not None:
            corridor_kwargs['passageway_width'] = args.passageway_width
        if hasattr(args, 'wall_thickness') and args.wall_thickness is not None:
            corridor_kwargs['wall_thickness'] = args.wall_thickness
        if hasattr(args, 'start_y') and args.start_y is not None:
            corridor_kwargs['start_y'] = args.start_y
        if hasattr(args, 'goal_y') and args.goal_y is not None:
            corridor_kwargs['goal_y'] = args.goal_y
    
    # create environments
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    # Only pass num_obs if it's provided and not QuadrupedCorridor (which doesn't use it)
    env_kwargs = {}
    if env_id != 'QuadrupedCorridor':
        obs_value = config.obs if args.obs is None else args.obs
        if obs_value is not None:
            env_kwargs['num_obs'] = obs_value
    
    env = make_env(
        env_id=env_id,
        num_agents=num_agents,
        max_step=args.max_step,
        full_observation=args.full_observation,
        **corridor_kwargs,
        **env_kwargs
    )

    # create algorithm
    path = args.path
    model_path = os.path.join(path, "models")
    if args.step is None:
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
    else:
        step = args.step
    print("step: ", step)

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
        def act_fn(x, z, rnn_state, key):
            action, _, new_rnn_state = algo.step(x, z, rnn_state, key)
            return action, new_rnn_state
        act_fn = jax.jit(act_fn)
    else:
        act_fn = algo.act
    act_fn = jax.jit(act_fn)
    init_rnn_state = algo.init_rnn_state

    # set up keys
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    # create rollout function
    rollout_fn = ft.partial(test_rollout,
                            env,
                            act_fn,
                            init_rnn_state,
                            stochastic=args.stochastic)
    rollout_fn = jax_jit_np(rollout_fn)

    def unsafe_mask(graph_: GraphsTuple) -> Array:
        cost = env.get_cost(graph_)
        return jnp.any(cost >= 0.0, axis=-1)

    is_unsafe_fn = jax_jit_np(jax_vmap(unsafe_mask))

    # test results
    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    rates = []

    # test
    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        rollout = rollout_fn(key_x0)
        is_unsafes.append(is_unsafe_fn(rollout.graph))

        epi_reward = rollout.rewards.sum()
        epi_cost = rollout.costs.max()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)
        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, safe rate: {safe_rate * 100:.3f}%")

        rates.append(np.array(safe_rate))

    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()

    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%"
    )

    # save results
    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{safe_mean * 100:.3f},{safe_std * 100:.3f}\n")

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos" / f"{step}"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        safe_rate = rates[ii] * 100
        video_name = f"n{num_agents}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}_sr{safe_rate:.0f}"
        viz_opts = {}
        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--path", type=str, required=True)

    # custom arguments
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--log", action="store_true", default=False)

    # QuadrupedCorridor-specific arguments
    parser.add_argument("--num-passages", type=int, default=None,
                        help="Number of vertical passageways (QuadrupedCorridor only)")
    parser.add_argument("--passageway-width", type=float, default=None,
                        help="Width of narrow passages, normalized (QuadrupedCorridor only)")
    parser.add_argument("--wall-thickness", type=float, default=None,
                        help="Thickness of walls between passages, normalized (QuadrupedCorridor only)")
    parser.add_argument("--start-y", type=float, default=None,
                        help="Y position where agents start, normalized (QuadrupedCorridor only)")
    parser.add_argument("--goal-y", type=float, default=None,
                        help="Y position where goals are placed, normalized (QuadrupedCorridor only)")

    # default arguments
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
