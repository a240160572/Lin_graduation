"""Learning script for single agent problems.

Agents are based on `stable_baselines3`'s implementation of A2C, PPO SAC, TD3, DDPG.

Example
-------
To run the script, type in a terminal:

    $ python singleagent.py --env <env> --algo <alg> --obs <ObservationType> --act <ActionType> --cpu <cpu_num>

Notes
-----
Use:

    $ tensorboard --logdir ./results/save-<env>-<algo>-<obs>-<act>-<time-date>/tb/

To check the tensorboard results at:

    http://localhost:6006/

"""
import os
import time
from datetime import datetime
from sys import platform
import argparse
import subprocess
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env # Module cmd_util will be renamed to env_util https://github.com/DLR-RM/stable-baselines3/pull/197
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3 import PPO, DQN

from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env

from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

import shared_constants

EPISODE_REWARD_THRESHOLD = -0 # Upperbound: rewards are always negative, but non-zero
"""float: Reward threshold to halt the script."""
retrain = 0

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning experiments script')
    parser.add_argument('--env',        default='flythrugate',      type=str,             choices=['planning','takeoff', 'hover', 'flythrugate', 'tune'], help='Task (default: hover)', metavar='')
    parser.add_argument('--algo',       default='ppo',        type=str,             choices=['a2c', 'ppo', 'sac', 'td3', 'ddpg'],        help='RL agent (default: ppo)', metavar='')
    parser.add_argument('--obs',        default='xy',        type=ObservationType,                                                      help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',        default='xy',        type=ActionType,                                                           help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--cpu',        default='1',          type=int,                                                                  help='Number of training environments (default: 1)', metavar='')        
    ARGS = parser.parse_args()

    #### Save directory ########################################
    
    if retrain:
        filename = "/home/lin/gym_pybullet_drones/experiments/learning/results/save-flythrugate-ppo-kin-xy-04.13.2022_14.34.24"
    else:
        filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ARGS.env+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print out current git commit hash #####################
    # if platform == "linux" or platform == "darwin":
    #     git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
    #     with open(filename+'/git_commit.txt', 'w+') as f:
    #         f.write(str(git_commit))

    #### Warning ###############################################
    if ARGS.env == 'tune' and ARGS.act != ActionType.TUN:
        print("\n\n\n[WARNING] TuneAviary is intended for use with ActionType.TUN\n\n\n")
    if ARGS.act == ActionType.ONE_D_RPM or ARGS.act == ActionType.ONE_D_DYN or ARGS.act == ActionType.ONE_D_PID:
        print("\n\n\n[WARNING] Simplified 1D problem for debugging purposes\n\n\n")
    #### Errors ################################################
        if not ARGS.env in ['takeoff', 'hover']: 
            print("[ERROR] 1D action space is only compatible with Takeoff and HoverAviary")
            exit()
    if ARGS.act == ActionType.TUN and ARGS.env != 'tune' :
        print("[ERROR] ActionType.TUN is only compatible with TuneAviary")
        exit()
    if ARGS.algo in ['sac', 'td3', 'ddpg'] and ARGS.cpu!=1: 
        print("[ERROR] The selected algorithm does not support multiple environments")
        exit()

    #### Uncomment to debug slurm scripts ######################
    # exit()

    env_name = ARGS.env+"-aviary-v0"
    sa_env_kwargs = dict(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=ARGS.obs, act=ARGS.act)
    # train_env = gym.make(env_name, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=ARGS.obs, act=ARGS.act) # single environment instead of a vectorized one    

    if env_name == "flythrugate-aviary-v0":
        train_env = make_vec_env(FlyThruGateAviary,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=ARGS.cpu,
                                 seed=0
                                 )
    # check_env(FlyThruGateAviary,warn=True, skip_render_check=True)
    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)
    # check_env(train_env, warn=True, skip_render_check=True)
    # exit()
    #### On-policy algorithms ##################################
    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
                           ) # or None

    #https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
    # net_arch = [<shared layers>, dict(vf=[<non-shared value network layers>], pi=[<non-shared policy network layers>])]
    # eg: [128, dict(vf=[256], pi=[16])]
    #           obs
    #             |
    #          <128>
    #   /               \
    # <16>             <256>
    #  |                |
    # action            value

    if ARGS.algo == 'ppo':
        model = PPO(a2cppoMlpPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    ) if ARGS.obs in [ObservationType.KIN,ObservationType.XY] else PPO(a2cppoCnnPolicy,
                                                                  train_env,
                                                                  policy_kwargs=onpolicy_kwargs,
                                                                  tensorboard_log=filename+'/tb/',
                                                                  verbose=1
                                                                  )
    

    #### Create eveluation environment #########################
    if ARGS.obs in [ObservationType.KIN,ObservationType.XY]: 
        eval_env = gym.make(env_name,
                            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                            obs=ARGS.obs,
                            act=ARGS.act
                            )

    #### Train the model #######################################
    # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=filename+'-logs/', name_prefix='rl_model')
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1
                                                     )
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(200/ARGS.cpu),
                                 deterministic=True,
                                 render=False
                                 )

    
    if retrain: 
        trained_model = PPO.load(filename+"/success_model.zip", tensorboard_log=filename+"/tb")

        trained_model.set_env(train_env)  
        trained_model.learn(int(1e2 * 2048),
                            callback=eval_callback,
                            log_interval=100,reset_num_timesteps=False)
        model.save(filename+'/retrained_model.zip')
    else:
        model.learn(total_timesteps= 1e2 * 2048,#int(1e12),
                    callback=eval_callback,
                    log_interval=100,
                    )
        
        #### Save the model ########################################
        model.save(filename+'/success_model.zip')
        print(filename)

    #### Print training progression ############################
    # with np.load(filename+'/evaluations.npz') as data:
    #     for j in range(data['timesteps'].shape[0]):
    #         # print(np.shape(data['results']),data['results'],j)
    #         print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    # exit()
    # # Continue training
    # filename = "/home/lin/gym-pybullet-drones/experiments/learning/results/save-flythrugate-ppo-kin-dyn-03.18.2022_15.21.22"
    # trained_model = PPO.load(filename+"/best_model.zip", tensorboard_log=filename+"/tb")

    # trained_model.set_env(train_env)  
    # trained_model.learn(int(1e12),
    #                     callback=eval_callback,
    #                     log_interval=100,reset_num_timesteps=False)
    # model.save(filename+'/retrained_model.zip')