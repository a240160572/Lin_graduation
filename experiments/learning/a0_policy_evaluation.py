####################################################

#       Compute policy success rate

####################################################

from random import sample
import numpy as np
import os

import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import argparse
from a0_training_w_dataset import planning

from stable_baselines3.common.policies import ActorCriticPolicy

def policy_eva(eva_num, model, env):

  success_counter = 0

  for i in range(eva_num):
    observation = env.reset()
    for step in range(env.max_step):
      action, _ = model.predict(observation, deterministic=True)
      observation, reward, done, info = env.step(action)

      if done:
        # print(env.crash, env.solved, done, env.step_num)
        # print(reward)
        break

    if env.solved:
      success_counter += 1
    
  success_rate = 100*success_counter/eva_num
  
  print("success rate is {} out of {} ({}%)".format(success_counter,eva_num,success_rate))

  duration = .5  # seconds
  freq = 500  # Hz
  os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

  return success_counter,eva_num


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
  parser.add_argument('--exp',                           type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
  parser.add_argument('--model',   default='success',    type=str)
  parser.add_argument('--env',     default="training",   type=str)
  parser.add_argument('--it',      default=5000,         type=int)
  ARGS = parser.parse_args()
  models_dir = ARGS.exp

  eva_num = ARGS.it

  dataset = np.load('0_training_dataset.npy')
  env =  planning(dataset)

  if ARGS.model.isdigit():
    model = PPO.load(models_dir+"/success_model_{}.zip".format(int(ARGS.model)), tensorboard_log=ARGS.exp)
  else:
    model = PPO.load(models_dir+"/{}_model.zip".format(ARGS.model), tensorboard_log=ARGS.exp)
  model.set_env(env) 

  policy_eva(eva_num, model, env)