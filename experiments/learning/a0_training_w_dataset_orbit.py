import numpy as np
import gym
from gym import spaces
import os
import torch as th
from os import walk

from stable_baselines3 import PPO, A2C, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from sb3_contrib import TRPO, RecurrentPPO

from datetime import datetime
import argparse
from stable_baselines3.common.callbacks import EvalCallback
from inspect import getframeinfo, stack
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pathlib import Path

MAX_STEP_NUM = 100 
NEW_ENV_RATE = 1      # for optimality
OBSTACLE_NUM = 2
DISTANCE_TOL = 0.1
gym.logger.set_level(40)

class planning(gym.Env):

  def __init__(self, dataset):
    self.dataset = dataset[:]

    super(planning, self).__init__()
    self.env_size = 2.0
    self.cf_rad = 0.06

    self.max_step = MAX_STEP_NUM
    self.step_num = 0
    self.scenatio_num = 0

    self.solved = False
    self.crash = False
    
    self.generate_goal()
    self.generate_obstacle()
    
    self.action_space = spaces.Box(low= -1 * np.ones(1),
                                   high= np.ones(1),
                                   dtype= np.float32
                                  )
    
    self.generate_ini_observation()                   


  def generate_goal(self):
      self.goal = self.dataset[self.scenatio_num, 0:2]
      self.ini_pos = self.dataset[self.scenatio_num, 2:4]

      self.pos = np.array(self.ini_pos)


  def generate_orbit(self):
    r_obstacle = np.linalg.norm(self.goal-self.pos)/3
    theta_obstacle = np.random.uniform(0, 2*np.pi) 

    obstacle1_pos = np.array([self.pos + (self.goal - self.pos)/2])
                  
    for i in range(1):
      add_on = np.random.randint(2,5)
      orb_x = np.zeros((MAX_STEP_NUM+1,))
      orb_y = np.zeros((MAX_STEP_NUM+1,))

      for j in range(add_on):
        r = r_obstacle*np.random.uniform(0.8, 1.2) 
        k = np.random.randint(2,10)
        alpha = np.random.uniform(0, 2*np.pi) 
        l = np.random.uniform(2, 3) *k

        theta = np.linspace(theta_obstacle, theta_obstacle+2*np.pi, MAX_STEP_NUM+1)
        x = r*(np.cos(theta) + (np.cos(k*theta + alpha)/l)*np.cos(theta)) + obstacle1_pos[0,0]
        y = r*(np.sin(theta) + (np.cos(k*theta + alpha)/l)*np.sin(theta)) + obstacle1_pos[0,1]

        orb_x = orb_x + x/add_on
        orb_y = orb_y + y/add_on

    self.obstacle2_pos = np.column_stack((orb_x, orb_y))
    self.obstacle3_pos = np.vstack((self.obstacle2_pos[int(len(self.obstacle2_pos)/2):len(self.obstacle2_pos),:], 
                                    self.obstacle2_pos[0:int(len(self.obstacle2_pos)/2),:]))
    return np.array([obstacle1_pos[0], self.obstacle2_pos[0,:], self.obstacle3_pos[0,:]])

  def generate_obstacle(self):
    self.obstacle_rad = 0.15
    # ob_dir = np.array([-self.goal[1]+self.ini_pos[1], self.goal[0]-self.ini_pos[0]])

    # ini_obstacle_pos_updated = np.array([self.dataset[self.scenatio_num, 4:6], 
    #                                      self.dataset[self.scenatio_num, 6:8]])

    # # obstacle dynamics
    # obstacle_travel = self.dataset[self.scenatio_num, 8:10]
    # obstacle_direction = self.dataset[self.scenatio_num, 10:12]
    # self.obstacle_axis_dir = obstacle_direction * obstacle_travel

    # obstacle_angle = self.dataset[self.scenatio_num, 12:14]
    # self.obstacle_axis = np.array([[ob_dir[0] * np.cos(obstacle_angle[0]) - ob_dir[1] * np.sin(obstacle_angle[0]), 
    #                                 ob_dir[0] * np.sin(obstacle_angle[0]) + ob_dir[1] * np.cos(obstacle_angle[0])],
    #                                [ob_dir[0] * np.cos(obstacle_angle[1]) - ob_dir[1] * np.sin(obstacle_angle[1]), 
    #                                 ob_dir[0] * np.sin(obstacle_angle[1]) + ob_dir[1] * np.cos(obstacle_angle[1])]])
    
    ini_obstacle_pos_updated = self.generate_orbit()

    self.obstacle_pos = np.array(ini_obstacle_pos_updated)
    self.ini_obstacle = np.array(ini_obstacle_pos_updated)

    # self.obstacle_randomness = {"dir":obstacle_direction,
    #                             "travel":obstacle_travel,
    #                             "diviation": obstacle_angle}

  def reset(self):
    self.crash = False
    self.pos[:] = self.ini_pos
    self.obstacle_pos[:] = self.ini_obstacle
    self.step_num = 0

    caller = getframeinfo(stack()[1][0])      # check the caller
    if caller.function == "reset":            # if the caller is from training function, check for optimality ratio
      env_reset = self.solved and np.random.random_sample() < NEW_ENV_RATE 
    else:                                     # if the caller is from validation function, always generate new scenario
      env_reset = True
    
    if env_reset:
      self.scenatio_num += 1 
      self.generate_goal()
      self.generate_obstacle()
      self.solved = False
      # print(self.scenatio_num, self.goal)

    observation = self.generate_ini_observation()
    return observation

  def generate_ini_observation(self):
    self.ini_observation = np.concatenate((self.pos, self.goal, self.obstacle_pos,  self.obstacle_pos), 
                                  axis=None)/self.env_size 
    self.observation_space = spaces.Box(low=np.ones(len(self.ini_observation))*-1, 
                                        high=np.ones(len(self.ini_observation)),
                                        dtype=np.float32)
    return self.ini_observation

  def step(self, action):
    
    # k = np.linalg.norm(self.goal-self.pos)/(2*self.env_size)
    k = 0.1

    self.step_num += 1

    ########### Update obstacle position #############
    old_obstacle_pos = np.array(self.obstacle_pos)

    # ob_dir = np.array([-self.goal[1]+self.ini_pos[1], self.goal[0]-self.ini_pos[0]])
    # if OBSTACLE_NUM == 1:
    #   self.obstacle_pos = self.ini_obstacle + np.sin(np.sin(self.step_num/(2*np.pi)))*(ob_dir/np.linalg.norm(ob_dir))
    # elif OBSTACLE_NUM ==2:
    #   self.obstacle_pos[0,:] = self.ini_obstacle[0,:] + self.obstacle_axis_dir[0] * np.sin(np.sin(self.step_num/(2*np.pi)))*(self.obstacle_axis[0]/np.linalg.norm(self.obstacle_axis[0]))
    #   self.obstacle_pos[1,:] = self.ini_obstacle[1,:] + self.obstacle_axis_dir[1] * np.sin(np.sin(self.step_num/(2*np.pi)))*(self.obstacle_axis[1]/np.linalg.norm(self.obstacle_axis[1]))
    
    self.obstacle_pos[0,:] = self.ini_obstacle[0,:]
    self.obstacle_pos[1,:] = self.obstacle2_pos[self.step_num,:]
    self.obstacle_pos[2,:] = self.obstacle3_pos[self.step_num,:]
    ########### Update drone position #############
    # dir_len = k * (0.5 + 0.5*action[0])
    dir_len = 0.1
    
    direction_to_goal = np.array(self.goal-self.pos)
    dir_vec =  direction_to_goal / np.linalg.norm(direction_to_goal)
    theta = 0 + action[0]*np.pi/2

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    steer_vec = np.dot(rot, dir_vec)

    self.pos = self.pos + dir_len*steer_vec

    ########### Update observation #############
    observation = np.concatenate((self.pos, self.goal , self.obstacle_pos,  old_obstacle_pos), 
                                      axis=None)/self.env_size 
    reward = self.compute_reward()
    done = self.compute_done()
    info = {}
    return observation, reward, done, info

  def compute_done(self):
    if self.crash == True or self.solved == True:   # crash
      return True
    elif self.step_num >= self.max_step:            # overdue
      self.solved = False
      return True
    else:
      return False

  def compute_reward(self):
    self.dis_to_obstacle = np.linalg.norm(self.pos - self.obstacle_pos,axis = 1)

    if max(abs(self.pos)) >= 1.0*self.env_size:                         # hit the boundary 
      # print("Wall", np.round(self.pos,2), self.goal)
      self.crash = True
      self.solved = False
    
    if any(self.dis_to_obstacle <= self.obstacle_rad + self.cf_rad):    # hit the obstacle
      # print("Obs ", np.round(self.pos,2), self.goal)
      self.crash = True
      self.solved = False
    
    ########### Distance penality #############
    # distance_penalty = -100 * np.linalg.norm(self.goal-self.pos)
    distance_penalty = -100 * (1-np.exp(-np.linalg.norm(self.goal-self.pos)/0.5))

    ########### Collision penality #############
    if self.crash == True:
      collision_penalty = -1e4 
    else:
      collision_penalty = 0
    
    ########### Hazard flight penality #############
    if any(self.dis_to_obstacle <= ((self.obstacle_rad+self.cf_rad)*1.5)):
      # hazard_penalty = np.max(-10 * np.exp(-10*np.sqrt(self.dis_to_obstacle) + 8))  # for r=0.5
      hazard_penalty = np.max(-10 * np.exp(-10*self.dis_to_obstacle + 5))  # for r=0.25
    else:
      hazard_penalty = 0

    ########### Realization reward #############
    if np.allclose(self.pos,self.goal,atol=DISTANCE_TOL):
      position_reward = 1e4 
      self.solved = True
    else:
      position_reward = 0 

    reward = distance_penalty + collision_penalty + hazard_penalty + position_reward

    return reward

  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()

def set_env(dataset):
  env = planning(dataset)

  # env = SubprocVecEnv([lambda: single_env, single_env])
  env.reset() 

  eval_callback = EvalCallback(env, 
                verbose=0,
                best_model_save_path=alg_dir, 
                log_path=alg_dir,
                eval_freq=MAX_STEP_NUM * 20 * 100,
                deterministic=True, render=False)
  return env, eval_callback


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
  parser.add_argument('--exp',                       type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
  parser.add_argument('--it',                        type=int)
  parser.add_argument('--note',                      type=str)
  ARGS = parser.parse_args()
  models_dir = ARGS.exp
  # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  if ARGS.exp is None:
    if ARGS.note is not None:
      models_dir = str(Path().absolute()) + "/model_logs/"+datetime.now().strftime("%m.%d_%H.%M")+"_orbit_" + ARGS.note
    else:
      models_dir = str(Path().absolute()) + "/model_logs/"+datetime.now().strftime("%m.%d_%H.%M")+"_orbit"

    if os.path.exists(models_dir) == False:
      os.makedirs(models_dir+'/')
  
  dataset = np.load('0_training_dataset.npy')
  th.utils.data.DataLoader(dataset, num_workers=10, pin_memory = True)  # To speed up the data loading

  alg_list = ["PPO","A2C","TRPO","RPPO", "SAC", 'DDPG'] #RPPO - recurrent PPO

  if ARGS.exp is not None:
    shuffled_dataset = np.random.permutation(dataset)

    alg_class_list = [PPO, A2C, TRPO, RecurrentPPO]
    # alg_class_list = [PPO, A2C]
    filenames = next(walk(models_dir), (None, [], None))[1]

    alg_index = input("Please type the algorithm index to continue training\n {} (seperate indexes with comma)\n".format(str(list(enumerate(filenames)))))
    model_index = [int(x) for x in alg_index.split(",")]
    

    for i in model_index:
      alg_dir = models_dir + filenames[i]
      env, eval_callback =set_env(shuffled_dataset)

      model = alg_class_list[int(filenames[i].split("_")[0])].load(alg_dir + "/success_{}.zip".format(alg_list[int(filenames[i].split("_")[0])]), device="cpu")
      model.set_env(env) 
      print("Training model: " + alg_list[int(filenames[i].split("_")[0])])

      if ARGS.it != []:
        model.learn(ARGS.it * model.n_steps, 
                    reset_num_timesteps=False,
                    log_interval= model.n_steps * 100, 
                    callback=eval_callback)
      else:
        model.learn(1000 * model.n_steps, reset_num_timesteps=False, 
                  log_interval=5000, callback=eval_callback)
      
      model.save(alg_dir+'/success_{}.zip'.format(alg_list[int(filenames[i].split("_")[0])]))
      model.save(alg_dir+'/success_{}_{}.zip'.format(alg_list[int(filenames[i].split("_")[0])], np.int_(model._total_timesteps/model.n_steps)))
    
    del model, env
  else:
    alg_index = input("Please select the algorithm\n 0) {}; 1) {}; 2) {}; 3) {} (seperate indexes with comma)\n".format(alg_list[0],alg_list[1],alg_list[2],alg_list[3]))
    model_index = [int(x) for x in alg_index.split(",")]
    # batch_size_list = [15,15,15,5,5,5]
    device_list = ["cuda", "cpu"]

    for ind, i in enumerate(model_index):
      print(ind)
      alg_dir = models_dir+"/{}_{}/".format(i, alg_list[i])
      counter = 1
      while os.path.exists(alg_dir):
        alg_dir = models_dir+"/{}_{}/".format(i, alg_list[i]+ str(counter))
        counter += 1

      env, eval_callback = set_env(dataset)
      if i == 0:
        model = PPO("MlpPolicy", env, 
                    verbose=0, 
                    n_steps = MAX_STEP_NUM * 20,
                    n_epochs = 1,
                    batch_size = 25,
                    device = "cpu",
                    policy_kwargs=dict(
                        optimizer_class=th.optim.RMSprop,
                        optimizer_kwargs=dict(
                        alpha=0.99, eps=1e-5, weight_decay=0,
                        )),
                    tensorboard_log=alg_dir)
        # model = PPO("MlpPolicy", env, 
        #             verbose=0, 
        #             n_steps = MAX_STEP_NUM * 20,
        #             tensorboard_log=alg_dir,
        #             policy_kwargs=dict(
        #                 optimizer_class=th.optim.RMSprop,
        #                 optimizer_kwargs=dict(
        #                 alpha=0.99, eps=1e-5, weight_decay=0,
        #                 )),
        #             learning_rate=7e-4, # match A2C's learning rate
        #             gae_lambda=1, # disable GAE
        #             n_epochs=1, # match PPO's and A2C's objective
        #             batch_size=MAX_STEP_NUM * 20, # perform update on the entire batch
        #             normalize_advantage=False, # don't normalize advantages
        #             clip_range_vf=None                     
        #             )
      elif i == 1:
        model = A2C("MlpPolicy", env, 
                    verbose=0, 
                    n_steps = MAX_STEP_NUM * 20,
                    device = device_list[1],
                    tensorboard_log=alg_dir)
      elif i == 2:
        model = TRPO("MlpPolicy", env, 
                    verbose=0, 
                    n_steps = MAX_STEP_NUM * 20,
                    batch_size = MAX_STEP_NUM,
                    tensorboard_log=alg_dir)
      elif i == 3:
        model = RecurrentPPO("MlpLstmPolicy", env,
                    verbose=0, 
                    n_steps = MAX_STEP_NUM * 20,
                    tensorboard_log=alg_dir)
      elif i == 4:
        model = SAC("MlpPolicy", env, 
                    verbose=0, 
                    learning_starts = MAX_STEP_NUM * 20,
                    tensorboard_log=alg_dir)
      elif i == 5:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", env, action_noise=action_noise, 
                    verbose=0, 
                    learning_starts = MAX_STEP_NUM * 20,
                    tensorboard_log=alg_dir)

      print("Training model: " + alg_list[i])

      if isinstance(ARGS.it, int):
        model.learn(ARGS.it * MAX_STEP_NUM * 20,
                    log_interval= MAX_STEP_NUM * 20 * 100, 
                    callback=eval_callback)
      else:
        model.learn(1000 * model.n_steps, callback=eval_callback, log_interval=5000)
      
      model.save(alg_dir+'/success_{}.zip'.format(alg_list[i]))
      model.save(alg_dir+'/success_{}_{}.zip'.format(alg_list[i], np.int_(model._total_timesteps/(MAX_STEP_NUM * 20))))
      # np.save("scenario_{}".format(env.scenatio_num), env.scenatio_num)

      del model, env

  print("Test result saved")

