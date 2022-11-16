####################################################

#      Visualization plot of policy evolution

####################################################

from ntpath import join
import numpy as np

import matplotlib.pyplot as plt
from stable_baselines3 import PPO,A2C,DDPG, SAC
from sb3_contrib import TRPO, RecurrentPPO
import argparse
import os

# from a0_training_w_dataset_2action import planning
from os import walk

def plot_traj(log, line_color):
    # fig = plt.figure()
    # ax = plt.gca()
    # ax.axis('square')

    goal = plt.plot(log["goal_pos"][0], log["goal_pos"][1], 'ro')
    # circle_goal = plt.Circle((log["goal_pos"][0], log["goal_pos"][1]), 2, color='0.8', fill=False)
    # ax.add_patch(circle_goal)

    ini = plt.plot(log["ini_pos"][0], log["ini_pos"][1], 'bx')
    # print(log["ini_pos"][0], log["ini_pos"][1])
    obstacle_rad = 0.15

    # print(OBSTACLE_NUM)
    if "obs_pos" in log and len(np.shape(log["obs_pos"])) == 1:
      circle1 = plt.Circle((log["obs_pos"][0], log["obs_pos"][1]), obstacle_rad*1.25, color='0.2', fill=False)
      ax.add_patch(circle1)
      circle2 = plt.Circle((log["obs_pos"][0], log["obs_pos"][1]), obstacle_rad, color='0.8', fill=True)
      ax.add_patch(circle2)


    if len(np.shape(log["obs_pos"])) == 2:
        # print(log["obs_pos"])
        circle1 = plt.Circle((log["obs_pos"][0][0], log["obs_pos"][0][1]), obstacle_rad*1.5, color='0.2', fill=False)
        ax.add_patch(circle1)
        obs_ini = plt.plot(log["obs_pos"][0][0], log["obs_pos"][0][1], 'o', color='0.8')
        circle2 = plt.Circle((log["obs_pos"][0][0], log["obs_pos"][0][1]), obstacle_rad, color='0.8', fill=True)
        ax.add_patch(circle2)
        circle3 = plt.Circle((log["obs_pos"][1][0], log["obs_pos"][1][1]), obstacle_rad*1.5, color='0.2', fill=False)
        ax.add_patch(circle3)
        circle4 = plt.Circle((log["obs_pos"][1][0], log["obs_pos"][1][1]), obstacle_rad, color='0.8', fill=True)
        ax.add_patch(circle4)
    
    # if len(log["obs_pos"]) == 3:
    #   circle5 = plt.Circle((log["obs_pos"][2][0], log["obs_pos"][2][1]), obstacle_rad*1.5, color='0.2', fill=False)
    #   ax.add_patch(circle5)
    #   plt.plot(log["obs_pos"][2][0], log["obs_pos"][2][1], 'o', color='0.8')
    #   circle6 = plt.Circle((log["obs_pos"][2][0], log["obs_pos"][2][1]), obstacle_rad, color='0.8', fill=True)
    #   ax.add_patch(circle6)
    
    traj = plt.plot(log["obs"][:,0], log["obs"][:,1], color = line_color)
    obs_traj = plt.plot(log["obs"][:,4], log["obs"][:,5], color='0.8')
    # plt.plot(log["obs"][:,8], log["obs"][:,9], color='0.8')
    # plt.plot(log["obs"][:,6], log["obs"][:,7], color='0.8')

    plt.plot(log["obs"][:,6], log["obs"][:,7], color='0.8')
    # plt.plot(log["goal_fin"][0], log["goal_fin"][1], 'ro')
    plt.axis([-3, 3, -3, 3])
    
    # fig.savefig(path+'traj.png', dpi=fig.dpi)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
  parser.add_argument('--exp',                           type=str)
  parser.add_argument('--model',   default='success',    type=str,                help= 'example: --model 13200; default is the success model')
  parser.add_argument('--env',     default='def',        type=str,                help= 'select training environment')
  parser.add_argument('--subs',    default="False",      action='store_true',     help= 'whether plot the 5x5 policy visualization')
  parser.add_argument('--it',      default=5000,         type=int,                help= 'iteration number for success rate computation')
  ARGS = parser.parse_args()
  models_dir = ARGS.exp
  
  ################### Choose algorithm  ########################
  filenames = next(walk(models_dir))[1]
  filenames.sort()

  print("Please type the algorithm index to continue training")
  print(*list(enumerate(filenames)),sep='\n')
  # alg_index = input("(seperate indexes with comma)\n")
  # algo_index = [int(x) for x in alg_index.split(",")]
  alg_index = input("(seperate indexes with comma or type 'all')\n")
  algo_index = [int(x) for x in alg_index.split(",")] if alg_index != "all" else [x for x in range(len(filenames))]

  algo_list = ["PPO", "A2C", "TRPO", "RPPO", "SAC", "DDPG"]
  alg_class_list = [PPO, A2C, TRPO, RecurrentPPO, SAC, DDPG]

  for i in algo_index:
    ################### Choose model  ########################
    algo_type = [s for s in algo_list if s in filenames[i]]
    print("Algo: {}; Model: {}\n".format(algo_type, filenames[i]))

    print(int(filenames[i].split("_")[0]))
    if ARGS.model.isdigit():
      model = alg_class_list[int(filenames[i].split("_")[0])].load(models_dir+"/{}/success_{}_{}.zip".format(filenames[i],algo_type[0],ARGS.model), tensorboard_log=ARGS.exp)
    elif ARGS.model == "best":
      model = alg_class_list[int(filenames[i].split("_")[0])].load(models_dir+"/{}/best_model.zip".format(filenames[i]), tensorboard_log=ARGS.exp)
    else:
      # model = PPO.load(models_dir+"/{}/success_{}.zip".format(filenames[i],algo_type[0]), tensorboard_log=ARGS.exp)
      model = alg_class_list[int(filenames[i].split("_")[0])].load(models_dir+"/{}/success_{}.zip".format(filenames[i],algo_type[0]), tensorboard_log=ARGS.exp)
      print(int(filenames[i].split("_")[0]))
      print(model)

    dataset = np.load('0_testing_dataset.npy')

    if ARGS.env == "orbit":
      from a0_training_w_dataset_orbit import planning
    elif ARGS.env == "2act":
      from a0_training_w_dataset_2action import planning
    else:
      from a0_training_w_dataset import planning
    
    env =  planning(dataset)
    
    model.set_env(env) 

    plot_num = 4
    plot_num1 = 4

    if not os.path.exists(models_dir+'/{}/subs_log'.format(filenames[i])):
      os.makedirs(models_dir+'/{}/subs_log'.format(filenames[i]))

    it_count = ["13200", "26400", "33000"]
    color_map = ["r", "g", "b"]
    fig = plt.figure()
    if ARGS.subs == True:
        for policy_ind, policy_it in enumerate(it_count):
            model = alg_class_list[int(filenames[i].split("_")[0])].load(models_dir+"/{}/success_{}_{}.zip".format(filenames[i],algo_type[0],policy_it), tensorboard_log=ARGS.exp)
            env =  planning(dataset)
            for j in range(plot_num*plot_num1):
                plt.subplot(plot_num, plot_num1, int(j+1))
                ax = plt.gca()
                # ax.axis('square')
                plt.xticks([])              # set no ticks 
                plt.yticks([])
                fig.subplots_adjust(wspace=0, hspace=0)
                
                
                observation = env.reset()
                
                obs_list = {"ini_pos": env.ini_pos,
                            "goal_pos": env.goal,
                            "obs_pos": env.ini_obstacle,
                            "obs": np.round(np.array([observation*env.env_size]),2)}
 
                for step in range(env.max_step):
                    action, _ = model.predict(observation, deterministic=True)

                    observation, reward, done, info = env.step(action)
                    obs_list["obs"] = np.concatenate((obs_list["obs"], np.array([np.round(observation*env.env_size,2)])), axis=0)

                    if done:
                        break 

                plot = plot_traj(obs_list, line_color = color_map[policy_ind])

                np.save(models_dir+'/{}/subs_log/eva_result_{}.npy'.format(filenames[i], j), obs_list)
                # if j == 11:
                #   np.save(models_dir+'/eva_result.npy', obs_list)
            
        # from a0_policy_evaluation import policy_eva

        # success_counter,eva_num = policy_eva(eva_num = ARGS.it, model = model, env = env)

        # plt.suptitle("{} success flight out of {} tests ({}%)".format(success_counter, eva_num, 100*success_counter/eva_num),y=0.92)
        # plt.subplots_adjust(top=0.9)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [Line2D([0], [0], marker='o', lw=0,label='Goal position', markerfacecolor='r', markeredgecolor='w'),
                            Line2D([0], [0], marker='x', lw=0,label='Drone initial position', markerfacecolor='g'),
                            Patch(edgecolor='0.8', facecolor='0.8', label='Obstacle'),
                            # Line2D([0], [0], color ='0.8', lw=1.5, label='Obstacle trajectory'),
                            # Line2D([0], [0], color ='k', lw=1.5, label='Hazard bound'),
                            Line2D([0], [0], color ='r', lw=1.5, label='Policy after 20M training step'),
                            Line2D([0], [0], color ='g', lw=1.5, label='Policy after 40M training step'),
                            Line2D([0], [0], color ='b', lw=1.5, label='Policy after 50M training step')]
        

        ax.legend(handles=legend_elements, loc='lower center', 
                ncol=3,
                bbox_to_anchor=(-2, -.48), fontsize = 18)
        fig.set_size_inches(16, 8)

        legend_elements = [
                            Line2D([0], [0], color ='r', lw=1.5, label='Policy after 20M training step'),
                            Line2D([0], [0], color ='g', lw=1.5, label='Policy after 40M training step'),
                            Line2D([0], [0], color ='b', lw=1.5, label='Policy after 50M training step')]
        ax.legend(handles=legend_elements, loc='lower center', 
                ncol=2,
                bbox_to_anchor=(-1, -.48), fontsize = 18)
        fig.set_size_inches(11, 11)
        # plt.margins(y = 2)
        # plt.subplots_adjust(bottom=.5)
        # print(filenames[i])
        # fig.savefig(models_dir+'/{}/subs_{}_{}.png'.format(filenames[i], algo_type[0], ARGS.model), dpi=fig.dpi)
        
        plt.suptitle("Policy evolution of PPO-1", fontsize = 20, y=0.92)
        # fig.savefig(models_dir+'/{}/evoluation_{}_{}.eps'.format(filenames[i], algo_type[0], ARGS.model), dpi=fig.dpi,format='eps')
        plt.savefig(models_dir+'/{}/evoluation1_{}_{}.png'.format(filenames[i], algo_type[0], ARGS.model), bbox_inches='tight')
        plt.show()
        print(models_dir+'/{}/subs_{}_{}.eps'.format(filenames[i], algo_type[0], ARGS.model))
        print("Test subplot saved")

