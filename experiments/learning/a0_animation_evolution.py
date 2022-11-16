####################################################

#      Animation for trajection evolution

####################################################

import os
import pybullet
import pybullet_data
import argparse

import time
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
parser.add_argument('--exp',    type=str,   help='example:--exp ./model_logs/08.30_18.08_t_ppo_opt90_25_1_adam_cpu___FIN/0_PPO/subs_log/')
parser.add_argument('--sce',    type=int,   help='Scenario index')
ARGS = parser.parse_args()

track_list = [0, 1, 2]
it_num = [13200]

# print(Path().absolute())
# print(Path().absolute().parents[1])

pybullet.connect(pybullet.GUI, options= "--mp4={}\"test_{}.mp4\" --mp4fps=100".format(ARGS.exp, ARGS.sce))
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

pybullet.resetDebugVisualizerCamera(cameraDistance=1,
                                cameraYaw=0,
                                cameraPitch=-89,
                                cameraTargetPosition=[0, 0, 2],
                                )

for log_num in track_list:
    log_data = np.load(ARGS.exp+'eva_result_{}_{}.npy'.format(it_num[log_num], ARGS.sce),allow_pickle='TRUE').item()
    
    
    pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    drone = pybullet.loadURDF(str(Path().absolute().parents[1])+"/gym_pybullet_drones/assets/cf2x.urdf", np.concatenate((log_data["ini_pos"], 0.75), axis=None))
    goal = pybullet.loadURDF(str(Path().absolute().parents[1])+"/gym_pybullet_drones/assets/goal.urdf", np.concatenate((log_data["goal_pos"], 0.75), axis=None))

    if "goal_fin" in log_data:
        goal1 = pybullet.loadURDF(str(Path().absolute().parents[1])+"/gym_pybullet_drones/assets/goal.urdf", np.concatenate((log_data["goal_fin"], 0.75), axis=None))
    # board = pybullet.loadURDF("/home/lin/graduation/gym_pybullet_drones/assets/board.urdf", [0,0,1], axis=None)

    print(log_data["obs_pos"])
    if "obs_pos" in log_data:
        if np.shape(np.atleast_2d(log_data["obs_pos"]))[0] == 1:
            obs = pybullet.loadURDF(str(Path().absolute().parents[1])+"/gym_pybullet_drones/assets/obs.urdf", np.concatenate((log_data["obs_pos"], 0.5), axis=None))
        elif np.shape(np.atleast_2d(log_data["obs_pos"]))[0] == 2:
            obs0 = pybullet.loadURDF(str(Path().absolute().parents[1])+"/gym_pybullet_drones/assets/obs.urdf", np.concatenate((log_data["obs_pos"][0], 0.5), axis=None))
            obs1 = pybullet.loadURDF(str(Path().absolute().parents[1])+"/gym_pybullet_drones/assets/obs.urdf", np.concatenate((log_data["obs_pos"][1], 0.5), axis=None))

        # for p in range(np.shape(np.atleast_2d(log_data["obs_pos"]))[0]):
        #     vars()["obs"+str(p)]=pybullet.loadURDF("/home/lin/graduation/gym_pybullet_drones/assets/obs.urdf", np.concatenate((log_data["obs_pos"][p], 0.5), axis=None))
        #     # obs = pybullet.loadURDF("/home/lin/graduation/gym_pybullet_drones/assets/obs.urdf", np.concatenate((log_data["obs_pos"][p], 0.5), axis=None))


    # pybullet.resetDebugVisualizerCamera(cameraDistance=.5,
    #                                  cameraYaw=-155,
    #                                  cameraPitch=-45,
    #                                  cameraTargetPosition=[.2,1.5,1],
    #                                  )

    old_pos = np.array(log_data["obs"][0])
    drone_x_ini = old_pos[0]
    drone_y_ini = old_pos[1]
    for i in range(75):
        # time.sleep(50)
        if i >= len(log_data["obs"]):
            time.sleep(0.1)
        else:
            pos = log_data["obs"][i]
            drone_x = np.linspace(old_pos[0] , pos[0] , num=10)
            drone_y = np.linspace(old_pos[1] , pos[1] , num=10)
            for j in range(10):
                pybullet.resetBasePositionAndOrientation(drone, [drone_x_ini, drone_y_ini, 0.75],pybullet.getQuaternionFromEuler([0,0,0]))
                if np.shape(np.atleast_2d(log_data["obs_pos"]))[0] == 1:
                    ob_x = np.linspace(old_pos[4] , pos[4] , num=10)
                    ob_y = np.linspace(old_pos[5] , pos[5] , num=10)
                    pybullet.resetBasePositionAndOrientation(obs, [ob_x[j], ob_y[j], 0.5],pybullet.getQuaternionFromEuler([0,0,0]))
                elif np.shape(np.atleast_2d(log_data["obs_pos"]))[0] == 2:
                    pybullet.resetBasePositionAndOrientation(obs0, 
                                                            [np.linspace(old_pos[4] , pos[4] , num=10)[j], 
                                                            np.linspace(old_pos[5] , pos[5] , num=10)[j], 0.5],
                                                            pybullet.getQuaternionFromEuler([0,0,0]))
                    pybullet.resetBasePositionAndOrientation(obs1, 
                                                            [np.linspace(old_pos[6] , pos[6] , num=10)[j], 
                                                            np.linspace(old_pos[7] , pos[7] , num=10)[j], 0.5],
                                                            pybullet.getQuaternionFromEuler([0,0,0]))
                
                # if j > 0:
                #     color_list = np.array([0,0,0])
                #     color_list[log_num] = 1
                #     print(color_list)
                #     pybullet.addUserDebugLine(lineFromXYZ=[drone_x[j], drone_y[j], 0.75],
                #                                 lineToXYZ=[drone_x[j-1], drone_y[j-1], 0.75],
                #                                 lineColorRGB=color_list,
                #                                 lineWidth=5,
                #                                 lifeTime=2000)
                time.sleep(0.01)
            old_pos = pos
    
    pybullet.removeBody(drone)
    pybullet.removeBody(goal)
    pybullet.removeBody(obs0)
    pybullet.removeBody(obs1)

    
pybullet.disconnect()

exit()
# print(" ini pos:", np.concatenate((log_data["ini_pos"], 0.75), axis=None),
#       "\ngoal pos:",np.concatenate((log_data["goal_pos"], 0.75), axis=None),
#       "\n end pos:", np.concatenate((log_data["obs"][-1], 0.75), axis=None))
# print(log_data["obs"])

