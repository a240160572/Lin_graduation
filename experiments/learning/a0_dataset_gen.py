####################################################

#      Generate training/testing data

####################################################


import numpy as np

def generate_goal():
  goal_pos = np.round(3.5*(np.random.random_sample(2,)-0.5),2)
  drone_pos = np.round(3.5*(np.random.random_sample(2,)-0.5),2)

  while np.linalg.norm(goal_pos-drone_pos) <= 3.5:
    goal_pos = np.round(3.5*(np.random.random_sample(2,)-0.5),2)
    drone_pos = np.round(3.5*(np.random.random_sample(2,)-0.5),2)

  return drone_pos, goal_pos

def generate_obstacle(drone_pos, goal_pos):
  ob_dir = np.array([-goal_pos[1]+drone_pos[1], goal_pos[0]-drone_pos[0]])
  ini_obstacle_pos = np.array([drone_pos + (goal_pos - drone_pos)/3, 
                              drone_pos + 2*(goal_pos - drone_pos)/3])
  ini_obstacle_pos_updated = np.array([ini_obstacle_pos[0,:] + np.sin(np.sin(-np.random.randint(-5,5)/(2*np.pi)))*(ob_dir/np.linalg.norm(ob_dir)), 
                                        ini_obstacle_pos[1,:] + np.sin(np.sin(np.random.randint(-5,5)/(2*np.pi)))*(ob_dir/np.linalg.norm(ob_dir))]).reshape((4))
  
  return np.round(ini_obstacle_pos_updated,2)

def generate_obstacle_dynamics():
  obstacle_direction = np.random.choice([-1,0,1], size = 2, p=[0.45, 0.1, 0.45])
  obstacle_travel = np.random.uniform(0.8, 1.2, size = 2)
  obstacle_angle = np.random.uniform(np.deg2rad(-10), np.deg2rad(10), size = 2)

  obstacle_dynamics = np.concatenate((obstacle_travel, obstacle_direction, obstacle_angle), axis = None)
  
  return np.round(obstacle_dynamics,2)

data_num = int(1e5)            # Estimated scenario number = 1e7/75, with some margin

dataset = np.zeros((data_num, 14))
# Data structure: [drone_pos_x, drone_pos_y, goal_pos_x, goal_pos_y,          0 to 3      drone initial position (x,y); goal initial position (x,y), float
#                  obs1_pos_x, obs1_pos_y, obs2_pos_x, obs2_pos_y,            4 to 7      obstacle initial positions (x,y), float
#                  obs1_travel, obs2_travel,                                  8 to 9      obstacle max travels, in [0.8, 1.2], float
#                  obs1_dir, obs2_dir,                                        10 to 11    obstacle initial directions, in [-1, 0, 1], int
#                  obs1_div, obs2_div]                                        12 to 13    obstacle angle diviations, in [-10, 10] degress, float

for i in range(data_num):
  drone_pos, goal_pos = generate_goal()
  obstacle_poses = generate_obstacle(drone_pos, goal_pos)
  obstacle_dynamics = generate_obstacle_dynamics()

  dataset[i,:] = np.concatenate((drone_pos, goal_pos, obstacle_poses, obstacle_dynamics))

np.save('0_training_dataset.npy', dataset)
