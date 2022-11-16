import os
import numpy as np
from gym import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary


class FlyThruGateAviary(BaseSingleAgentAviary):
    """Single agent RL problem: fly through a gate."""
    
    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.DYN,
                 freq: int= 10,#240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
                         initial_xyzs=np.array([[0,-1,0.75]]),
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    def _defineBoarder(self):

        super()._defineBoarder()
        self._randomEnds()

        self.goal_indicator = p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../../assets/goal.urdf",
                    self.goal_pos,
                    p.getQuaternionFromEuler([0,0,0]),
                    physicsClientId=self.CLIENT
                    )
        # print("Goal loaded")

        # envSize = 2.0
        # transparency = 0.2

        # board_vis = p.createVisualShapeArray(shapeTypes = [p.GEOM_BOX,p.GEOM_BOX,p.GEOM_BOX,p.GEOM_BOX,p.GEOM_BOX,p.GEOM_BOX],
        #                                  halfExtents = [[envSize,0.02,envSize],[envSize,0.02,envSize],[envSize,0.02,envSize],[envSize,0.02,envSize],[envSize,0.02,envSize],[envSize,0.02,envSize]],
        #                                  rgbaColors = [[0,0,0,transparency],[1,1,1,transparency],[1,1,1,transparency],[1,1,1,transparency],[1,1,1,transparency],[1,1,1,transparency]],
        #                                  visualFramePositions = [[envSize,envSize,envSize],[0,0,envSize],[-envSize,envSize,envSize],[0,2*envSize,envSize],[0,envSize,2*envSize],[0,envSize,0]],
        #                                  visualFrameOrientations = [[0,0,1,1],[0,1,0,1],[0,0,1,1],[0,1,0,1],[1,0,0,1],[1,0,0,1]],
        #                                  physicsClientId=self.CLIENT)
        # board_col = p.createCollisionShapeArray(shapeTypes = [p.GEOM_BOX,p.GEOM_BOX,p.GEOM_BOX,p.GEOM_BOX,p.GEOM_BOX,p.GEOM_BOX],
        #                                  halfExtents = [[envSize,0.02,envSize],[envSize,0.02,envSize],[envSize,0.02,envSize],[envSize,0.02,envSize],[envSize,0.02,envSize],[envSize,0.02,envSize]],
        #                                  collisionFramePositions = [[envSize,envSize,envSize],[0,0,envSize],[-envSize,envSize,envSize],[0,2*envSize,envSize],[0,envSize,2*envSize],[0,envSize,0]],
        #                                  collisionFrameOrientations = [[0,0,1,1],[0,1,0,1],[0,0,1,1],[0,1,0,1],[1,0,0,1],[1,0,0,1]],
        #                                  physicsClientId=self.CLIENT)
        # # print(board_vis, board_col)
        # self.boardid = p.createMultiBody(baseVisualShapeIndex = board_vis,
        #                                  baseCollisionShapeIndex = board_col,
        #                                  basePosition = [0, -0.5*envSize,0.02],
        #                                  physicsClientId=self.CLIENT)
        # collision_board = p.getCollisionShapeData(self.PLANE_ID,-1)

        # print(self.boardid, self.PLANE_ID,self.DRONE_IDS, self.goal_indicator)
        # print("Base plane: ",self.PLANE_ID, p.getCollisionShapeData(self.PLANE_ID,-1))
        # print("Quadrotor: ", self.DRONE_IDS[0], p.getCollisionShapeData(self.DRONE_IDS[0],-1))
        # print("Goal: ", self.goal_indicator, p.getCollisionShapeData(self.goal_indicator,-1))

        # for i in range(len(collision_board)):
        #     print(collision_board[i])
        # exit()
        

    def _randomEnds(self):
        self.goal_pos = [0, 1.0, 0.75]

    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the gate build of cubes and an architrave.

        """
        super()._addObstacles()

        self.with_obs = 0

        if self.with_obs == 1:            
            self.env_obs = p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../../assets/obs.urdf",
                        [0, -1, 0.5],
                        p.getQuaternionFromEuler([0,0,0]),
                        physicsClientId=self.CLIENT
                        )
        
        # for i in range(p.getNumBodies()):
        #     print(p.getBodyInfo(i), p.getBasePositionAndOrientation(i))
        # exit()
    ################################################################################
    
    def _computeObstacleDynamics(self):
        """compute the obstacle dynamics
        """
        self.env_obs_dy = 0
        # if the obstacle is dynamic
        if self.env_obs_dy:
            self.env_obs_pos = [1+np.sin(self.step_counter*self.TIMESTEP), -1, 0.5]
            self.env_obs_ori = [0, 0, 1, 0]

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        norm_ep_time = (self.step_counter/self.SIM_FREQ) / self.EPISODE_LEN_SEC
        # print(self.step_counter)
        # print(state[0:3])
        x_pos = state[0]
        y_pos = state[1]
        # test_collision = p.getContactPoints(bodyA = self.boardid, bodyB = self.env_obs_test)
        # print(test_collision, p.getBasePositionAndOrientation(self.boardid))

        #### Penalty for collision ########
        if self.with_obs == 1:
            obs_collision = p.getContactPoints(bodyA = self.DRONE_IDS, bodyB = self.env_obs)
        else:
            obs_collision = []

        # board_collision = p.getContactPoints(bodyA = self.DRONE_IDS, bodyB = self.boardid)

        if np.abs(x_pos) > 2 or np.abs(y_pos) > 2:
            board_collision = [1]
        else:
            board_collision = []

        if len(obs_collision) != 0 and self.crash != True:
            print("obs crash at ", np.round(state[0:3], 2))
            self.crash = True
            crash_penalty = -1e5
        elif len(board_collision) != 0 and self.crash != True:
            print("crash", np.round(state[0:3], 2)) 
            self.crash = True
            crash_penalty = -1e5
        elif state[2] < 0.2:
            print("fly low at ", state[2])
            self.crash = True
            crash_penalty = -1e5
        else: 
            crash_penalty = 0

        #### Penalty for hazard flight (too close to obstacle/boundary) #######
        # min_dis_board = p.getClosestPoints(bodyA=self.DRONE_IDS, bodyB=self.boardid, distance=10)

        # if self.with_obs == 1:
        #     min_dis_obs = p.getClosestPoints(bodyA=self.DRONE_IDS, bodyB=self.env_obs, distance=10)
        #     min_dis = np.minimum(min_dis_obs[0][8], min_dis_board[0][8])
        # else:
        #     min_dis = min_dis_board[0][8]
        
        min_dis = np.min([np.abs(2-np.abs(x_pos)), np.abs(2-np.abs(y_pos))])

        if min_dis < 0.5:
            # print("Warning: Too close to wall, distance to the closet point is: ",min_dis)
            hazard_penalty = -np.exp(-10*min_dis + 10)
        else:
            hazard_penalty = 0


        #### Penalty for distance to goal #########
        pos_weight = [1,1,1] 
        distance_penalty = -100 * np.linalg.norm((self.goal_pos-state[0:3])*pos_weight)**2

        reward = distance_penalty + crash_penalty + hazard_penalty
        # print(reward)
        # return -10 * np.linalg.norm(np.array([0, -2*norm_ep_time, 0.75])-state[0:3])**2
        return reward
        
    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)

        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            print("time ", np.round(state[0:3], 2))
            return True
        elif self.crash == True:
            # print("crash")
            # self.crash = False
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = 2.5 #MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = 4 #MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi/4 # Full range

        

        if self.OBS_TYPE == ObservationType.XY: 
            clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
            return clipped_pos_xy / MAX_XY
        elif self.OBS_TYPE == ObservationType.KIN: 
            clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
            clipped_pos_z = np.clip(state[2], 0, MAX_Z)
            clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
            clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
            clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

            if self.GUI:
                self._clipAndNormalizeStateWarning(state, #12
                                                clipped_pos_xy,
                                                clipped_pos_z,
                                                clipped_rp,
                                                clipped_vel_xy,
                                                clipped_vel_z  #12
                                                )

            normalized_pos_xy = clipped_pos_xy / MAX_XY
            normalized_pos_z = clipped_pos_z / MAX_Z
            normalized_rp = clipped_rp / MAX_PITCH_ROLL
            normalized_y = state[9] / np.pi # No reason to clip
            normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
            normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
            normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

            norm_and_clipped = np.hstack([normalized_pos_xy,
                                        normalized_pos_z,
                                        state[3:7],
                                        normalized_rp,
                                        normalized_y,
                                        normalized_vel_xy,
                                        normalized_vel_z,
                                        normalized_ang_vel,
                                        state[16:20]
                                        ]).reshape(20,)

            return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        # if not(clipped_pos_xy == np.array(state[0:2])).all():
        #     print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        # if not(clipped_pos_z == np.array(state[2])).all():
        #     print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        # if not(clipped_rp == np.array(state[7:9])).all():
        #     print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        # if not(clipped_vel_xy == np.array(state[10:12])).all():
        #     print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        # if not(clipped_vel_z == np.array(state[12])).all():
        #     print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
