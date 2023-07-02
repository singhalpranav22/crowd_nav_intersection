import abc
import logging
import numpy as np
from numpy.linalg import norm
from ..dynamicPropertiesUtils import *
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import ObservableState, FullState
from .utils import isIntersectionCrowded, isIntersectionCrossing,determineQuadrant,determineSubGoal


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, egx, egy, vx, vy, theta, radius=None, v_pref=None):
        """
         egx, egy -> final goal positions
         gx, gy -> subgoal positions (that can be even the ending goal positions)
         Subgoal heuristic check is applied here.
         """
        self.px = px
        self.py = py
        self.egx = egx
        self.egy = egy
        self.e_goal_quadrant = determineQuadrant(egx,egy)
        current_quad = determineQuadrant(self.px,self.py)
        if isSubGoalHeuristicEnabled():
            if self.e_goal_quadrant == current_quad:
                self.set_goal_position(self.egx,self.egy)
            else:
                subGoal = determineSubGoal(self.px,self.py,self.egx,self.egy)
                self.set_goal_position(subGoal[0],subGoal[1])
        else:
            self.set_goal_position(self.egx, self.egy)
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)



    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_end_goal_position(self):
        return self.egx,self.egy

    def set_goal_position(self,gx,gy):
        self.gx = gx
        self.gy = gy


    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        current_quad = determineQuadrant(self.px,self.py)
        if isSubGoalHeuristicEnabled():
            if self.e_goal_quadrant == current_quad or current_quad==0:
                self.set_goal_position(self.egx,self.egy)
            else:
                subGoal = determineSubGoal(self.px,self.py,self.egx,self.egy)
                self.set_goal_position(subGoal[0],subGoal[1])
        else:
            self.set_goal_position(self.egx, self.egy)
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius



    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
