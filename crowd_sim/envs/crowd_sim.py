import re
import gym
import math
import rvo2
import logging
import csv, os
import threading
import numpy as np
from turtle import position
from numpy.linalg import norm
from matplotlib import patches
import matplotlib.lines as mlines
from .utils.info import ReachSubgoal
from .dynamicPropertiesUtils import *
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from .CrowdSimConfigHolder import config_holder
from crowd_sim.envs.utils.utils import point_to_segment_dist
from .getRobotPositionFromCsv import getRobotPositionFromCsv
from .generateRandomPositions import generateRandomPositions
from .getHumansPositionsFromCsv import getHumanPositionsFromCsv
from .generateRandomRobotPositions import generateRandomRobotPositions
from .generateRandomRobotPositionsHeuristic import generate_random_robot_positions_heuristic
from .utils.utils import isIntersectionCrowded, isIntersectionCrossing, addRandomNoise, getDistance, isNearIntersetion, \
    shouldRobotStopHeuristic, is_near_wall


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.human_radius = None
        self.robot_radius = None
        self.robot_num = 1
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.data = []

        # Variable to store initial human and robot positions
        self.initialHumanPositions = []
        self.initialRobotPosition = []

        # heuristics configuration data
        self.intersection_crowd_threshold = 4
        self.robot_velocity_limit = 2

        self.inflatedRadius = None
        # read string from file configs/csvLocation.txt
        with open('configs/csvLocation.txt', 'r') as file:
            self.csvFilePath = file.read().replace('\n', '')
            if (self.csvFilePath == ""):
                self.configFromCSV = False
            else:
                self.configFromCSV = True
        self.heuristicsMap = {'subgoal': False, 'stopIfIntersectionCrowded': False, 'closeToBoundary': False,
                              'velocityControl': False, 'leftRightGoal': False}
        config_holder.set_parameter('heuristicsMap', self.heuristicsMap)
        self.data_gen_type = 'mixed'

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.subgoal_velocity_dirn_factor = config.getfloat('reward', 'subgoal_velocity_dirn_factor')
        self.near_wall_reward = config.getfloat('reward', 'near_wall_reward')
        self.subgoal_reached = config.getfloat('reward', 'subgoal_reached')
        self.robot_radius = config.getfloat('robot', 'radius')
        self.human_radius = config.getfloat('humans', 'radius')
        self.inflatedRadius = self.human_radius * 2
        config_holder.set_parameter('config', self.config)
        self.data_gen_type = config.get('data_gen_type', 'data_gen_type')
        self.heuristicsMap = {'subgoal': config.getboolean('heuristics', 'subgoal'),
                              'stopifintersectioncrowded': config.getboolean('heuristics', 'stopifintersectioncrowded'),
                              'closetoboundary': config.getboolean('heuristics', 'closetoboundary'),
                              'velocitycontrol': config.getboolean('heuristics', 'velocitycontrol'),
                              'leftrightgoal': config.getboolean('heuristics', 'leftrightgoal'),
                              'robotStopHeuristicInCrowd': config.getboolean('heuristics', 'robotStopHeuristicInCrowd')}
        config_holder.set_parameter('heuristicsMap', self.heuristicsMap)
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
            self.intersection_crowd_threshold = config.getint('sim', 'intersection_crowd_threshold')
            self.robot_velocity_limit = config.getfloat('sim', 'robot_velocity_limit')
            # getting config from csv
            if (self.configFromCSV):
                with open(self.csvFilePath, 'r') as csvFile:
                    csvReader = csv.reader(csvFile)
                    i = 0
                    for row in csvReader:
                        if (i == 0):
                            self.human_num = len(row) - 2
                            break
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def sendInitialPositions(self):
        return self.initialRobotPosition, self.initialHumanPositions

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """

        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':

            human = Human(self.config, 'humans')
            humanPos = []
            # If data is coming from the csv, use the csv to get the data
            if (self.configFromCSV):
                humanPos = getHumanPositionsFromCsv(self.csvFilePath, self.human_num)
            else:
                humanPos = generateRandomPositions(self.human_num, human.radius)

            #### Uncomment, if you want to hardcode the initial positions
            # humanPos =  [[(5.9, 0.4), (3.3, 0.4)], [(-1.3, 2.5), (-0.4, -5.1)], [(0.8, -2.6), (-4.9, -0.5)]]
            # humanPos = [[(0, 4),(-2.9, -1.4)], [(1.2, -5), (1.3, 5.5)], [(7, 1.5), (-1.3, -4)]]

            self.initialHumanPositions = humanPos
            self.humans = []
            for i in range(human_num):
                human = Human(self.config, 'humans')
                [(px, py), (gx, gy)] = humanPos[i]
                # Uncomment below lines if you want to add some noise to the initial and goal positions
                # (px, py) = addRandomNoise(px, py, 0.2)
                # (gx, gy) = addRandomNoise(gx, gy, 0.2)

                human.set(px, py, gx, gy, 0, 0, 0)
                self.humans.append(human)

            """
             Using static humans as a wall for our intersection scenario.
             They, help in making a static obstacle at a fixed position.
             There are 8 lines on which humans are placed - on the walls for our 8x8 intersection area.
             Intersection lies for the area: -2 <= x <= 2 and -2 <=y <= 2
            """
            curr = -7.5
            while curr <= -2.5:
                human1 = Human(self.config, 'humans')
                human1.radius = 0.35
                human1.v_pref = 0
                human1.set(curr, 2, curr, 2, 0, 0, 0)
                self.humans.append(human1)
                human2 = Human(self.config, 'humans')
                human2.radius = 0.35
                human2.v_pref = 0
                human2.set(curr, -2, curr, -2, 0, 0, 0)
                self.humans.append(human2)
                human3 = Human(self.config, 'humans')
                human3.radius = 0.35
                human3.v_pref = 0
                human3.set(-2, curr, -2, curr, 0, 0, 0)
                self.humans.append(human3)
                human4 = Human(self.config, 'humans')
                human4.radius = 0.35
                human4.v_pref = 0
                human4.set(2, curr, 2, curr, 0, 0, 0)
                self.humans.append(human4)
                curr2 = curr * -1
                human5 = Human(self.config, 'humans')
                human5.radius = 0.35
                human5.v_pref = 0
                human5.set(curr2, -2, curr2, -2, 0, 0, 0)
                self.humans.append(human5)
                human6 = Human(self.config, 'humans')
                human6.radius = 0.35
                human6.v_pref = 0
                human6.set(curr2, 2, curr2, 2, 0, 0, 0)
                self.humans.append(human6)
                human7 = Human(self.config, 'humans')
                human7.radius = 0.35
                human7.v_pref = 0
                human7.set(2, curr2, 2, curr2, 0, 0, 0)
                self.humans.append(human7)
                human8 = Human(self.config, 'humans')
                human8.radius = 0.35
                human8.v_pref = 0
                human8.set(-2, curr2, -2, curr2, 0, 0, 0)
                self.humans.append(human8)
                curr += 0.8

        elif rule == 'mixed':
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((
                                    px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                ### Position and goals array for humans
                humanPosHc = [[(-7.5, -1), (7.5, -1)], [(0, 7.5), (0, -7.5)], [(-7.5, 1), (7.5, 1)],
                              [(-1, 7.5), (1, 7.5)]]
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    [(px, py), (gx, gy)] = humanPosHc[i]
                    human.set(px, py, gx, gy, 0, 0, 0)
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.processObstacles()
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

            robotPos = []
            if (self.configFromCSV):
                robotPos = getRobotPositionFromCsv(self.csvFilePath)
            else:
                if isLeftRightGoalHeuristicEnabled():
                    robotPos = generate_random_robot_positions_heuristic(1, self.robot_radius,
                                                                         self.initialHumanPositions)
                else:
                    robotPos = generateRandomRobotPositions(1, self.robot_radius, self.initialHumanPositions)
            self.initialRobotPosition = robotPos
            ######## Uncomment below lines if you want to hardcode the robot positions
            # robotPos = [(-6, 0), (-1, 3)]
            # robotPos = [[(-1.2, 6.1), (5.2, 1.2)]]

            ######## Uncomment below lines, if you want to add some noise to the robot positions
            # robotPos[0] = addRandomNoise(robotPos[0][0], robotPos[0][1], 0.2)
            # robotPos[1] = addRandomNoise(robotPos[1][0], robotPos[1][1], 0.2)

            self.robot.set(robotPos[0][0][0], robotPos[0][0][1], robotPos[0][1][0], robotPos[0][1][1], 0, 0, np.pi / 2)

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        # data to be stored in csv later, for the dataset generation part
        row = [self.global_time]
        robotState = self.robot.get_full_state()
        row.append(robotState.toDictionary())
        for human in self.humans:
            humanState = human.get_full_state()
            row.append(humanState.toDictionary())
        self.data.append(row)
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        # For robot, traverse to each human and check whether there it lies in the visibility of robot or not
        for i in range(self.human_num):
            currentAgent = self.humans[i]
            d = getDistance(self.robot.px, self.robot.py, currentAgent.px, currentAgent.py)
            # find tan inverse of angle between robot and human
            theta = np.arctan2(currentAgent.py - self.robot.py, currentAgent.px - self.robot.px)
            # find sin inverse between robot and human
            alpha = 0
            if d >= self.inflatedRadius:
                alpha = np.arcsin(self.inflatedRadius / d)
            d1 = d * np.cos(alpha)
            # q1 is one of the tangent points on the circle
            q1_x = self.robot.px + d1 * np.cos(theta + alpha)
            q1_y = self.robot.py + d1 * np.sin(theta + alpha)
            # q2 is the other point tangent on the circle
            q2_x = self.robot.px + d1 * np.cos(theta - alpha)
            q2_y = self.robot.py + d1 * np.sin(theta - alpha)
            # line can be found using y = mx + c where m is the slope and c is the y intercept
            # equation of line between robot and q1
            m1 = (q1_y - self.robot.py) / (q1_x - self.robot.px)
            c1 = self.robot.py - m1 * self.robot.px
            # equation of line between robot and q2
            m2 = (q2_y - self.robot.py) / (q2_x - self.robot.px)
            c2 = self.robot.py - m2 * self.robot.px
            # Line 1: y = m1x + c1
            # Line 2: y = m2x + c2

        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_end_goal_position())) < self.robot.radius
        reaching_subgoal = True
        isCsvRequired = True

        # calculate angle in radians between velocity and current position to goal vector
        px = self.robot.px
        py = self.robot.py
        vx = self.robot.vx
        vy = self.robot.vy
        gx = self.robot.gx
        gy = self.robot.gy
        egx = self.robot.egx
        egy = self.robot.egy

        angle = np.arctan2(gy - py, gx - px) - np.arctan2(vy, vx)

        reward = (np.cos(angle) * self.subgoal_velocity_dirn_factor * norm((vx, vy)) - 0.1) * self.time_step * 0.1
        if isSubGoalHeuristicEnabled():
            if math.isclose(gx, egx) and math.isclose(gy, egy):
                reaching_subgoal = False
            else:
                reaching_subgoal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        isSuccess = False
        if isSubGoalHeuristicEnabled() and reaching_subgoal:
            reward += self.subgoal_reached
        if isCloseToBoundaryHeuristicEnabled():
            if is_near_wall(px, py, 0.5):
                reward += self.near_wall_reward

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif px < -8 or px > 8 or py < -8 or py > 8:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
            isSuccess = True
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            done = False
            info = Nothing()

        isCsvRequired = False
        if self.data_gen_type == 'mixed':
            isCsvRequired = True
        if self.data_gen_type == 'success' and isSuccess == True:
            isCsvRequired = True
        if self.data_gen_type == 'failure' and isSuccess == False:
            isCsvRequired = True

        if isCsvRequired and self.configFromCSV != True and done == True:
            header = ['time']
            for i in range(self.robot_num):
                header.append(f'robot{i + 1}')
            for i in range(self.human_num):
                header.append(f'human{i + 1}')
            self.writer = None
            files = os.listdir('testcases')
            lastFileNum = 0
            for file in files:
                if len(file) > 5:
                    match = re.search(r'\d+(?=\.csv)', file)
                    if match:
                        currFileNum = int(match.group())
                    else:
                        currFileNum = 0
                    lastFileNum = max(lastFileNum, currFileNum)
            lastFileNum += 1
            self.csvFileName = f'testcases/testcase{lastFileNum}.csv'
            print("GENERATED TESTCASE NUMBER: ", self.csvFileName)
            with open(f'testcases/testcase{lastFileNum}.csv', 'w', encoding='UTF8') as f:
                self.writer = csv.writer(f)
                self.writer.writerow(header)
                for (i, row) in enumerate(self.data):
                    self.writer.writerow(row[0:self.human_num + 2])

        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            # update all agents
            self.robot.step(action)
            if shouldRobotStopHeuristic(self.robot, self.humans) and isRobotStopHeuristicInCrowd():
                self.robot.vx = 0
                self.robot.vy = 0
            else:
                self.robot.step(action)

            intersectionCrowded = isIntersectionCrowded(self.humans, [self.robot])
            numHumansInIntersection = 0
            if intersectionCrowded and isStopIfIntersectionCrowdedHeuristicEnabled():
                for human in self.humans:
                    isCrossing = isIntersectionCrossing(human)
                    if isCrossing:
                        numHumansInIntersection = numHumansInIntersection + 1
                        pass

            if isNearIntersetion(self.robot.px, self.robot.py):
                if numHumansInIntersection >= self.intersection_crowd_threshold:
                    robot.vx = 0
                    robot.vy = 0

            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                if i >= self.human_num:
                    break
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info

    def render(self, mode='human', output_file=None):
        # pass
        from matplotlib import animation
        import matplotlib.pyplot as plt
        from matplotlib import collections as mc
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'blue'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(8, 8))
            lines = [[(-10, -2), (-2, -2)], [(-10, 2), (-2, 2)], [(-2, -2), (-2, -10)], [(2, -10), (2, -2)],
                     [(2, -2), (10, -2)], [(2, 2), (10, 2)], [(2, 2), (2, 10)], [(-2, 2), (-2, 10)]]
            c = np.array(
                [(1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1),
                 (1, 0, 0, 1)])
            lc = mc.LineCollection(lines, colors=c, linewidths=2)
            ax.tick_params(labelsize=16)
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            ax.add_collection(lc)
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.egx], [self.robot.egy], color=goal_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')

            for human in self.humans:
                ax.add_patch(plt.Circle((human.gx, human.gy), 0.2, edgecolor="black", linewidth=3, ls='-'))
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]

            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            # if self.attention_weights is not None:
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                               agent_state.py + radius * np.sin(
                                                                                   theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    # if self.attention_weights is not None:
                    #     human.set_color(str(self.attention_weights[frame_num][i]))
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid(axis='x', color='r', linestyle='-', linewidth=2)
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
