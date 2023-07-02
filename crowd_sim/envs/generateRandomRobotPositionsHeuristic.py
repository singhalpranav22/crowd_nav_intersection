"""
Script to generate random robot positions within left-right goal heuristic
"""
import time
import random
from .utils.utils import *
from numpy.linalg import norm


def checkIfPreexistingCoordinates(x, y, humanCoordinates):
    for humanCoordinate in humanCoordinates:
        if (getDistance(x, y, humanCoordinate[0], humanCoordinate[1]) <= 1.5) :
            return True
    return False

def generate_random_robot_positions_heuristic(robot_nums, robot_radius, initialHumanPositions):
    robotPos = []
    random.seed(time.time())
    occupied_starting_positions = []
    occupied_goal_positions = []
    for item in initialHumanPositions:
        occupied_starting_positions.append(item[0])
        occupied_goal_positions.append(item[1])

    # get the coordinate range of x and y for each quadrant
    quadrant_ranges = {}
    for i in range(1, 5):
        quadrant_ranges[i] = []
    for i in range(1, 5):
        if i == 1:
            x_range = [2.25, 7.75]
            y_range = [-1.75, 1.75]
        elif i == 2:
            x_range = [-1.75, 1.75]
            y_range = [2.25, 7.75]
        elif i == 3:
            x_range = [-2.25, -7.75]
            y_range = [-1.75, 1.75]
        elif i == 4:
            x_range = [-1.75, 1.75]
            y_range = [-2.25, -7.75]
        quadrant_ranges[i].append(x_range)
        quadrant_ranges[i].append(y_range)

    # choose a random number from 1 to 4 to determine the intersection
    selected_quadrant = random.randint(1, 4)
    goal_quadrant = {1: 4, 2: 1, 3: 2, 4: 3}

    while True and len(robotPos) < robot_nums:
        while True:
            # generate random source and goal positions with the selected quadrant
            xSource = round(
                random.uniform(quadrant_ranges[selected_quadrant][0][0], quadrant_ranges[selected_quadrant][0][1]), 1)
            ySource = round(
                random.uniform(quadrant_ranges[selected_quadrant][1][0], quadrant_ranges[selected_quadrant][1][1]), 1)

            while (checkIfPreexistingCoordinates(xSource, ySource, occupied_starting_positions) or
                   (-7.75 <= xSource <= -1.25 and -7.75 <= ySource <= -1.25) or (
                           -7.75 <= xSource <= -1.25 and 1.25 <= ySource <= 7.75) or (
                           1.25 <= xSource <= 7.75 and -7.75 <= ySource <= -1.25) or (
                           1.25 <= xSource <= 7.75 and 1.25 <= ySource <= 7.75) or
                   (determineQuadrant(xSource, ySource) == 0)):
                xSource = round(
                    random.uniform(quadrant_ranges[selected_quadrant][0][0], quadrant_ranges[selected_quadrant][0][1]),
                    1)
                ySource = round(
                    random.uniform(quadrant_ranges[selected_quadrant][1][0], quadrant_ranges[selected_quadrant][1][1]),
                    1)

            quadrant_for_source = selected_quadrant

            xGoal = round(random.uniform(quadrant_ranges[goal_quadrant[selected_quadrant]][0][0],
                                         quadrant_ranges[goal_quadrant[selected_quadrant]][0][1]), 1)
            yGoal = round(random.uniform(quadrant_ranges[goal_quadrant[selected_quadrant]][1][0],
                                         quadrant_ranges[goal_quadrant[selected_quadrant]][1][1]), 1)

            while (checkIfPreexistingCoordinates(xGoal, yGoal, occupied_goal_positions) or
                   (-7.75 <= xGoal <= -1.25 and -7.75 <= yGoal <= -1.25) or (
                           -7.75 <= xGoal <= -1.25 and 1.25 <= yGoal <= 7.75) or (
                           1.25 <= xGoal <= 7.75 and -7.75 <= yGoal <= -1.25) or (
                           1.25 <= xGoal <= 7.75 and 1.25 <= yGoal <= 7.75) or (
                   (-2 <= xGoal <= 2 and -2 <= yGoal <= 2)) or
                   (quadrant_for_source == determineQuadrant(xGoal, yGoal) or (determineQuadrant(xGoal, yGoal) == 0))):
                xGoal = round(random.uniform(quadrant_ranges[goal_quadrant[selected_quadrant]][0][0],
                                             quadrant_ranges[goal_quadrant[selected_quadrant]][0][1]), 1)
                yGoal = round(random.uniform(quadrant_ranges[goal_quadrant[selected_quadrant]][1][0],
                                             quadrant_ranges[goal_quadrant[selected_quadrant]][1][1]), 1)


            # Checking if the source and goal positions are at same place
            collide = False
            for [(xS, yS), (xG, yG)] in robotPos:
                if norm((xS - xSource, yS - ySource)) < 2 * robot_radius:
                    collide = True
                    break
                if norm((xG - xGoal, yG - yGoal)) < 2 * robot_radius:
                    collide = True
                    break
            if not collide:
                robotPos.append([(xSource, ySource), (xGoal, yGoal)])
                break
            else:
                continue

    return robotPos
