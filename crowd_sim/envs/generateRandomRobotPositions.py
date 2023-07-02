""" 
Script to generate random human positions within the constraints
"""
import time
import random
from numpy.linalg import norm
from .utils.utils import determineQuadrant
from .utils.utils import determineQuadrant, getDistance


def checkIfPreexistingCoordinates(x, y, humanCoordinates):
    for humanCoordinate in humanCoordinates:
        if (getDistance(x, y, humanCoordinate[0], humanCoordinate[1]) <= 1.5) :
            return True
    return False

def generateRandomRobotPositions(robot_nums, robot_radius, initialHumanPositions):
    """
    Returns in this format:  robotPosHc = [[(-6,0),(7,-1.25)],[(1,-7),(-1.25,7.5)],[(4.5,0),(-1.25,-4)]]
    While making sure that the position doesn't go out of bounds
    """
    robotPos = []
    random.seed(time.time())

    checkIfPreexistingCoordinates(2.5, -3.5, [])

    # Filter out the starting and goal positions
    starting_positions = []
    goal_positions = []
    for item in initialHumanPositions:
        starting_positions.append(item[0])
        goal_positions.append(item[1])

    while True and len(robotPos) < robot_nums:
        while True:
            # generate random source and goal positions
            xSource = round(random.uniform(-7.3, 7.3), 1)
            ySource = round(random.uniform(-7.3, 7.3), 1)

            while (checkIfPreexistingCoordinates(xSource, ySource, starting_positions) or
                    (-7.75 <= xSource <= -1.25 and -7.75 <= ySource <= -1.25) or (
                    -7.75 <= xSource <= -1.25 and 1.25 <= ySource <= 7.75) or (
                           1.25 <= xSource <= 7.75 and -7.75 <= ySource <= -1.25) or (
                           1.25 <= xSource <= 7.75 and 1.25 <= ySource <= 7.75) or
                    (determineQuadrant(xSource, ySource) == 0)):
                xSource = round(random.uniform(-7.3, 7.3), 1)
                ySource = round(random.uniform(-7.3, 7.3), 1)

            quadrant_for_source = determineQuadrant(xSource, ySource)

            xGoal = round(random.uniform(-7.3, 7.3), 1)
            yGoal = round(random.uniform(-7.3, 7.3), 1)

            while (checkIfPreexistingCoordinates(xGoal, yGoal, goal_positions) or
                (-8.75 <= xGoal <= -1.25 and -8.75 <= yGoal <= -1.25) or (-8.75 <= xGoal <= -1.25 and 1.25 <= yGoal <= 8.75) or (
                    1.25 <= xGoal <= 8.75 and -8.75 <= yGoal <= -1.25) or (1.25 <= xGoal <= 8.75 and 1.25 <= yGoal <= 8.75) or ((-2.5<=xGoal<=2.5 and -2.5<=yGoal<=2.5)) or
                   (quadrant_for_source == determineQuadrant(xGoal, yGoal) or (determineQuadrant(xGoal, yGoal) == 0))):
                xGoal = round(random.uniform(-7.3, 7.3), 1)
                yGoal = round(random.uniform(-7.3, 7.3), 1)

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
