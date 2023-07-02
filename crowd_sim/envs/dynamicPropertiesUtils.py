from .CrowdSimConfigHolder import config_holder


def isSubGoalHeuristicEnabled():
    heuristics_map = config_holder.get_parameter('heuristicsMap')
    return heuristics_map.get('subgoal')


def isStopIfIntersectionCrowdedHeuristicEnabled():
    heuristics_map = config_holder.get_parameter('heuristicsMap')
    return heuristics_map.get('stopifintersectioncrowded')


def isCloseToBoundaryHeuristicEnabled():
    heuristics_map = config_holder.get_parameter('heuristicsMap')
    return heuristics_map.get('closetoboundary')


def isVelocityControlHeuristicEnabled():
    heuristics_map = config_holder.get_parameter('heuristicsMap')
    return heuristics_map.get('velocityControl')


def isLeftRightGoalHeuristicEnabled():
    heuristics_map = config_holder.get_parameter('heuristicsMap')
    return heuristics_map.get('leftrightgoal')


def isRobotStopHeuristicInCrowd():
    heuristics_map = config_holder.get_parameter('heuristicsMap')
    return heuristics_map.get('robotstopheuristicincrowd')