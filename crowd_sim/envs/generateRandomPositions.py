""" 
Script to generate random human positions within the constraints
"""
import time
import random
from numpy.linalg import norm

def generateRandomPositions(human_nums,human_radius):
    """
    Returns in this format:  humanPosHc = [[(-6,0),(7,-1.25)],[(1,-7),(-1.25,7.5)],[(4.5,0),(-1.25,-4)]]
    """
    humanPos = []
    random.seed(time.time())
    while True and len(humanPos)<human_nums:
        while True:
            # generate random source and goal positions
            xSource = round(random.uniform(-7.25,7.25), 1)
            ySource = round(random.uniform(-7.25,7.25), 1)
            while ((-7.25<=xSource<=-1.25 and -7.25<=ySource<=-1.25) or (-7.25<=xSource<=-1.25 and 1.25<=ySource<=7.25) or (1.25<=xSource<=7.25 and -7.25<=ySource<=-1.25) or (1.25<=xSource<=7.25 and 1.25<=ySource<=7.25)):
                xSource = round(random.uniform(-7.25,7.25), 1)
                ySource = round(random.uniform(-7.25,7.25), 1)
            xGoal = round(random.uniform(-7.25,7.25), 1)
            yGoal = round(random.uniform(-7.25,7.25), 1)
            while ((-7.25<=xGoal<=-1.25 and -7.25<=yGoal<=-1.25) or (-7.25<=xGoal<=-1.25 and 1.25<=yGoal<=7.25) or (1.25<=xGoal<=7.25 and -7.25<=yGoal<=-1.25) or (1.25<=xGoal<=7.25 and 1.25<=yGoal<=7.25) or (-2<=xGoal<=2 and -2<=yGoal<=2)):
                xGoal = round(random.uniform(-7.25,7.25), 1)
                yGoal = round(random.uniform(-7.25,7.25), 1)
            collide = False
            for [(xS,yS),(xG,yG)] in humanPos:
                 if norm((xS - xSource, yS - ySource )) < 2*human_radius:
                        collide = True
                        break
                 if norm((xG - xGoal, yG - yGoal )) < 2*human_radius:
                        collide = True
                        break
            if not collide:
                humanPos.append([(xSource,ySource),(xGoal,yGoal)])
                break
            else:
                continue
    return humanPos
                
        
