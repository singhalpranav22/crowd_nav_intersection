"""
Script to get robot positions from the csv path and human numbers
"""

import csv
import json
def getRobotPositionFromCsv(csvFilePath):
    with open(csvFilePath, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        i = 0
        for row in csvReader:
            if(i==1):
                robotObject = json.loads(row[1])
                px = robotObject['px']
                py = robotObject['py']
                gx = robotObject['gx']
                gy = robotObject['gy']
                return [[(px,py),(gx,gy)]]
            i += 1
            if(i>=2):
                break
    return []