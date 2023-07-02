"""
Script to get human positions from the csv path and human numbers
"""

import csv
import json
def getHumanPositionsFromCsv(csvFilePath,human_num):
    humanPositions = []
    with open(csvFilePath, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        i = 0
        for row in csvReader:
            if(i==1):
                for j in range(2,human_num+2):
                    # parse json
                    humanObj = json.loads(row[j])
                    px = humanObj['px']
                    py = humanObj['py']
                    gx = humanObj['gx']
                    gy = humanObj['gy']
                    humanPositions.append([(px,py),(gx,gy)])
            i += 1
            if(i>=2):
                break
    return humanPositions