import os
import csv

test_cases = 100
fields = ['S.no.', 'k', 'Success Rate', 'Collision Rate', 'Nav Time', 'Total Reward', 'Freq. Danger',
          'Avg. Min. Separation dist in Danger', 'Configuration of Robot', 'Configuration of humans']

# rows.append(row)
filename = "results.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the data rows
    csvwriter.writerow(fields)

for i in range(test_cases):
    os.system("python3 test.py --policy orca --phase test")