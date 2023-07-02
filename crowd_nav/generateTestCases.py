### Script to generate test caases data
import os

iterations = 50
# Non-visualization test cases generation for 'iterations' -> parameter
for i in range(iterations):
    os.system("python test.py --policy orca --phase test --test_case 0")
