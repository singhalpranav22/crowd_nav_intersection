"""
gui utility that manages the data generation and visualisation process
"""

import os
import cv2
import numpy as np
import pandas as pd
import configparser
import PySimpleGUI as sg

configFilename = 'configs/env.config'
data_options = ['success', 'failure', 'mixed']
heuristics_labels = ['subgoal', 'stopifintersectioncrowded', 'closetoBoundary', 'velocitycontrol', 'leftrightgoal', 'robotstopheuristicincrowd']
# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the file
config.read(configFilename)
choices = []
# column1
col1 = [
    [sg.T("Visualise test cases from a valid csv data file:", font=('Arial', 20), text_color='orange')],
    [sg.Text('Select csv file to run:'), sg.In(size=(25,1), enable_events=True ,key='-FILE-'), sg.FileBrowse()],
    [sg.Text('CSV file headers:', font=('Arial', 14), text_color='black')],
    [sg.Listbox(choices, size=(40, len(choices)), key='-FILE_NAME-', font=('Arial', 12))],
    [sg.Button('Start visualisation from csv config', key='-START_VISUALISATION-', size=(25,1), font=('Arial', 12))]
]

# column2
col2 = [
    [sg.T("Generate test cases:", font=('Arial', 20), text_color='orange')],
    [sg.Text('Enter number of iterations:', font=('Arial', 14), text_color='black')],
    [sg.InputText(key='-ITERATIONS-', font=('Arial', 12))],
    [sg.Text('Enter number of humans:', font=('Arial', 14), text_color='black')],
    [sg.InputText(key='-HUMANS-', font=('Arial', 12))],
    [sg.Text('Enter time step:', font=('Arial', 14), text_color='black')],
    [sg.InputText(key='-TIME_STEP-', font=('Arial', 12))],
    [sg.Text('Time limit:', font=('Arial', 14), text_color='black')],
    [sg.InputText(key='-TIME_LIMIT-', font=('Arial', 12))],
    [sg.Text('Select heuristics for the dataset generation:', font=('Arial', 17), text_color='blue')],
]

for label in heuristics_labels:
    col2.append([sg.Checkbox(label, key=label, font=('Arial', 12))])

col2.append([sg.Text('Select data type:', font=('Arial', 14), text_color='black')])
col2.append([sg.Spin(data_options, initial_value=data_options[0], size=(20, 5), key='-DATA_TYPE_SPINNER-', font=('Arial', 12))])
col2.append([sg.Button('Start generating dataset', key='-START_CSV_GEN-', size=(25,1), font=('Arial', 12)), sg.Button('Exit', key='-EXIT-', size=(10,1), font=('Arial', 12))])

# final layout
layout = [
    [sg.Column(col1),
     sg.VSeperator(pad=(0, 0)),
     sg.Column(col2)
    ]
]

# Set a theme for the GUI
sg.theme('DarkTeal2')

# creating the window
window = sg.Window('Data Config Gui', layout, auto_size_text=True,
                   auto_size_buttons=True, resizable=True, grab_anywhere=False, border_depth=5,
                   default_element_size=(15, 1), finalize=True, element_justification='l')

# GUI event loop
while True:
    event, values = window.read()
    # if the event is to close the window
    if event in (sg.WIN_CLOSED, '-EXIT-'):
        break
    # if the event is to browse the folder
    if event == '-FILE-':
        # find the path of file selected
        file = values['-FILE-']
        # check if file is csv
        if file.endswith(('.csv')):
            # if yes then read the csv file
            df = pd.read_csv(file)
            # get the column names
            columns = df.columns
            # get the column names as list
            columns = columns.tolist()
            window['-FILE_NAME-'].update(values=columns)
        pass
    # if the event is to start the watermarking
    if event == '-START_CSV_GEN-':
        selectedHeuristics = []
        if 'heuristics' in config:
            heuristics_section = config['heuristics']
            for heuristic in heuristics_labels:
                heuristics_section[heuristic] = 'false'
                if values[heuristic]:
                    heuristics_section[heuristic] = 'true'
        if 'env' in config:
            env_section = config['env']
            if values['-TIME_STEP-']:
                env_section['time_step'] = values['-TIME_STEP-']
            if values['-TIME_LIMIT-']:
                env_section['time_limit'] = values['-TIME_LIMIT-']

        if 'sim' in config:
            sim_section = config['sim']
            if values['-HUMANS-']:
                sim_section['human_num'] = values['-HUMANS-']
        if 'data_gen_type' in config:
            data_type_section = config['data_gen_type']
            if values['-DATA_TYPE_SPINNER-']:
                data_type_section['data_gen_type'] = values['-DATA_TYPE_SPINNER-']

        with open(configFilename, 'w') as file:
            config.write(file)
        iterations = 0
        if values['-ITERATIONS-']:
            iterations = values['-ITERATIONS-']
            iterations = int(iterations)
        for i in range(iterations):
            os.system("python3 test.py --policy orca --phase test --test_case 0")
        sg.popup('Done')
    if event == '-START_VISUALISATION-':
        # open the file
        file = values['-FILE-']
        # check if file is csv
        if file.endswith(('.csv')) == False:
            sg.popup('Please select a csv file')
            pass
        # if yes then write the file in the text file current directory+configs/csvLocation.csv
        with open('configs/csvLocation.txt', 'w') as f:
            f.write(file)
        # run the visualisation
        os.system("python3 test.py --policy orca --phase test --visualize --test_case 0")
        # empty the contents of the file
        with open('configs/csvLocation.txt', 'w') as f:
            f.write('')
        sg.popup('Done')

window.close()
