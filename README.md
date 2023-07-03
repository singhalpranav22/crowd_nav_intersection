## Enhancing Crowd Navigation with Heuristics and RL: A Dataset Generation and Training Framework
### Code Guide
#### Video Guides 
How to use? -> https://www.youtube.com/watch?v=Alr0t8d-a-w <br>
Code walkthrough -> https://youtu.be/tCL5uadcabc
## INTRODUCTION
Welcome to the Crowd Navigation Project! In this project, we have extended the capabilities of the existing crowd_nav system to handle complex navigation scenarios involving multiple humans and a single robot. Our goal is to create a dataset for training crowd navigation models, incorporating various heuristics and parameters through a user-friendly graphical user interface (GUI). 
The primary objective of this project is to generate a dataset that captures diverse navigation scenarios, allowing for robust training of crowd navigation models. To achieve this, we have developed a GUI utility that simplifies the process of configuring heuristics and parameters for dataset generation. The GUI allows you to specify the number of humans, time step, time limit, and various heuristics such as subgoal, stop if intersection crowded, close to boundary, velocity control, left-right goal, and robot stop heuristic in crowd.
By adjusting these settings, you can create customized navigation scenarios that reflect real-world complexities. Furthermore, our project integrates state-of-the-art reinforcement learning (RL) models to train the crowd navigation model within an intersection. By combining the generated dataset with these advanced RL techniques, we aim to improve the navigation capabilities of the robot and ensure safe and efficient interactions with human agents. The user-friendly GUI empowers you to easily configure the dataset generation process and visualize the test cases from a valid CSV data file. 
Whether you're a researcher, developer, or enthusiast in the field of crowd navigation, our project provides a powerful tool to explore and enhance the capabilities of navigation systems. We invite you to join us on this exciting journey as we delve into the complexities of crowd navigation, leverage heuristics, and train advanced models to tackle real-world navigation challenges in intersections. Let's make navigation safer and more efficient for both humans and robots!

## INSTALLATION

To install and set up the Crowd Navigation project, please follow the steps below:

Clone the repository:

Visit the GitHub repository for the Crowd Navigation project: Crowd Navigation Repository
Clone the repository to your local machine using the command: 
```git clone https://github.com/singhalpranav22/crowd_nav_intersection.git```

Create a new conda environment or virtual environment (optional):

Open a command line or terminal.

Navigate to the root directory of the cloned repository using the cd command.

```cd crowd-navigation```


Create a new conda environment using the command: 
```conda create --name crowd_navigation_env```


Activate the environment by running: 
```conda activate crowd_navigation_env```


Install the project requirements:

Run the following command to install the required dependencies: pip install -e .
Install the Python-Rvo2 library:

Visit the Python-Rvo2 library repository on GitHub: 
Follow the installation instructions provided in the repository to install the library.
Mac-specific Configuration (if applicable):

If you are using macOS, you may need to set the MACOSX_DEPLOYMENT_TARGET environment variable. Replace 10.xx with the current macOS version on your system.
Run the following command to export the environment variable: 
```export MACOSX_DEPLOYMENT_TARGET=10.xx```



Now, its time to install the framework's dependencies, just on the root folder, run the following command:


``` pip install -e .```


This command would install all the requirements for the project.

Once you have completed these steps, you have successfully installed the Crowd Navigation project and its dependencies. You can now explore the functionalities of the project, including generating datasets, configuring heuristics, and training crowd navigation models.

To run the project, ensure that you are in the root directory of the installed repository. From there, you can execute the necessary commands mentioned in the previous instructions.

## USAGE:
To use the GUI for the Crowd Navigation project, please follow the steps below:

Navigate to the crowd_nav directory:

Open a command line or terminal.

Change the current directory to the crowd_nav directory of the cloned repository using the following command:

```cd crowd_nav```

Run the GUI:

Execute the following command to start the GUI:

```python gui.py```
### Visualizing a CSV Dataset:

In the first pane, you can select a compatible CSV dataset for visualization.
Click the "Select csv file to run" button and browse your file system to locate and select the desired CSV file.
The GUI will display the headers of the selected CSV file in the "CSV file headers" listbox.
Click the "Start visualization from csv config" button to initiate the visualization process.
Generating a Dataset:


In the second pane, you can generate a dataset by configuring various parameters and heuristics.

Enter the desired number of iterations, number of humans, time step, and time limit in their respective input fields.
The "Select heuristics for the dataset generation" section allows you to choose specific heuristics for dataset generation.
Tick the checkboxes corresponding to the desired heuristics.
Select the data type from the dropdown list of "Select data type".
Once you have configured the parameters and heuristics, click the "Start generating dataset" button to initiate the dataset generation process.
Note: The generated dataset will be saved in the configured file location according to the settings in the env.config file.

Exiting the GUI:

To exit the GUI, either close the GUI window or click the "Exit" button.
By following these instructions, you can effectively use the GUI to visualize CSV datasets and generate custom datasets for training and testing crowd navigation models.e GUI:

### Contributors
1. Pranav Singhal - IIT2019050
2. Mrinal Bhave - IIT2019152
