# Joystick-controlled non-domiant hand motor adaptation

## Question

Does adaptation under a perturbation with the non-dominant (left) hand generalize to the dominant (right) hand?

## Experimental design

2 groups:

- Group 1:  No adapt

- Group 2: Adapt

6 blocks:

### **Group 1 (No adapt)**

**Blocks 1-3 -- baseline for both hands**
- Block 1: 30 trials right hand

- Block 2: 30 trials left hand

- Block 3: 100 trials left hand (to match same number of trials in other group)

**Block 4 -- perturbation**
- Block 4: 100 trials left hand with 45º cursor perturbation

**Blocks 5-6 -- Aftereffects**
- Block 5: 50 trials right hand

- Block 6: 50 trials left hand


### **Group 2 (Adapt)**

**Blocks 1-2 -- baseline for both hands**
- Block 1: 30 trials right hand

- Block 2: 30 trials left hand

**Blocks 3-4 -- perturbation**
- Block 3: 100 trials right hand with 45º cursor perturbation

- Block 4: 100 trials left hand with 45º cursor perturbation

**Blocks 5-6 -- Aftereffects**
- Block 5: 50 trials right hand

- Block 6: 50 trials left hand


## Data and trial configs

### Data

- Data is found in the ``data/`` directory. 

- Group 1 will be in ``data/group_1``, group 2 will be in ``data/group_2``.

- In each group directory, participant data will be in each subdirectory named ``firstname_lastname_group``

- In each participant data, there is a csv file for each trial, with the name ``curror1_trial{trial_number}``; trial numbers in data files are 0-indexed.

- Ignore all cursor2_* files.

### Trial configurations

- Each trial file contains the cursor coordinates in the format x,y,timestamp (example: ``-0.032117678225040434,1.9665756412203562E-18,2026-01-07T17:23:05.4160512+00:00``)

- Trial configurations are in ``experiments/experiment_adapt_full.csv``(group 2) and ``experiments/experiment_no_adapt_full.csv`` (group 1)

- Trial config files contain columns ``Trial,TargetX,TargetY,AimX,AimY,Angle1,Angle2,Layer1,Layer2,Instruction``; Trial numbers are 0-indexed. 

- Target coordinates are contained in ``TargetX,TargetY``; the perturbation angle is contained in ``Angle1``.


## Available analysis scripts

### Trajectory visualization scripts

All scripts are located in the ``src/`` directory and can be run using ``uv run python src/<script_name>.py``.

#### 1. Single trial trajectories (``plot_block_trajectories.py``)

Plots superimposed trajectories for a single trial from all participants in each block.

**Output**: 12 plots (6 blocks × 2 groups) saved to ``data/trajectory_plots/``

**Configuration**: Choose which trial to visualize by passing a command-line argument:
- Default (middle trial): ``uv run python src/plot_block_trajectories.py``
- First trial: ``uv run python src/plot_block_trajectories.py first``
- Last trial: ``uv run python src/plot_block_trajectories.py last``
- Specific trial number (e.g., trial 100): ``uv run python src/plot_block_trajectories.py 100``

**Features**:
- Black background matching experimental screen
- All 4 target positions displayed as white circles
- Active target for the trial highlighted in yellow
- Starting position marked in red
- Consistent axes across all plots

#### 2. Mean trajectories (``plot_mean_trajectories.py``)

Computes and plots mean trajectories for each participant to each target within each block.

**Output**: 12 plots (6 blocks × 2 groups) saved to ``data/mean_trajectory_plots/``

**Usage**: ``uv run python src/plot_mean_trajectories.py``

**Features**:
- Averages multiple trials to the same target for each participant
- Normalizes trajectories to 100 points using linear interpolation
- Shows superimposed mean trajectories for all participants and all targets
- Same visual style as single trial plots (black background, consistent axes)

