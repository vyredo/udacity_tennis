# Tennis - Reinforcement Learning Project

This repository contains the solution to the Udacity Nanodegree project for the Tennis Navigation environment. The agent is trained using a MADDPGAgent (Multi Agent Deep Deterministic Policy Gradient ) with Replay Buffer, The example provide is for linux OS, you will need to include another file for another OS.

---

## Report

Check this [link to view Report section](https://github.com/vyredo/udacity_tennis/blob/main/Report/Report.md)

## Getting Started

Follow the steps below to run the project and train the agent:

1. clone this repository with: `git clone`
2. Open the Jupyter notebook: `jupyter notebook Main.ipynb`

3. Configure your Python environment:  
   Update the notebook with the correct paths to your Python binaries and library locations. for example Replace `[USERNAME]` with your system username in the snippet below:

```python
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/home/[USERNAME]/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/home/[USERNAME]/mambaforge/envs/py310/lib/python3.10/site-packages"
```

3. Run all the cells in the notebook to start training the agent.

---

### Project Details

#### Key Features

- Algorithm: MADDPG Agent with Replay Buffer for efficient learning.

- Training is done several times to allow me to debug each training, the first training is done in 1250 episodes, this the plot for first 1250 episodes
  <img src="https://github.com/vyredo/udacity_tennis/blob/main/Report/scores_plot_prev_1250.png" />
- The second training's plot
  <img src="https://github.com/vyredo/udacity_tennis/blob/main/Report/scores_plot.png" />

- Pretrained Model: The trained model is saved at:

```bash
Report/agent1_actor.pth
Report/agent1_critic.pth
Report/agent2_actor.pth
Report/agent2_critic.pth
```

- Logs: Training logs are available at:

```bash
Report/scores_logs.txt
```

---

### Training Progress

Below are the training results from the episode 100 to 1200. The agent achive highest score of 1.5 in these trainings :

```mathematica

Episode 100: 0.0
Episode 200: 0.0
Episode 300: 0.0
Episode 400: 0.10000000149011612
Episode 500: 0.0
Episode 600: 1.1000000163912773
Episode 700: 0.20000000298023224
Episode 800: 0.0
Episode 900: 1.5000000223517418
Episode 1000: 0.0
Episode 1100: 0.10000000149011612
Episode 1200: 0.20000000298023224

```

---

### Code Overview

Here is an overview of the main components:

- **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)**:

  - Tennis environments has 2 agents. Each agents has it's own actor and critic network
  - Evaluation of critics is centralized during training because the critic uses the global state and actions of both agents to optimize policies. However, during execution, each agent operates independently based on its policy

- **Actor-Critic-Network**:
  The Tennis environment has 2 agents. Each agent has its own actor and critic networks:

  - Actor Network: Responsible for selecting actions based on the current state.
  - Critic Network: Evaluates the quality of the state-action pair.

  Implementation of the ActorCritic can be found in:

  ```bash
  ActorCritic.py
  ```

  Implementation of the DDPGAgent can be found in:

  ```bash
  Agent.py
  ```

- **Replay Memory**:
  Code for Replay Buffer is in:

  ```bash
  ReplayBuffer.py
  ```

- **Ornstein-Uhlenbeck Noise (OU Noise)**:
  To encourage exploration in continuous action environment, the agent uses OUNoise  
  Code for OUNoise is in:

  ```bash
  OUNoise.py
  ```

### Run Pretrained Model

1. open with Jupyter Notebook

```
Run_Pretrained.ipynb
```

2. Replace the file `Tennis.x86_64` according to your OS. The example provided is for Linux

3. Run all
