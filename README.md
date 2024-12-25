# Tennis - Reinforcement Learning Project

This repository contains the solution to the Udacity Nanodegree project for the Tennis Navigation environment. The agent is trained using a MADDPG (Multi Agent Deep Deterministic Policy Gradient ) with Replay Buffer, The example provide is for linux OS, you will need to include another file for another OS.

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

## Project Details

### Environment Overview

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Solution

This environment has a continuous action space, making it suitable for algorithms like DDPG. Since there are two agents interacting, a multi-agent version of DDPG (MADDPG) is used.

The algorithm is trained in large number of episodes by using NVIDIA RTX 4060Ti

### Key Features

- Algorithm: MADDPG Agent with Replay Buffer for efficient learning.

- Training is done iteratively to debug each session. The first training session ran for 1250 episodes. The plot for this session
  <img src="https://github.com/vyredo/udacity_tennis/blob/main/Report/scores_plot_prev_1250.png" />

- Similarly, for the second training session: The second training session ran for 10,000 episodes. The plot for this session
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

## Training Progress

Overall the training is done in around 10_000 episodes.

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

## Code Overview

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

## Run Pretrained Model

1. open with Jupyter Notebook

```
Run_Pretrained.ipynb
```

2. Replace the file `Tennis.x86_64` according to your OS. The example provided is for Linux

3. Run all
