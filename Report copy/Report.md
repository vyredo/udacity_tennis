# Reinforcement Learning Nanodegree - Reacher Unity Environment Report

## Pretrained execution

Below is the youtube video of running pretrained model (click to open in youtube).
<a href="https://www.youtube.com/watch?v=fT6UDkvKAaI">
<img src="https://github.com/vyredo/udacity_tennis/blob/main/Report/udacity_reacher_20_agents.jpg"/>
</a>

- You can open file `Run_Pretrained.ipynb` to run it.
- pretrained model for agent1 is located at [`Report/agent1_actor.pth and Report/agent1_critic.pth`](https://github.com/vyredo/udacity_tennis/tree/main/Report)
- pretrained model for agent2 is located at [`Report/agent2_actor.pth and Report/agent2_critic.pth`](https://github.com/vyredo/udacity_tennis/tree/main/Report)

## Learning Algorithm

The agent is implemented using:

1. **MADDPG**:

   - The Deep Deterministic Policy Gradient (DDPG) algorithm leverages a deterministic policy for continuous action spaces.
     It uses an Actor-Critic architecture where:
     - The Actor generates the optimal action for a given state.
     - The Critic evaluates the Q-value of the action-state pair.

2. **ReplayBuffer**:

   - The Replay Buffer stores past experiences as tuples of (state, action, reward, next_state) to break temporal correlations in the data.

3. **OU Noise**:
   - It generates temporally correlated noise to encourage exploration while maintaining smooth changes in actions

## Log of Rewards

For the full log, check [this link](https://github.com/vyredo/udacity_tennis/blob/main/Report/scores_logs.txt)

The whole training is done for **10000 episodes**.

## Plot of Rewards

This are the plot for 6600 episodes,

<img src="https://github.com/vyredo/udacity_tennis/blob/main/Report/scores_plot.png" alt="Reward Plot" />
