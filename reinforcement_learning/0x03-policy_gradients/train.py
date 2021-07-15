#!/usr/bin/env python3
""" the trtaining method using policy gradients """
import numpy as np
import policy_gradient as pg


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """ train the policy gradients
        env: the initia environment (from openai gym)
        nb_episodes: number of episodes for training
        alpha: learning rate
        gamma: discount factor
        Returns: all vlaues of the score (sum of rewards during ea. episode)
    """
    # initializ future return
    scores = []
    # initialize rnadom starting weights
    weights = np.random.rand(env.observation_space.shape[0],
                             env.action_space.n)
    # loop through episodes performing steps
    for ep in range(nb_episodes):
        state = env.reset()[None, :]
        # initialize variables for the ep
        grads = []
        rewards = []
        actions = []
        done = False
        counter = 0
        # run episode
        while not done:
            # if using colab be ware of import changes
            action, grad = pg.policy_gradient(state, weights)
            state, reward, done, info = env.step(action)
            state = state[None, :]
            grads.append(grad)
            rewards.append(reward)
            actions.append(action)
            counter += 1
        # when episodes ended calculate rewards/new weights
        for i in range(len(grads)):
            # Loop through everything that happend in the episode
            rew = sum([r * (gamma ** r) for t, r in enumerate(rewards[i:])])
            weights += alpha * grads[i] * rew
        # end_reward = 0
        # for i in range(counter)
        #     end_reward = reward[counter - i] + end_reward * gamma
        #     weights[:, action] += alpha * grad[:, action] *
        scores.append(sum(rewards))
        print(ep, scores[ep], end="\r", flush=False)
    return scores
