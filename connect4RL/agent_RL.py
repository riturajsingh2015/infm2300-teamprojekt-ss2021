import numpy as np
import random
import csv
import os
from collections import defaultdict


class TicRLAgent:

    def __init__(self, tag, epsilon=1):
        self.tag = tag
        self.epsilon = epsilon
        self.prev_state = np.zeros((6, 7))
        self.prev_move = -1
        self.state = None
        self.move = None
        self.count_memory = 0
        self.cache = []
        self.Q = defaultdict(lambda: np.zeros(7))
        self.alpha = 0.5
        self.gamma = 1
        self.policy = self.createEpsilonGreedyPolicy(self.Q, self.epsilon, 7)
        
        
    def createEpsilonGreedyPolicy(self, Q, epsilon, num_actions):
        """
        Creates an epsilon-greedy policy based
        on a given Q-function and epsilon.

        Returns a function that takes the state
        as an input and returns the probabilities
        for each action in the form of a numpy array 
        of length of the action space(set of possible actions).
        """
        def policyFunction(state):
     
            Action_probabilities = np.ones(num_actions) * epsilon / num_actions

            best_action = np.argmax(self.Q[state])
            Action_probabilities[best_action] += (1.0 - epsilon)
            return Action_probabilities

        return policyFunction 
    
    def qLearning(self, next_state, action, winner, discount_factor = 1.0,
                                alpha = 0.6, epsilon = 0.1):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while improving
        following an epsilon-greedy policy"""   
       
        # self.check_winner(board, wins)

        reward = self.reward(winner)

        # Update statistics
        #stats.episode_rewards[ith_episode] += reward
        #stats.episode_lengths[ith_episode] = t

        # TD Update
        best_next_action = np.argmax(self.Q[next_state]) 
        #best_actions[best_next_action] = best_actions[best_next_action] + 1

        self.Q[self.state][action] = ((1-alpha) * self.Q[self.state][action]) + (alpha * (reward + (discount_factor * self.Q[self.state][best_next_action]) - self.Q[next_state][action]))

        self.state = next_state
        
        return self.Q
    
    
    def choose_move(self, state, winner):

         # get probabilities of all actions from current state
        action_probabilities = self.policy(state.tobytes())

        # choose action according to 
        # the probability distribution
        self.action = np.random.choice(np.arange(len(action_probabilities)),p = action_probabilities)
        #actions[self.action] = actions[self.action] + 1

        avaMoves = np.where(state[0, :] == 0)[0]
        while winner is None: 
            if self.action not in avaMoves:
                self.action = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)
            else:
                break
                    
        return self.action

    def reward(self, winner):

        if winner is self.tag:
            reward = 100
        elif winner is None:
            reward = 1
        elif winner == 0:
            reward = 50
        else:
            reward = -100
        return reward

    def game_winner(self, state):
        winner = None
        for i in range(len(state[:,0])-3):
            for j in range(len(state[0, :])-3):
                winner = self.square_winner(state[i:i+4, j:j+4])
                if winner is not None:
                    # print('winner is:', self.winner)
                    break
            if winner is not None:
                # print('winner is:', self.winner)
                break

        if np.min(np.abs(state[0, :])) != 0:
            winner = 0
            # print('no winner')

        return winner

    @staticmethod
    def ava_moves(state):
        moves = np.where(state[0, :] == 0)[0]
        return moves
    
    def square_winner(square):
        s = np.append([np.sum(square, axis=0), np.sum(square, axis=1).T],
                      [np.trace(square), np.flip(square,axis=1).trace()])
        if np.max(s) == 4:
            winner = 1
        elif np.min(s) == -4:
            winner = 2
        else:
            winner = None
        return winner

    @staticmethod
    def state_to_vector(state, move):

        vec = np.zeros((1, 7))
        if move != -1:
            vec[0, move] = 1
        tensor = np.append(vec, state, axis=0)
        tensor = tensor.reshape((1, 7, 7, 1))

        return tensor

    @staticmethod
    def make_state_from_move(state, move, player):
        if move is None:
            return state

        state = np.array(state)
        if player == 1:
            tag = 1
        else:
            tag = -1
            
        idy = np.where(state[:, move] == 0)[0][-1]
        state = np.array(state)
        state[idy, move] = tag

        return state