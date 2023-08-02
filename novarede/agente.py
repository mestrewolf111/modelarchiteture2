import random
import numpy as np
from rede_neural import QNetwork

class Agente:
    def __init__(self, q_network):
        self.q_network = q_network
        self.exploration_rate = 0.9  # Taxa de exploração (epsilon-greedy)
        self.actions = ['w', 's', 'a', 'd']  # Ações possíveis (cima, baixo, esquerda, direita)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)
        else:
            q_values = [self.q_network.get_q_value(state, action) for action in self.actions]
            max_q = max(q_values)
            count = q_values.count(max_q)
            if count > 1:
                best_options = [i for i in range(len(self.actions)) if q_values[i] == max_q]
                return random.choice(best_options)
            else:
                max_index = q_values.index(max_q)
                return max_index

    def learn(self, state, action, reward, next_state, discount_factor):
        old_q_value = self.q_network.get_q_value(state, action)
        q_values_next = self.q_network.predict(next_state)
        next_max_q = np.max(q_values_next)
        new_q_value = (1 - self.q_network.learning_rate) * old_q_value + self.q_network.learning_rate * (reward + discount_factor * next_max_q)
        self.q_network.update_q_value(state, action, new_q_value)

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay