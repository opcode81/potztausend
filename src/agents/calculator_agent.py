from .agent import Agent, MatrixColumn
import numpy as np


class CalculatorAgent(Agent):
    total_count = 0
    _factors = np.array([100, 10, 1])

    def __init__(self):
        CalculatorAgent.total_count += 1
        if CalculatorAgent.total_count == 1:
            super().__init__("Sheldon")
        else:
            super().__init__(f"Sheldon{CalculatorAgent.total_count}")
        self.lines = np.zeros(3, int)
        self.risk = 0.1 # will maximize points, but with a probability of overshooting of 10%
        self.matrix = np.zeros([3], dtype=int)

    def startGame(self):
        self.lines = self.lines * 0

    def doMove(self, diceValue):
        available_actions = [line_id for line_id, line_value in enumerate(self.lines) if line_value < 3]
        action_risk = [(action, self._evaluate_action(diceValue, action)) for action in available_actions]
        low_group = [(action, risk) for action, risk in action_risk if risk <= self.risk]
        if len(low_group) > 0:
            action = sorted(low_group, key=lambda x: x[1])[-1] # take max risk within low group
        else:
            action = sorted(action_risk, key=lambda x: x[1])[0] # take min risk
        self.lines[action[0]] += 1
        self.matrix[action[0]] += diceValue
        return MatrixColumn(action[0])

    def _evaluate_action(self, diceValue, action_index) -> float: # returns the probability of overshooting with this action
        future_value = self._eval(self.matrix + self._delta(action_index, diceValue))
        gap = 1000 - future_value
        return 0.5

    def _eval(self, matrix=None):
        if matrix is None:
            return np.dot(self.matrix, self._factors)
        else:
            return np.dot(matrix, self._factors)

    def _delta(self, index, value):
        mat = np.zeros(3, dtype=int)
        mat[index] = value
        return mat
