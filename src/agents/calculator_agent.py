from .agent import Agent, MatrixColumn
import numpy as np
from functools import lru_cache


class CalculatorAgent(Agent):
    total_count = 0
    _factors = np.array([100, 10, 1])

    def __init__(self, risk = 0.2):
        CalculatorAgent.total_count += 1
        if CalculatorAgent.total_count == 1:
            super().__init__("Sheldon")
        else:
            super().__init__(f"Sheldon{CalculatorAgent.total_count}")
        self.risk = risk # will maximize points, but with a probability of overshooting of X
        self.value_matrix = np.zeros([3], int)
        self.occupancy = np.zeros(3, int)

    def startGame(self):
        self.occupancy *= 0
        self.value_matrix *= 0

    def doMove(self, diceValue):
        available_actions = np.where(3 - self.occupancy > 0)[0]
        action_risk = [(action, self._evaluate_action(diceValue, action)) for action in available_actions]
        low_risk_group = [(action, risk) for action, risk in action_risk if risk <= self.risk]
        if len(low_risk_group) > 0:
            action = sorted(low_risk_group, key=lambda x: x[1])[-1] # take max risk within low group
        else:
            action = sorted(action_risk, key=lambda x: x[1])[0] # take min risk
        self.occupancy[action[0]] += 1
        self.value_matrix[action[0]] += diceValue
        return MatrixColumn(action[0])

    def _evaluate_action(self, diceValue, action_index) -> float: # returns the probability of overshooting with this action
        future_value = self._eval(self.value_matrix + self._delta(action_index, diceValue))
        gap = 1000 - future_value
        test_occupancy = self.occupancy.copy()
        test_occupancy[action_index] += 1
        return self.eval_risk(CalculatorAgent.__concat_args(test_occupancy, gap))

    @lru_cache(maxsize=None)
    def eval_risk(self, occupancy_gap) -> float:
        occupancy = np.array(occupancy_gap[:3])
        gap = occupancy_gap[3]
        lower_bound = np.dot((3-occupancy), self._factors)
        upper_bound = lower_bound * 6
        if gap < lower_bound:
            return 1.0
        elif gap > upper_bound:
            return 0.0
        possible_throws = np.array([1,2,3,4,5,6])
        if np.sum(3 - occupancy) == 1:
            # last throw
            factor = np.dot(3-occupancy, self._factors)
            risk = np.sum((factor * possible_throws) > gap) / 6
            return risk
        risk_per_throw = []
        for throw in possible_throws: # choose last to fill column
            throw_risk = np.ones(3, float)
            for c in np.where(occupancy < 3)[0]:
                test_occupancy = occupancy.copy()
                test_occupancy[c] += 1
                throw_risk[c] = self.eval_risk(CalculatorAgent.__concat_args(test_occupancy, gap - self._factors[c]*throw))
            try:
                risk_per_throw.append(np.max(throw_risk[throw_risk < self.risk]))
            except ValueError:
                risk_per_throw.append(np.min(throw_risk))
        risk = np.mean(np.array(risk_per_throw))
        return risk

    @staticmethod
    def __concat_args(occ, gap):
        return (occ[0], occ[1], occ[2], gap)

    def _eval(self, matrix=None):
        if matrix is None:
            return np.dot(self.value_matrix, self._factors)
        else:
            return np.dot(matrix, self._factors)

    def _delta(self, index, value):
        mat = np.zeros(3, dtype=int)
        mat[index] = value
        return mat
