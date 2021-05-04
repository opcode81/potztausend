from .agent import Agent, MatrixColumn
import numpy as np
from functools import lru_cache
import random


class UtilAgent(Agent):
    """
    A utilitarian agent, named after Jeremy Bentham, the father of modern utilitarianism.
    It maximises the expected utility according to the rules of the game (-1000 for overshoot).
    """
    _factors = np.array([100, 10, 1])
    OVERSHOOT_REWARD = -1000

    def __init__(self):
        super().__init__("Jeremy")
        self.value_matrix = np.zeros([3], int)
        self.occupancy = np.zeros(3, int)
        self.rand = random.Random(42)

    def startGame(self):
        self.occupancy *= 0
        self.value_matrix *= 0

    def doMove(self, diceValue):
        available_actions = np.where(3 - self.occupancy > 0)[0]
        action_util = [(action, self._evaluate_action(diceValue, action)) for action in available_actions]
        max_util = max([au[1] for au in action_util])
        candidate_actions = [au[0] for au in action_util if au[1] == max_util]
        action = self.rand.choice(candidate_actions)
        self.occupancy[action] += 1
        self.value_matrix[action] += diceValue
        return MatrixColumn(action)

    def _evaluate_action(self, diceValue, action_index) -> float:  # returns the utility of the action
        old_value = self._eval(self.value_matrix)
        added_value = self._factors[action_index] * diceValue
        future_value = old_value + added_value
        gap = 1000 - future_value
        test_occupancy = self.occupancy.copy()
        test_occupancy[action_index] += 1
        return self.eval_utility(UtilAgent.__concat_args(test_occupancy, gap), added_value)

    @lru_cache(maxsize=None)
    def eval_utility(self, occupancy_gap, util) -> float:
        occupancy = np.array(occupancy_gap[:3])
        gap = occupancy_gap[3]

        if gap < 0:
            return self.OVERSHOOT_REWARD

        available_columns = np.where(occupancy < 3)[0]

        if len(available_columns) == 0:
            return util

        lower_bound = np.dot((3-occupancy), self._factors)
        if gap < lower_bound:
            return self.OVERSHOOT_REWARD

        possible_throws = np.array([1,2,3,4,5,6])

        util_per_throw = []
        for throw in possible_throws: # choose last to fill column
            throw_util = np.ones(3, float) * self.OVERSHOOT_REWARD
            for c in available_columns:
                test_occupancy = occupancy.copy()
                test_occupancy[c] += 1
                added_value = self._factors[c] * throw
                u = self.eval_utility(UtilAgent.__concat_args(test_occupancy, gap - added_value), util + added_value)
                throw_util[c] = u
            util_per_throw.append(np.max(throw_util))
        mean_util = np.mean(np.array(util_per_throw))
        return mean_util

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
