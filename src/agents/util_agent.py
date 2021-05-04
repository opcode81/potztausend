import random
from functools import lru_cache
from typing import Tuple

import numpy as np

from .agent import Agent, MatrixColumn
from .. import competition


class UtilAgent(Agent):
    """
    A utilitarian agent, named after Jeremy Bentham, the father of modern utilitarianism.
    It maximises the expected utility according to the rules of the game (-1000 for overshoot).
    """
    FACTORS = np.array([100, 10, 1])
    DICE_VALUES = [1, 2, 3, 4, 5, 6]
    OVERSHOOT_UTIL = competition.OVERSHOOT_SCORE

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

    def _evaluate_action(self, diceValue, action_index) -> float:
        old_value = np.dot(self.value_matrix, self.FACTORS)
        added_value = self.FACTORS[action_index] * diceValue
        future_value = old_value + added_value
        resulting_occupancy = self.occupancy.copy()
        resulting_occupancy[action_index] += 1
        return self._eval_utility(tuple(resulting_occupancy), future_value)

    @lru_cache(maxsize=None)
    def _eval_utility(self, occupancy: Tuple[int, int, int], util) -> float:
        gap = 1000 - util
        if gap < 0:
            return self.OVERSHOOT_UTIL

        occupancy = np.array(occupancy)
        available_columns = np.where(occupancy < 3)[0]

        if len(available_columns) == 0:
            return util

        lower_bound = np.dot((3-occupancy), self.FACTORS)
        if gap < lower_bound:
            return self.OVERSHOOT_UTIL

        next_dice_value_utils = []
        for dice_value in self.DICE_VALUES:
            column_util = np.ones(3, float) * self.OVERSHOOT_UTIL
            for c in available_columns:
                next_occupancy = occupancy.copy()
                next_occupancy[c] += 1
                added_value = self.FACTORS[c] * dice_value
                column_util[c] = self._eval_utility(tuple(next_occupancy), util + added_value)
            next_dice_value_utils.append(np.max(column_util))
        mean_util = np.mean(next_dice_value_utils)
        return mean_util
