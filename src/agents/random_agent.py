import random

from .agent import Agent, MatrixColumn


class RandomAgent(Agent):

    def __init__(self):
        super().__init__("Randy")
        self.lines = {}
        self.rand = random.Random(42)

    def startGame(self):
        self.lines = {MatrixColumn.LEFT_COLUMN: 0,
                      MatrixColumn.MIDDLE_COLUMN: 0,
                      MatrixColumn.RIGHT_COLUMN: 0}

    def doMove(self, diceValue):
        action = self.rand.choice([line for line, count in self.lines.items() if count < 3])
        self.lines[action] += 1
        return action
