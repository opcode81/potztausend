import os
import pickle
import random

import numpy as np

from .agent import Agent, MatrixColumn


class QLearningAgent(Agent):
    def __init__(self):
        super().__init__("Paul")
        path = os.path.join("model_resources", "paul", "qtable.pickle")
        with open(path, "rb") as f:
            self.qtable = pickle.load(f)
        self.state = None

    def startGame(self):
        self.state = QLearningAgent.State()

    def doMove(self, diceValue) -> MatrixColumn:
        possibleActions = [a.value for a in self.state.getPossibleActions()]
        stateIndex = self.state.toIndex()
        bestPossibleActionValue = np.max(self.qtable[stateIndex, diceValue - 1, possibleActions])
        bestPossibleAction = possibleActions[random.choice(np.where(self.qtable[stateIndex, diceValue - 1, possibleActions] == bestPossibleActionValue)[0])]
        self.state.performAction(bestPossibleAction, diceValue)
        #print(f"{self.state}")
        #print(f"action = {MatrixColumn(bestPossibleAction)} - {bestPossibleAction}")
        #print(f"possibleActions = {possibleActions}")
        return MatrixColumn(bestPossibleAction)

    class State:
        def __init__(self):
            self.stateLines = {MatrixColumn.LEFT_COLUMN: [],
                               MatrixColumn.MIDDLE_COLUMN: [],
                               MatrixColumn.RIGHT_COLUMN: []}

        def getPossibleActions(self):
            actions = []
            for action in list(MatrixColumn):
                if len(self.stateLines[action]) < 3:
                    actions.append(action)
            return actions

        def performAction(self, action, value):
            self.stateLines.get(MatrixColumn(action)).append(value)

        def toIndex(self):
            def _calcIndexOfLine(stateLine):
                return int(sum(stateLine) + (3 - len(stateLine)) * 19)

            return _calcIndexOfLine(self.stateLines[MatrixColumn.LEFT_COLUMN]) + \
                   _calcIndexOfLine(self.stateLines[MatrixColumn.MIDDLE_COLUMN]) * 76 + \
                   _calcIndexOfLine(self.stateLines[MatrixColumn.RIGHT_COLUMN]) * 76 * 76

        def __str__(self):
            rows = []
            firstLine = self.stateLines.get(MatrixColumn.LEFT_COLUMN)
            secondLine = self.stateLines.get(MatrixColumn.MIDDLE_COLUMN)
            thirdLine = self.stateLines.get(MatrixColumn.RIGHT_COLUMN)
            for row in range(0, 3):
                rows.append("%d | %d | %d" % (firstLine[row] if row < len(firstLine) else 0, secondLine[row] if row < len(secondLine) else 0,
                                              thirdLine[row] if row < len(thirdLine) else 0))
            return "\n".join(rows)
