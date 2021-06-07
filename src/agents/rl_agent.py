import os
import pickle
import random

import numpy as np

from .agent import Agent, MatrixColumn


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

    def copyAndPerformAction(self, action, value):
        newState = State()
        newState.stateLines = {
                MatrixColumn.LEFT_COLUMN: self.stateLines[MatrixColumn.LEFT_COLUMN].copy(),
                MatrixColumn.MIDDLE_COLUMN: self.stateLines[MatrixColumn.MIDDLE_COLUMN].copy(),
                MatrixColumn.RIGHT_COLUMN: self.stateLines[MatrixColumn.RIGHT_COLUMN].copy()}
        newState.performAction(action, value)
        return newState

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


class QLearningAgent(Agent):
    def __init__(self, name="Paul"):
        super().__init__(name)
        path = os.path.join("model_resources", "paul", "q.pickle")
        with open(path, "rb") as f:
            self.qTable = pickle.load(f)
        self.state = None
        self.rand = random.Random(42)

    def startGame(self):
        self.state = State()

    def doMove(self, diceValue) -> MatrixColumn:
        possibleActions = [a.value for a in self.state.getPossibleActions()]
        stateIndex = self.state.toIndex()
        bestPossibleActionValue = np.max(self.qTable[stateIndex, diceValue - 1, possibleActions])
        bestPossibleAction = possibleActions[self.rand.choice(np.where(self.qTable[stateIndex, diceValue - 1, possibleActions] == bestPossibleActionValue)[0])]
        self.state.performAction(bestPossibleAction, diceValue)
        return MatrixColumn(bestPossibleAction)


class TemporalDifferenceAgent(Agent):
    def __init__(self):
        super().__init__("Gunter")
        path = os.path.join("model_resources", "gunter", "v.pickle")
        with open(path, "rb") as f:
            self.vTable = pickle.load(f)
        self.state = None

    def startGame(self):
        self.state = State()

    def doMove(self, diceValue) -> MatrixColumn:
        actions = self.state.getPossibleActions()
        bestV = None
        bestAction = None
        for action in actions:
            newState = self.state.copyAndPerformAction(action, diceValue)
            if bestAction is None:
                bestAction = action
                bestV = self.vTable[newState.toIndex()]
            elif self.vTable[newState.toIndex()] > bestV:
                bestAction = action
                bestV = self.vTable[newState.toIndex()]
        self.state.performAction(bestAction, diceValue)
        return MatrixColumn(bestAction)


class MonteCarloAgent(Agent):
    def __init__(self):
        super().__init__("Lotte")
        path = os.path.join("model_resources", "lotte", "v.pickle")
        with open(path, "rb") as f:
            self.vTable = pickle.load(f)
        self.state = None

    def startGame(self):
        self.state = State()

    def doMove(self, diceValue) -> MatrixColumn:
        actions = self.state.getPossibleActions()
        bestV = None
        bestAction = None
        for action in actions:
            newState = self.state.copyAndPerformAction(action, diceValue)
            if bestAction is None:
                bestAction = action
                bestV = self.vTable[newState.toIndex()]
            elif self.vTable[newState.toIndex()] > bestV:
                bestAction = action
                bestV = self.vTable[newState.toIndex()]
        self.state.performAction(bestAction, diceValue)
        return MatrixColumn(bestAction)
