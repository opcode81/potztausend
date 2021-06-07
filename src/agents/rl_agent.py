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

    def calcScore(self):
        score = sum(self.stateLines[MatrixColumn.LEFT_COLUMN]) * 100
        score += sum(self.stateLines[MatrixColumn.MIDDLE_COLUMN]) * 10
        score += sum(self.stateLines[MatrixColumn.RIGHT_COLUMN])

        if score > 1000:
            return -1000

        return score

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
    def __init__(self, name="Paul", epsilon=0.3, alpha=0.25, gamma=1, nIterations=20000000):
        super().__init__(name)
        self.path = os.path.join("model_resources", "paul", "q.pickle")
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.Q = pickle.load(f)
        else:
            self.Q = np.zeros((76 * 76 * 76, 6, 3))
        self.state = None
        self.rand = random.Random(42)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.nIterations = nIterations

    def train(self):
        for i in range(0, self.nIterations):
            state = State()

            diceValues = np.random.randint(1, 7, 9)
            for diceValue in diceValues:

                if random.uniform(0, 1) < self.epsilon:
                    # explore
                    action = random.choice(state.getPossibleActions())
                else:
                    action = self._choiceAction(state, diceValue)

                newState, reward, done = self._performAction(state, action, diceValue)
                self._updateQ(state, newState, diceValue, action, reward)
                state = newState

        with open(self.path, "wb") as f:
            pickle.dump(self.Q, f)

    def startGame(self):
        self.state = State()

    def doMove(self, diceValue) -> MatrixColumn:
        possibleActions = [a.value for a in self.state.getPossibleActions()]
        stateIndex = self.state.toIndex()
        bestPossibleActionValue = np.max(self.Q[stateIndex, diceValue - 1, possibleActions])
        bestPossibleAction = possibleActions[self.rand.choice(np.where(self.Q[stateIndex, diceValue - 1, possibleActions] == bestPossibleActionValue)[0])]
        self.state.performAction(bestPossibleAction, diceValue)
        return MatrixColumn(bestPossibleAction)

    @staticmethod
    def _performAction(state, action, diceValue):
        newState = state.copyAndPerformAction(action, diceValue)

        done = len(newState.getPossibleActions()) == 0

        reward = QLearningAgent._reward(newState, done)
        return newState, reward, done

    @staticmethod
    def _reward(state, done):
        reward = state.calcScore()

        if reward > 1000 or reward == -1000:
            return -1000

        if not done:
            return 0
        else:
            return reward

    def _updateQ(self, state, newState, diceValue, action, reward):
        stateIndex = state.toIndex()
        stateNewIndex = newState.toIndex()
        qNewState = sum([np.max(self.Q[stateNewIndex, d, :]) for d in range(0, 6)]) / 6
        self.Q[stateIndex, diceValue - 1, action] += self.alpha * (reward + self.gamma * qNewState - self.Q[stateIndex, diceValue - 1, action])

    def _choiceAction(self, state, diceValue):
        possibleActions = state.getPossibleActions()
        stateIndex = state.toIndex()
        bestPossibleActionValue = np.max(self.Q[stateIndex, diceValue - 1, possibleActions])
        bestPossibleAction = random.choice(np.where(self.Q[stateIndex, diceValue - 1, possibleActions] == bestPossibleActionValue)[0])
        return possibleActions[bestPossibleAction]


class TemporalDifferenceAgent(Agent):
    def __init__(self, alpha=0.2, gamma=1):
        super().__init__("Gunter")
        self.path = os.path.join("model_resources", "gunter", "v.pickle")
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.V = pickle.load(f)
        else:
            self.V = np.zeros(76 * 76 * 76)
        self.state = None
        self.alpha = alpha
        self.gamma = gamma

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
                bestV = self.V[newState.toIndex()]
            elif self.V[newState.toIndex()] > bestV:
                bestAction = action
                bestV = self.V[newState.toIndex()]
        self.state.performAction(bestAction, diceValue)
        return MatrixColumn(bestAction)

    def train(self):
        for i in range(0, 20000000):
            state = State()
            diceValues = np.random.randint(1, 7, 9)
            for diceValue in diceValues:
                action = self._choiceAction(state, diceValue)
                newState = state.copyAndPerformAction(action, diceValue)
                if len(newState.getPossibleActions()) == 0:
                    reward = newState.calcScore()
                else:
                    reward = 0
                self.V[state.toIndex()] = self.V[state.toIndex()] + self.alpha * (reward + self.gamma * self.V[newState.toIndex()] - self.V[state.toIndex()])
                state = newState

        with open(self.path, "wb") as f:
            pickle.dump(self.V, f)

    def _choiceAction(self, state, diceValue):
        actions = state.getPossibleActions()
        bestV = None
        bestAction = None
        for action in actions:
            newState = state.copyAndPerformAction(action, diceValue)
            if bestAction is None:
                bestAction = action
                bestV = self.V[newState.toIndex()]
            elif self.V[newState.toIndex()] > bestV:
                bestAction = action
                bestV = self.V[newState.toIndex()]
        return bestAction


class MonteCarloAgent(Agent):
    def __init__(self):
        super().__init__("Lotte")
        self.path = os.path.join("model_resources", "lotte", "v.pickle")
        self.visitedPath = os.path.join("temp", "lotte", "visited.pickle")
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.V = pickle.load(f)
        else:
            self.V = np.zeros(76 * 76 * 76)

        if os.path.exists(self.visitedPath):
            with open(self.visitedPath, "rb") as f:
                self.visited = pickle.load(f)
        else:
            self.visited = np.zeros(76 * 76 * 76)

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
                bestV = self.V[newState.toIndex()]
            elif self.V[newState.toIndex()] > bestV:
                bestAction = action
                bestV = self.V[newState.toIndex()]
        self.state.performAction(bestAction, diceValue)
        return MatrixColumn(bestAction)

    def train(self):
        actions = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        for i in range(0, 10000000):
            states = []
            state = State()
            gamesActions = np.random.permutation(actions)
            diceValues = np.random.randint(1, 7, 9)
            for a, d in zip(gamesActions, diceValues):
                newState = state.copyAndPerformAction(a, d)
                states.append(newState)
                state = newState

            reward = state.calcScore()
            for s in states:
                index = s.toIndex()
                self.V[index] = (self.V[index] * self.visited[index] + reward) / (self.visited[index] + 1)
                self.visited[index] += 1

        with open(self.path, "wb") as f:
            pickle.dump(self.V, f)
        with open(self.visitedPath, "wb") as f:
            pickle.dump(self.visited, f)
