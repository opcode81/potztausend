import random
from typing import List
import matplotlib.pyplot as plt

from src.agents import MatrixColumn, Agent
epsilon = 1e-10
e_inf = 1.0/epsilon

class GameState:
    def __init__(self):
        self.stateLines = {
                MatrixColumn.LEFT_COLUMN: [],
                MatrixColumn.MIDDLE_COLUMN: [],
                MatrixColumn.RIGHT_COLUMN: []}

    def getPossibleActions(self) -> List[MatrixColumn]:
        actions = []
        for action in list(MatrixColumn):
            if len(self.stateLines[action]) < 3:
                actions.append(action)
        return actions

    def performAction(self, action: MatrixColumn, diceValue: int):
        self.stateLines.get(action).append(diceValue)

    def calcScore(self) -> int:
        score = sum(self.stateLines[MatrixColumn.LEFT_COLUMN]) * 100
        score += sum(self.stateLines[MatrixColumn.MIDDLE_COLUMN]) * 10
        score += sum(self.stateLines[MatrixColumn.RIGHT_COLUMN])

        if score > 1000:
            return -score

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


class Participant:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.state = GameState()
        self.invalid = False
        self.score = 0
        self.score_history = []

    def startGame(self):
        self.state = GameState()
        self.agent.startGame()

    def doMove(self, diceValue: int):
        if not self.invalid:
            action = self.agent.doMove(diceValue)
            if action in self.state.getPossibleActions():
                self.state.performAction(action, diceValue)
            else:
                self.invalid = True
                self.score = 0

    def getGameResult(self) -> int:
        return self.state.calcScore()

    def getScore(self) -> int:
        return self.score

    def updateScore(self, score: int):
        self.score += score
        self.score_history.append(abs(self.getGameResult()))


class Competition:
    def __init__(self, numberOfGames: int = 10000):
        self.numberOfGames = numberOfGames
        self.participants: List[Participant] = []

    def startCompetition(self):
        for game in range(0, self.numberOfGames):
            # init game
            for participant in self.participants:
                participant.startGame()

            # play game
            for gameRound in range(0, 9):
                diceValue = random.randint(1, 6)

                for participant in self.participants:
                    participant.doMove(diceValue)

            # evaluate game
            bestResult = 0
            bestParticipant = None
            for participant in self.participants:
                result = participant.getGameResult()
                if result < 0:
                    participant.updateScore(-3)
                elif result > bestResult:
                    bestResult = result
                    bestParticipant = participant

            if bestParticipant:
                bestParticipant.updateScore(5)

    def registerParticipant(self, agent: Agent):
        self.participants.append(Participant(agent))

    def printResult(self):
        self.participants = sorted(self.participants, key=lambda x: x.getScore() - x.invalid*e_inf, reverse=True)
        for rank, participant in enumerate(self.participants):
            if participant.invalid:
                print(f"\t {participant.agent.agentName}: disqualified")
            else:
                print(f"{rank+1}\t {participant.agent.agentName}: {participant.getScore()}")

    def plot_score_history(self):
        plt.figure()
        for participant in self.participants:
            plt.plot(participant.score_history, label=participant.agent.agentName)
        plt.legend()
        plt.show()

    def plot_histogram(self):
        plt.figure()
        for participant in self.participants:
            plt.hist(participant.score_history, label=participant.agent.agentName, alpha=0.5, bins=20)
        plt.legend()
        plt.show()
