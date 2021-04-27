import random
from typing import List

from agents import Agent, MatrixColumn


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


class Participant:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.state = GameState()
        self.invalid = False
        self.score = 0

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
                if result == -1000:
                    participant.updateScore(-3)
                elif result > bestResult:
                    bestResult = result
                    bestParticipant = participant

            bestParticipant.updateScore(5)

    def registerParticipant(self, agent: Agent):
        self.participants.append(Participant(agent))

    def printResult(self):
        for participant in self.participants:
            print(f"{participant.agent.agentName}: {participant.getScore()}")

