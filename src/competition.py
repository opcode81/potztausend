import logging
import random
from enum import Enum
from typing import List
import matplotlib.pyplot as plt
import numpy as np


from agents import MatrixColumn, Agent
epsilon = 1e-10
e_inf = 1.0/epsilon

log = logging.getLogger(__name__)

OVERSHOOT_SCORE = -1000
INVALID_MOVE_SCORE = -10000


class GameState:
    class Status(Enum):
        RUNNING = 0
        OVER_SUCCESS = 1
        OVER_OVERSHOOT = 2
        OVER_INVALID_MOVE = 3

        def isOver(self):
            return self != self.RUNNING

    def __init__(self):
        self.stateLines = {
                MatrixColumn.LEFT_COLUMN: [],
                MatrixColumn.MIDDLE_COLUMN: [],
                MatrixColumn.RIGHT_COLUMN: []}
        self.invalidActionPerformed = False
        self.numbersPlaced = 0

    def getPossibleActions(self) -> List[MatrixColumn]:
        actions = []
        for action in list(MatrixColumn):
            if len(self.stateLines[action]) < 3:
                actions.append(action)
        return actions

    def performAction(self, action: MatrixColumn, diceValue: int):
        if not self.invalidActionPerformed:
            stateLine = self.stateLines[action]
            if len(stateLine) < 3:
                stateLine.append(diceValue)
                self.numbersPlaced += 1
            else:
                self.invalidActionPerformed = True

    def getGameStatus(self) -> Status:
        if self.invalidActionPerformed:
            return self.Status.OVER_INVALID_MOVE
        elif self.calcSum() > 1000:
            return self.Status.OVER_OVERSHOOT
        elif self.numbersPlaced == 9:
            return self.Status.OVER_SUCCESS
        else:
            return self.Status.RUNNING

    def isGameOver(self):
        return self.getGameStatus().isOver()

    def calcSum(self) -> int:
        s = sum(self.stateLines[MatrixColumn.LEFT_COLUMN]) * 100
        s += sum(self.stateLines[MatrixColumn.MIDDLE_COLUMN]) * 10
        s += sum(self.stateLines[MatrixColumn.RIGHT_COLUMN])
        return s

    def stateToList(self):
        stateList = []
        firstLine = self.stateLines.get(MatrixColumn.LEFT_COLUMN)
        secondLine = self.stateLines.get(MatrixColumn.MIDDLE_COLUMN)
        thirdLine = self.stateLines.get(MatrixColumn.RIGHT_COLUMN)
        for row in range(0, 3):
            stateList.append(firstLine[row] if row < len(firstLine) else 0)
            stateList.append(secondLine[row] if row < len(secondLine) else 0)
            stateList.append(thirdLine[row] if row < len(thirdLine) else 0)
        return stateList

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
        self.points = 0
        self.sum_history = []
        self.score_history = []
        self.game_stats = {'games_won': 0, 'overshot': 0, 'invalid_moves': 0}

    def startGame(self):
        self.state = GameState()
        self.agent.startGame()

    def isGameOver(self):
        return self.state.isGameOver()

    def doMove(self, diceValue: int):
        action = self.agent.doMove(diceValue)
        self.state.performAction(action, diceValue)

    def getGameResult(self) -> int:
        """
        :return: the sum achieved by the agent in this game or 0 if the agent performed an invalid action
        """
        if self.state.invalidActionPerformed:
            return 0
        else:
            return self.state.calcSum()

    def getPoints(self) -> int:
        """
        :return: the agent's league points
        """
        return self.points

    def getScores(self) -> List[float]:
        return self.score_history

    def getMeanScore(self) -> float:
        return np.mean(self.getScores())

    def finishGame(self, rank: int):
        """Finish game. Assigns score according to final rank.
        Parameters
        ----------
        rank : int
            rank == -2 if agent performed an invalid move
            rank == -1 if overshot target of 1000
            rank == 1 if win
            rank > 1 otherwise
        """
        s = self.getGameResult()
        if rank == -2:
            score = INVALID_MOVE_SCORE
            s = 0  # set sum to 0 because the value accumulated before invalid move is meaningless
        elif rank == -1:
            score = OVERSHOOT_SCORE
        else:
            score = s
        self.sum_history.append(s)
        self.score_history.append(score)
        if rank == -2:
            self.points += -9
            self.game_stats['invalid_moves'] += 1
        if rank == -1:
            self.points += -3
            self.game_stats['overshot'] += 1
        else:
            self.points += {1: 5, 2: 3, 3: 1}.get(rank, 0)
            if rank == 1:
                self.game_stats['games_won'] += 1


class Competition:
    def __init__(self, numberOfGames: int = 10000, randomSeed=42):
        self.numberOfGames = numberOfGames
        self.participants: List[Participant] = []
        self.rand = random.Random(randomSeed)

    def startCompetition(self):
        for i, game in enumerate(range(self.numberOfGames), start=1):
            log.info(f"Game #{i}")
            # init game
            for participant in self.participants:
                participant.startGame()

            # play game
            for gameRound in range(9):
                diceValue = self.rand.randint(1, 6)
                if not participant.isGameOver():
                    for participant in self.participants:
                        participant.doMove(diceValue)

            # evaluate game
            ranked_participants = []
            for participant in self.participants:
                status = participant.state.getGameStatus()
                if status == GameState.Status.OVER_INVALID_MOVE:
                    participant.finishGame(rank=-2)
                elif status == GameState.Status.OVER_OVERSHOOT:
                    participant.finishGame(rank=-1)
                else:
                    ranked_participants.append((participant, participant.state.calcSum()))
            ranked_participants = sorted(ranked_participants, key=lambda x: x[1], reverse=True)
            rank = 1
            prevScore = None
            for participant, score in ranked_participants:
                if prevScore is not None:
                    if score != prevScore:
                        rank += 1
                participant.finishGame(rank)
                prevScore = score

    def registerParticipant(self, agent: Agent):
        self.participants.append(Participant(agent))

    def printLeagueResult(self):
        self.participants = sorted(self.participants, key=lambda x: x.getPoints(), reverse=True)
        print("League results:")
        for rank, participant in enumerate(self.participants, start=1):
            print(f"Rank #{rank}\t {participant.agent.agentName:10s} {participant.getPoints():-6d} points\t{participant.game_stats}")

    def _plotAgentHistory(self, datafn):
        plt.figure()
        for participant in self.participants:
            plt.plot(datafn(participant), label=participant.agent.agentName)
        plt.legend()
        plt.show()

    def plotAgentSumHistory(self):
        self._plotAgentHistory(lambda participant: participant.sum_history)

    def plotCombinedSumHistogram(self):
        plt.figure()
        for participant in self.participants:
            plt.hist(participant.sum_history,
                label=participant.agent.agentName, 
                alpha=0.5, 
                bins=np.linspace(329, 1897, 15)) # aligns bin bound on 1000

        plt.vlines(x=1000, ymin=0, ymax=plt.ylim()[1], colors='black')
        plt.legend()
        plt.show()

    def plotParticipantSumHistograms(self):
        for p in self.participants:
            self.plotSumHistogramForParticipant(p)

    @staticmethod
    def plotSumHistogramForParticipant(p: Participant):
        plt.figure()
        if np.min(p.sum_history) == 0:  # agent used invalid moves
            bins = np.linspace(0, 1901, 20)
        else:
            bins = np.linspace(301, 1901, 17)  # true range is 333 to 999+900, modified to ensure 1000 is a bin boundary
        plt.hist(p.sum_history, color="grey", bins=bins)
        plt.vlines(x=1000, ymin=0, ymax=plt.ylim()[1], colors='black')
        plt.title(p.agent.agentName)
        plt.xlabel("score")
        plt.show()

    def printMeanScores(self):
        print("Mean scores achieved by the agents:")
        for rank, p in enumerate(sorted(self.participants, key=lambda x: x.getMeanScore(), reverse=True), start=1):
            print(f"#{rank}: {p.agent.agentName:10s} {p.getMeanScore()}")
