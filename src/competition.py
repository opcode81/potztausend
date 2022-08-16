import logging
import random
from typing import List
import matplotlib.pyplot as plt
import numpy as np


from agents import MatrixColumn, Agent
epsilon = 1e-10
e_inf = 1.0/epsilon

log = logging.getLogger(__name__)

OVERSHOOT_SCORE = -1000


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
        self.invalid = False
        self.points = 0
        self.sum_history = []
        self.game_stats = {'games_won': 0, 'games_lost': 0, 'failed': 0}

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
                self.points = 0

    def getGameResult(self) -> int:
        return self.state.calcSum()

    def getPoints(self) -> int:
        return self.points

    def getScores(self) -> List[float]:
        return [OVERSHOOT_SCORE if s > 1000 else s for s in self.sum_history]

    def getMeanScore(self) -> float:
        return np.mean(self.getScores())

    def finishGame(self, rank: int):
        """Finish game. Assigns score according to final rank.
        Parameters
        ----------
        rank : int
            rank == -1 if overbought
            rank == 1 if win
            rank > 1 else
        """
        self.sum_history.append(abs(self.getGameResult()))
        if rank == -1:
            self.points += -3
            self.game_stats['failed'] += 1
        else:
            self.points += {1: 5, 2: 3, 3: 1}.get(rank, 0)
            if rank == 1:
                self.game_stats['games_won'] += 1
            else:
                self.game_stats['games_lost'] += 1


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

                for participant in self.participants:
                    participant.doMove(diceValue)

            # evaluate game
            ranked_participants = []
            for participant in self.participants:
                result = participant.getGameResult()
                if result > 1000:
                    participant.finishGame(rank=-1)
                else:
                    ranked_participants.append((participant, result))
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
        self.participants = sorted(self.participants, key=lambda x: x.getPoints() - x.invalid * e_inf, reverse=True)
        print("League results:")
        for rank, participant in enumerate(self.participants, start=1):
            if participant.invalid:
                print(f"\t {participant.agent.agentName}: disqualified")
            else:
                print(f"Rank #{rank}\t {participant.agent.agentName:10s} {participant.getPoints():-6d} points\tgames won: {participant.game_stats['games_won']}, failed: {participant.game_stats['failed']}")

    def plot_score_history(self):
        plt.figure()
        for participant in self.participants:
            plt.plot(participant.sum_history, label=participant.agent.agentName)
        plt.legend()
        plt.show()

    def plot_histogram(self):
        plt.figure()
        for participant in self.participants:
            plt.hist(participant.sum_history,
                label=participant.agent.agentName, 
                alpha=0.5, 
                bins=np.linspace(329, 1897, 15)) # aligns bin bound on 1000
        plt.vlines(x=1000, ymin=0, ymax=plt.ylim()[1], colors='black')
        plt.legend()
        plt.show()

    @staticmethod
    def plotScoreHistoryForParticipant(p: Participant):
        plt.figure()
        plt.hist(p.sum_history, color="grey", bins=np.linspace(329, 1897, 15))
        plt.vlines(x=1000, ymin=0, ymax=5000, colors='black')
        plt.title(p.agent.agentName)
        plt.xlabel("score")
        plt.show()

    def printMeanScores(self):
        print("Mean scores achieved by the agents:")
        for rank, p in enumerate(sorted(self.participants, key=lambda x: x.getMeanScore(), reverse=True), start=1):
            print(f"#{rank}: {p.agent.agentName:10s} {p.getMeanScore()}")
