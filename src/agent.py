from abc import ABC, abstractmethod
from enum import Enum


class MatrixColumn(Enum):
    LEFT_COLUMN = 0
    MIDDLE_COLUMN = 1
    RIGHT_COLUMN = 2


class Agent(ABC):
    """
    Abstract base class for an agent which solves the `Potztausend` game
    """

    def __init__(self, agentName: str):
        """
        :param agentName: name of the agent
        """
        self.agentName = agentName

    @abstractmethod
    def startGame(self):
        """
        resets the internal game state of the agent
        """
        pass

    @abstractmethod
    def doMove(self, diceValue) -> MatrixColumn:
        """
        :param diceValue: value between 1 and 6 (inclusive) which was diced this round
        :returns the column in which the dice value was placed
        """
        pass
