import os
import random
from typing import Callable

from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QGroupBox

from src.agents import MonteCarloAgent, TemporalDifferenceAgent, QLearningAgent, CalculatorAgent, RandomAgent
from src.agents.util_agent import UtilAgent
from src.competition import Participant


class PotztausendApp(QWidget):
    def __init__(self):
        super().__init__()
        participants = [Participant(TemporalDifferenceAgent()),
                        Participant(MonteCarloAgent()),
                        Participant(QLearningAgent()),
                        Participant(UtilAgent()),
                        Participant(CalculatorAgent()),
                        Participant(RandomAgent())]
        layout = QVBoxLayout()
        layout.addWidget(self.ControlWidget(lambda diceValue: self._diced(diceValue), self._startGame))
        self.agentWidgets = []

        for p in participants:
            agentWidget = self.AgentWidget(p)
            self.agentWidgets.append(agentWidget)
            layout.addWidget(agentWidget)

        self.setLayout(layout)
        self.resize(300, 150)
        self.setWindowTitle("Potztausend")
        self.setWindowIcon(QIcon(os.path.join("app_resources", "fuenf.png")))

    def _diced(self, value):
        for agentView in self.agentWidgets:
            agentView.participant.doMove(value)
            agentView.updateAgent()

    def _startGame(self):
        for agentView in self.agentWidgets:
            agentView.participant.startGame()
            agentView.updateAgent()

    class AgentWidget(QWidget):

        def __init__(self, participant):
            super().__init__()
            self.participant = participant

            layout = QVBoxLayout()
            groupBox = QGroupBox(participant.agent.agentName)
            gridLayout = QGridLayout()
            gridLayout.addWidget(QLabel("Score:"), 0, 0)
            self.scoreLabel = QLabel("0")
            gridLayout.addWidget(self.scoreLabel, 1, 0)
            gridLayout.addWidget(QLabel("State:"), 0, 1)
            self.stateLabel = QLabel(str(participant.state))
            gridLayout.addWidget(self.stateLabel, 1, 1)
            groupBox.setLayout(gridLayout)
            layout.addWidget(groupBox)
            self.setLayout(layout)

        def updateAgent(self) -> None:
            self.scoreLabel.setText(str(self.participant.getGameResult()))
            self.stateLabel.setText(str(self.participant.state))
            self.update()

    class ControlWidget(QWidget):
        diceImages = {1: os.path.join("app_resources", "eins.png"),
                      2: os.path.join("app_resources", "zwei.png"),
                      3: os.path.join("app_resources", "drei.png"),
                      4: os.path.join("app_resources", "vier.png"),
                      5: os.path.join("app_resources", "fuenf.png"),
                      6: os.path.join("app_resources", "sechs.png")}

        def __init__(self, diceCallback: Callable[[int], None], startGameCallback: Callable[[], None]):
            super().__init__()

            self.dicedCount = 0
            self.dicedValue = None

            self.diceCallback = diceCallback
            self.startGameCallback = startGameCallback
            layoutDice = QHBoxLayout()

            #dice
            self.picLabel = QLabel(self)
            pixmap = QPixmap(self.diceImages.get(1)).scaled(150, 150)
            self.picLabel.setPixmap(pixmap)
            self.picLabel.resize(pixmap.width(), pixmap.height())

            self.playerWidget = self.PlayerWidget(self._playerMoved)
            layoutDice.addWidget(self.playerWidget)

            buttonWidget = QWidget()
            buttonLayout = QGridLayout()
            self.startButton = QPushButton("Start game")
            self.startButton.clicked.connect(self._startGame)
            self.diceButton = QPushButton("Dice")
            self.diceButton.setEnabled(False)
            self.diceButton.clicked.connect(self._dice)
            buttonLayout.addWidget(self.picLabel, 0, 0)
            buttonLayout.addWidget(self.startButton, 1, 0)
            buttonLayout.addWidget(self.diceButton, 2, 0)
            buttonWidget.setLayout(buttonLayout)

            layoutDice.addWidget(buttonWidget)

            self.setLayout(layoutDice)

        def _startGame(self):
            self.startGameCallback()
            self.diceButton.setEnabled(True)
            self.playerWidget.resetPlayer()
            self.dicedCount = 0

        def _dice(self):
            self.dicedValue = random.randint(1, 6)
            self.dicedCount += 1
            if self.dicedCount == 9:
                self.diceButton.setEnabled(False)
            pixmap = QPixmap(self.diceImages.get(self.dicedValue)).scaled(150, 150)
            self.picLabel.setPixmap(pixmap)
            self.playerWidget.updateDiceValue(self.dicedValue)

        def _playerMoved(self):
            self.diceCallback(self.dicedValue)

        class PlayerWidget(QWidget):
            def __init__(self, movedCallback: Callable[[], None]):
                super().__init__()
                self.movedCallback = movedCallback
                layout = QVBoxLayout()
                gameField = QWidget()
                gameFieldLayout = QGridLayout()
                self.fields = []

                for i in range(0, 9):
                    field = self.FieldButton(i, lambda index: self._buttonClicked(index))
                    if i < 3:
                        x = 0
                    elif i < 6:
                        x = 1
                    else:
                        x = 2
                    y = i % 3
                    gameFieldLayout.addWidget(field, x, y)
                    self.fields.append(field)
                gameField.setLayout(gameFieldLayout)

                layout.addWidget(gameField)
                layout.addWidget(QLabel("Score:"))
                self.scoreLabel = QLabel()
                self.score = 0
                layout.addWidget(self.scoreLabel)
                self.setLayout(layout)
                self.diceValue = None

            def resetPlayer(self):
                self.diceValue = None
                self.score = 0
                self.scoreLabel.setText("")
                for field in self.fields:
                    field.setText("")
                    field.setEnabled(True)

            def updateDiceValue(self, diceValue) -> None:
                self.diceValue = diceValue

            def _buttonClicked(self, index):
                if self.diceValue is not None:
                    self.fields[index].setText(str(self.diceValue))
                    self.fields[index].setEnabled(False)
                    if index % 3 == 0:
                        multiplier = 100
                    elif index % 2 == 1:
                        multiplier = 10
                    else:
                        multiplier = 1
                    self.score += self.diceValue * multiplier
                    self.scoreLabel.setText(str(self.score))
                    self.diceValue = None
                    self.movedCallback()

            class FieldButton(QPushButton):
                def __init__(self, index, clickedCallback: Callable[[int], None]):
                    super().__init__()
                    self.clickedCallback = clickedCallback
                    self.index = index
                    self.setFixedSize(50, 50)
                    self.setFont(QFont('Times', 20))
                    self.clicked.connect(self._buttonClicked)

                def _buttonClicked(self):
                    self.clickedCallback(self.index)
                    self.clearFocus()

