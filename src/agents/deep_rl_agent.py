import pickle
import os

import gym
import numpy as np
from stable_baselines3 import A2C

import competition
from agents import MatrixColumn, Agent
from competition import GameState, Competition


class Env(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self):
        self.state = GameState()
        self.diceValue = 0
        self.observation_space = gym.spaces.Box(0, 1.0, shape=[7])
        self.rand = np.random.RandomState()
        self.action_space = gym.spaces.Discrete(3)

    def get_obs(self, diceValue=None):
        if diceValue is None:
            diceValue = self.rand.randint(1, 7)
        self.diceValue = diceValue
        remainingActionsPerCol = [(3 - len(self.state.stateLines[c])) / 3 for c in MatrixColumn]
        numNorm = 1000
        columnValues = [self.diceValue/numNorm, self.diceValue*10/numNorm, self.diceValue*100/numNorm]
        return np.array([*columnValues, self.state.calcSum()/numNorm, *remainingActionsPerCol])

    def reset(self):
        self.state = GameState()
        return self.get_obs()

    def step(self, action: int):
        actionCol = MatrixColumn(action)
        self.state.performAction(actionCol, self.diceValue)
        status = self.state.getGameStatus()
        if status == GameState.Status.OVER_INVALID_MOVE:
            reward = competition.INVALID_MOVE_SCORE
            done = True
        elif status == GameState.Status.OVER_OVERSHOOT:
            reward = competition.OVERSHOOT_SCORE
            done = True
        elif status == GameState.Status.OVER_SUCCESS:
            reward = self.state.calcSum()
            done = True
        elif status == GameState.Status.RUNNING:
            reward = 10  # small positive reward for having reached the next round
            done = False
        else:
            raise ValueError()
        info = {}
        return self.get_obs(), reward, done, info

    def render(self, mode="ansi"):
        if mode == "ansi":
            return str(self.state)


class DeepRLAgent(Agent):
    def __init__(self, load=False):
        self.env = Env()
        self.path = os.path.join("model_resources", "deep_rl", "a2c.zip")
        self.model = A2C('MlpPolicy', self.env, verbose=1)
        if load:
            self.model = self.model.load(self.path)
        super().__init__(self.__class__.__name__)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self):
        self.model.save(self.path)

    def startGame(self):
        self.env.reset()

    def doMove(self, diceValue) -> MatrixColumn:
        action, _ = self.model.predict(self.env.get_obs(diceValue), deterministic=True)
        self.env.step(action)
        return MatrixColumn(action)


if __name__ == '__main__':
    agent = DeepRLAgent()
    agent.train(1000000)
    #agent.save()

    comp = Competition(numberOfGames=1000)
    comp.registerParticipant(agent)
    comp.startCompetition()

    comp.printMeanScores()
    comp.printLeagueResult()
    comp.plotParticipantSumHistograms()
