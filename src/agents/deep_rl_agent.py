import gym
import numpy as np
from stable_baselines3 import A2C, DQN

import competition
from agents import MatrixColumn, Agent
from competition import GameState, Competition


class Env(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self):
        self.state = GameState()
        self.diceValue = 0
        self.observation_space = gym.spaces.Box(0, 1.0, shape=[5])
        self.rand = np.random.RandomState()
        self.action_space = gym.spaces.Discrete(3)

    def get_obs(self, diceValue=None):
        if diceValue is None:
            diceValue = self.rand.randint(1, 7)
        self.diceValue = diceValue
        possibleActionsBool = [1 if a in self.state.getPossibleActions() else 0 for a in MatrixColumn]
        #return {"diceValue": self.diceValue, "sum": self.state.calcSum(), "possibleActions": possibleActionsBool}
        return np.array([self.diceValue/6, self.state.calcSum()/1000, *possibleActionsBool])

    def reset(self):
        self.state = GameState()
        return self.get_obs()

    def step(self, action: int):
        actionCol = MatrixColumn(action)
        done = False
        reward = 0
        possibleActions = self.state.getPossibleActions()
        if actionCol not in possibleActions:  # inadmissible action, apply negative reward and ignore action
            reward = 10 * competition.OVERSHOOT_SCORE
            done = True
        else:
            self.state.performAction(actionCol, self.diceValue)
            s = self.state.calcSum()
            if s > 1000:
                reward = competition.OVERSHOOT_SCORE
                done = True
            elif len(possibleActions) == 0:
                reward = s
                done = True
        info = {}
        return self.get_obs(), reward, done, info

    def render(self, mode="ansi"):
        if mode == "ansi":
            return str(self.state)


class DeepRLAgent(Agent):
    def __init__(self):
        self.env = Env()
        self.model = DQN('MlpPolicy', self.env, verbose=1)
        super().__init__(self.__class__.__name__)

    def train(self):
        self.model.learn(total_timesteps=1000000)

    def startGame(self):
        self.env.reset()

    def doMove(self, diceValue) -> MatrixColumn:
        action, _ = self.model.predict(self.env.get_obs(diceValue), deterministic=True)
        self.env.step(action)
        return MatrixColumn(action)


if __name__ == '__main__':
    agent = DeepRLAgent()
    agent.train()

    comp = Competition(numberOfGames=1000)
    comp.registerParticipant(agent)
    comp.startCompetition()

    comp.printMeanScores()

