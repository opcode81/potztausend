import logging
import os
from abc import ABC, abstractmethod

import gym
import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

import competition
from agents import MatrixColumn, Agent
from competition import GameState, Competition

log = logging.getLogger(__name__)


class Env(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self):
        self.state = GameState()
        self.diceValue = 0
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=[8])
        self.rand = np.random.RandomState()
        self.action_space = gym.spaces.Discrete(3)

    def get_obs(self, diceValue=None):
        if diceValue is None:
            diceValue = self.rand.randint(1, 7)
        self.diceValue = diceValue
        remainingActionsPerCol = [(3 - len(self.state.stateLines[c])) / 3 for c in MatrixColumn]
        target = 1000
        s = self.state.calcSum()
        resultingValues = [s + self.diceValue, s + self.diceValue*10, s + self.diceValue*100]
        resultingValuesNorm = [(x - target) / target for x in resultingValues]
        return np.array([s/target, diceValue/6, *resultingValuesNorm, *remainingActionsPerCol])

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


class DeepRLAgent(Agent, ABC):
    def __init__(self, load: bool, filebasename: str, useFallbackForForbiddenActions=False):
        """
        :param load: whether to load a previously stored model
        :param filebasename: the base filename for storage
        :param useFallbackForForbiddenActions: whether to use, in cases where the policy selects an inadmissible action,
            the policy's probabilistic outputs to select a fallback action (the most probable admissible action).
            This option is useful for agents that have not been trained a lot; it will allow them to achieve better
            performance.
        """
        self.useFallbackForForbiddenActions = useFallbackForForbiddenActions
        self.env = Env()
        self.path = os.path.join("model_resources", "deep_rl", f"{filebasename}.zip")
        self.model = self._createModel(self.env)
        if load:
            self.model = self.model.load(self.path)
        super().__init__(self.__class__.__name__)

    @abstractmethod
    def _createModel(self, env: Env) -> BaseAlgorithm:
        pass

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self):
        self.model.save(self.path)

    def startGame(self):
        self.env.reset()

    def doMove(self, diceValue) -> MatrixColumn:
        obs = self.env.get_obs(diceValue)
        if self.useFallbackForForbiddenActions:
            policy = self.model.policy
            obs, vectorized_env = policy.obs_to_tensor(obs)
            dist = self.model.policy.get_distribution(obs)
            actions = list(MatrixColumn)
            probs = dist.distribution.probs.tolist()[0]
            ap = sorted(zip(actions, range(len(actions)), probs), key=lambda x: -x[2])
            possibleActions = self.env.state.getPossibleActions()
            for action, actionIdx, p in ap:
                if action in possibleActions:
                    break
        else:
            actionIdx, _ = self.model.predict(obs, deterministic=True)
            action = MatrixColumn(actionIdx)
        self.env.step(actionIdx)
        return action


class A2CAgent(DeepRLAgent):
    def __init__(self, load=False, useFallbackForForbiddenActions=False):
        super().__init__(load, "a2c", useFallbackForForbiddenActions=useFallbackForForbiddenActions)

    def _createModel(self, env) -> BaseAlgorithm:
        return A2C('MlpPolicy', env, verbose=1)


class DQNAgent(DeepRLAgent):
    def __init__(self, load=False):
        super().__init__(load, "dqn")

    def _createModel(self, env) -> BaseAlgorithm:
        return DQN('MlpPolicy', env, verbose=1)


class PPOAgent(DeepRLAgent):
    def __init__(self, load=False):
        super().__init__(load, "ppo")

    def _createModel(self, env) -> BaseAlgorithm:
        return PPO('MlpPolicy', env, verbose=1)


if __name__ == '__main__':
    logging.basicConfig()

    #agent = DQNAgent()
    agent = A2CAgent()
    #agent = PPOAgent()
    #agent = A2CAgent(load=True)

    agent.train(3000000)
    #agent.save()

    comp = Competition(numberOfGames=1000)
    comp.registerParticipant(agent)
    comp.startCompetition()

    comp.printMeanScores()
    comp.printLeagueResult()
    comp.plotParticipantSumHistograms()
