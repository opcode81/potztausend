import logging
from pprint import pprint
import sys

import numpy as np

from src import Competition
from src.agents import RandomAgent, CalculatorAgent, QLearningAgent, TemporalDifferenceAgent, ValueIterationAgent
from time import time

from src.agents.util_agent import UtilAgent


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout)
    num_games = 1000
    competition = Competition(num_games)

    randy = RandomAgent()
    sheldon = CalculatorAgent()
    paul = QLearningAgent()
    gunter = TemporalDifferenceAgent()
    lotte = ValueIterationAgent()
    jeremy = UtilAgent()
    #competition.registerParticipant(randy)
    competition.registerParticipant(sheldon)
    competition.registerParticipant(paul)
    #competition.registerParticipant(gunter)
    #competition.registerParticipant(lotte)
    competition.registerParticipant(jeremy)

    start_time = time()
    competition.startCompetition()
    time_used = time() - start_time
    print(f'competition took {time_used} seconds. On average this is {time_used / num_games} seconds.')
    competition.printResult()
    # competition.plot_score_history()
    competition.plot_histogram()

    mean_scores = {p.agent.agentName: np.mean([-1000 if s > 1000 else s for s in p.score_history]) for p in competition.participants}
    pprint(mean_scores, width=1)