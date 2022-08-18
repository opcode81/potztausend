import logging
import sys
from time import time

from agents import RandomAgent, CalculatorAgent, QLearningAgent, TemporalDifferenceAgent, MonteCarloAgent
from agents.deep_rl_agent import A2CAgent, PPOAgent
from agents.util_agent import UtilAgent
from competition import Competition

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    num_games = 10000
    competition = Competition(num_games)

    randy = RandomAgent()
    sheldon = CalculatorAgent()
    paul = QLearningAgent()
    gunter = TemporalDifferenceAgent()
    lotte = MonteCarloAgent()
    jeremy = UtilAgent()
    a2c = A2CAgent(load=True)
    ppo = PPOAgent(load=True)
    competition.registerParticipant(randy)
    competition.registerParticipant(gunter)
    competition.registerParticipant(lotte)
    competition.registerParticipant(jeremy)
    competition.registerParticipant(sheldon)
    competition.registerParticipant(paul)
    competition.registerParticipant(a2c)
    competition.registerParticipant(ppo)

    start_time = time()
    competition.startCompetition()
    time_used = time() - start_time
    print(f'competition took {time_used} seconds. On average this is {num_games / time_used} games per second.\n')
    competition.printLeagueResult()
    print()
    competition.printMeanScores()
    competition.plotCombinedSumHistogram()

    for p in competition.participants:
        competition.plotSumHistogramForParticipant(p)

