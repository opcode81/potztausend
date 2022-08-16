import logging
import sys
from time import time

from competition import Competition
from agents import RandomAgent, CalculatorAgent, QLearningAgent, TemporalDifferenceAgent, MonteCarloAgent
from agents.util_agent import UtilAgent

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
    competition.registerParticipant(randy)
    competition.registerParticipant(gunter)
    competition.registerParticipant(lotte)
    competition.registerParticipant(jeremy)
    competition.registerParticipant(sheldon)
    competition.registerParticipant(paul)

    start_time = time()
    competition.startCompetition()
    time_used = time() - start_time
    print(f'competition took {time_used} seconds. On average this is {time_used / num_games} seconds.')
    competition.printLeagueResult()
    # competition.plot_score_history()
    competition.plot_histogram()
    competition.printMeanScores()

    for p in competition.participants:
        competition.plotScoreHistoryForParticipant(p)

