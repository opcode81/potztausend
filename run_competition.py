from src import Competition
from src.agents import RandomAgent, CalculatorAgent
from time import time

if __name__ == '__main__':
    num_games = 10000
    competition = Competition(num_games)

    randy = RandomAgent()
    sheldon = CalculatorAgent()
    competition.registerParticipant(randy)
    competition.registerParticipant(sheldon)

    start_time = time()
    competition.startCompetition()
    time_used = time() - start_time
    print(f'competition took {time_used} seconds. On average this is {time_used / num_games} seconds.')
    competition.printResult()
    # competition.plot_score_history()
    competition.plot_histogram()
