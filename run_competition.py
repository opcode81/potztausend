from src import Competition
from src.agents import RandomAgent, CalculatorAgent

if __name__ == '__main__':
    competition = Competition()

    randy = RandomAgent()
    sheldon = CalculatorAgent()
    competition.registerParticipant(randy)
    competition.registerParticipant(sheldon)

    competition.startCompetition()
    competition.printResult()
    # competition.plot_score_history()
    competition.plot_histogram()
