from src import Competition
from src.agents import RandomAgent

if __name__ == '__main__':
    competition = Competition()

    randy = RandomAgent()
    competition.registerParticipant(randy)

    competition.startCompetition()
    competition.printResult()
