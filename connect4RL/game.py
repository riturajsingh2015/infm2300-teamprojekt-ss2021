from agent_RL import *
from abc import abstractmethod
import matplotlib.pyplot as plt


class Game:

    def __init__(self, player1, player2, exp1=1, exp2=1, tag1=1, tag2=2):

        self.players = {1: player1(tag1, epsilon=exp1),
                        2: player2(tag2, epsilon=exp2)}

        self.state, self.winner, self.turn = self.init_game()
        self.memory = {}
        self.p1_wins = []
        self.p2_wins = []
        self.draws = []
        self.p1_win = 0
        self.p2_win = 0
        self.draw = 0
        self.episodes = 100


    def play_game(self):

        move_count = 0
        while self.winner is None:
            self.game_winner()
            move = self.play_move()

            if self.winner is None:
                self.state = self.make_state_from_move(move)
            
            self.next_player()
            move_count += 1

        self.play_move()
        self.next_player()
        #self.play_move()
        #self.next_player()

        return self.winner, move_count

    def play_move(self):
        player = self.players[self.turn]
        move = player.choose_move(self.state, self.winner)
        if isinstance(player, TicRLAgent):
            player.qLearning(self.state.tobytes(), move, self.winner)
        return move

    def play_multiple_games(self, episodes):
        
        q_counter = 0
        statistics = {1: 0, 2: 0, 0: 0, 'move_count': 0}
        move_count_total = []
        for i in range(episodes):
            self.episodes = i + 1
            winner, move_count = self.play_game()
            move_count_total.append(move_count)
            statistics[winner] = statistics[winner] + 1

            self.state, self.winner, self.turn = self.init_game()

        plt.ylabel('Game outcomes in %')
        plt.xlabel('Game number')
        plt.plot(range(episodes), self.draws, 'r-', label='Draw')
        plt.plot(range(episodes), self.p1_wins, 'g-', label='Player 1 wins')
        plt.plot(range(episodes), self.p2_wins, 'b-', label='Player 2 wins')
        plt.legend(loc='best', shadow=True, fancybox=True, framealpha =0.7)
        #plt.plot(count, best_actions, 'b-', label='Best next action')
        plt.show()
        
        countQValues = 0
        for key in self.players[1].Q: 
            Q = self.players[1].Q
            # print(Q[key])
            for i in Q[key]:
                if i != 0.:
                    countQValues += 1
            
        print("% of all actions used and associated states: " + str((countQValues/(len(self.players[1].Q)*7))*100) + "%")
            
        return statistics

    @abstractmethod
    def init_game(self):
        pass

    @abstractmethod
    def make_state_from_move(self, move):
        pass

    @abstractmethod
    def next_player(self):
        pass

    @abstractmethod
    def game_winner(self):
        pass

    @abstractmethod
    def print_game(self):
        pass
