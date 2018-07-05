#import random
#import logging
#import typing; from typing import *
#import itertools
#from itertools import product
#from sample_players import null_score, open_move_score, improved_score

class SearchTimeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player) -> float:
# This dynamic heuristic attempts to use domain-specific info of open squares
# to measure which phase in the game we are at.
# Emprical results show an average of around 14 open squares at game end, 
# compared to 49 at game start. This can be simplied to 
# 14/49=3.5 at end game, 1.0 at start    
# For this version we use this ratio to 
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    count_total_positions = float(game.height * game.width)
    count_empty_coords = float(len(game.get_blank_spaces()))
    #w=count_empty_coords/count_total_positions
    k=count_total_positions/count_empty_coords

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return own_moves - k*opp_moves

def custom_score_2(game, player) -> float:

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    count_total_positions = float(game.height * game.width)
    count_empty_coords = float(len(game.get_blank_spaces()))
    #w=count_empty_coords/count_total_positions
    k=count_total_positions/count_empty_coords

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return k*own_moves - opp_moves
    
    #return ((1-w)*own_moves - w*opp_moves)**2

def custom_score_3(game, player) -> float:

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    count_total_positions = float(game.height * game.width)
    count_empty_coords = float(len(game.get_blank_spaces()))
    #w=count_empty_coords/count_total_positions
    k=count_total_positions/count_empty_coords

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return k**2*own_moves - opp_moves




class IsolationPlayer:

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):

    def __init__(self,search_depth=3,score_fn=custom_score,timeout=80):
        super(MinimaxPlayer,self).__init__(search_depth=search_depth, score_fn=score_fn, timeout=timeout)
        self.schedule=[]

    def get_move(self, game, time_left):

        legal_moves=game.get_legal_moves()
        if not legal_moves:
            return (-1,-1)

        best_move = legal_moves[0]
        
        if time_left() <= 10:
            return best_move

        try:
            self.time_left = time_left

            best_move = self.minimax(game, depth=self.search_depth,is_maximizer=True)

        except SearchTimeout:

            pass

        return best_move

    def minimax(self, game, depth):#, is_maximizer=True):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        is_maximizer= (game.active_player == self)

        legal_moves = game.get_legal_moves(game.active_player)

        if not legal_moves:
            current_player = game.active_player if is_maximizer else game.inactive_player
            return game.utility(current_player), (-1,-1)
        
        best_move=legal_moves[0]
        best_score = float('-inf') if is_maximizer else float('inf')
        current_player = game.active_player if is_maximizer else game.inactive_player

        if depth == 0:
            return self.score(game, current_player), best_move

        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = self.minimax(next_state, depth - 1)#, not is_maximizer)

            
            if is_maximizer:
                if score > best_score:
                    best_score, best_move = score, move
            else:
                if score < best_score:
                    best_score, best_move = score, move

        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    
    def __init__(self,search_depth=3,score_fn=custom_score,timeout=80):
        super(AlphaBetaPlayer,self).__init__(search_depth=search_depth, score_fn=score_fn, timeout=timeout)
        #This controls number of iterations. set to 1 to disable ID
        iterations=1
        self.iterations=iterations
        assert self.TIMER_THRESHOLD == timeout
    def get_move(self, game, time_left):        

        legal_moves=game.get_legal_moves()
        if not legal_moves:
            return (-1,-1)
        best_move=legal_moves[0]

        if time_left() <= 5:
            return best_move

        self.time_left = time_left

        for d in range(1,100):

#        for d in range(self.search_depth,self.search_depth+self.iterations):
            try:
                _, best_move = self.alphabeta(game, depth=d)#,is_maximizer=True)

            except SearchTimeout:
                break

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):#, is_maximizer=True):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        is_maximizer= (game.active_player == self)
            

        best_score = float('-inf') if is_maximizer else float('inf')
        current_player = game.active_player if is_maximizer else game.inactive_player
        legal_moves = game.get_legal_moves(game.active_player)

        if not legal_moves:
            return game.utility(current_player), (-1,-1)
        
        best_move=legal_moves[0]
        if depth == 0:
            return self.score(game, current_player), best_move

        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = self.alphabeta(next_state, depth - 1, alpha, beta)#, not is_maximizer)

            if is_maximizer:
                if score > best_score:
                    best_score, best_move = score, move

                    if best_score >= beta:
                        break
                    alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score, best_move = score, move

                    if best_score <= alpha:
                        break
                    beta = min(beta, best_score)

        return best_score


