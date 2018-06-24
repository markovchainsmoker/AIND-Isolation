# -*- coding: utf-8 -*-

import random
import logging
import typing; from typing import *
import itertools
from itertools import product
from sample_players import null_score, open_move_score, improved_score

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def get_move_difference_factor(game, player) -> float:
    count_own_moves = len(game.get_legal_moves(player))
    count_opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return (count_own_moves - count_opp_moves)

def get_center_available_factor(game, player) -> float:
    own_moves = game.get_legal_moves(player)
    center_x, center_y = game.width / 2, game.height / 2
    center_available = -1
    # Center of grid is only available when odd width and odd height
    if not center_x.is_integer() and not center_y.is_integer():
        center_coords = (int(center_x), int(center_y))
        center_available = own_moves.index(center_coords) if center_coords in own_moves else -1
    # Next move should always be to center square if available
    return 2.0 if (center_available != -1) else 1.0

def is_empty_board(count_total_positions, count_empty_coords):
    all_empty = True if (count_total_positions == count_empty_coords) else False
    if all_empty:
        return 1.0

def get_reflection_available_factor(game, player) -> float:
    count_total_positions = game.height * game.width
    count_empty_coords = len(game.get_blank_spaces())

    # Return if no reflection move possible before first move
    if is_empty_board(count_total_positions, count_empty_coords):
        return 1.0

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    count_own_moves = len(game.get_legal_moves(player))
    count_opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    all_coords = list(itertools.product(range((game.width)), range((game.height))))
    player_coords = (player_x, player_y) = game.get_player_location(player)
    opp_coords = (opp_x, opp_y) = game.get_player_location(game.get_opponent(player))
    player_index = all_coords.index(player_coords)
    opp_index = all_coords.index(opp_coords)
    mirrored_all_coords = all_coords[::-1]
    mirrored_player_coords = mirrored_all_coords[player_index]
    mirrored_opp_coords = mirrored_all_coords[opp_index]

    # Return high Reflection Available Factor if the mirror coords that
    # correspond to the oppositions current coords is an available legal move for current player
    for legal_player_move_coords in own_moves:
        if legal_player_move_coords == mirrored_opp_coords:
            return 2.0
    return 1.0

def get_partition_possible_factor(game, player):
    count_total_positions = game.height * game.width
    count_empty_coords = len(game.get_blank_spaces())

    empty_coords = game.get_blank_spaces()

    # Return if no partition possible before first move
    if is_empty_board(count_total_positions, count_empty_coords):
        return 1.0

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    for move in own_moves:
        cell_left = (move[0]-1, move[1])
        cell_right = (move[0]+1, move[1])
        cell_below = (move[0], move[1]-1)
        cell_above = (move[0], move[1]+1)

        cell_left_x2 = (move[0]-2, move[1])
        cell_right_x2 = (move[0]+2, move[1])
        cell_below_x2 = (move[0], move[1]-2)
        cell_above_x2 = (move[0], move[1]+2)

        is_cell_left = cell_left not in empty_coords
        is_cell_right = cell_right not in empty_coords
        is_cell_below = cell_below not in empty_coords
        is_cell_above = cell_above not in empty_coords

        is_cell_left_x2 = cell_left_x2 not in empty_coords
        is_cell_right_x2 = cell_right_x2 not in empty_coords
        is_cell_below_x2 = cell_below_x2 not in empty_coords
        is_cell_above_x2 = cell_above_x2 not in empty_coords

        # Firstly check if two cells in sequence on either side of possible move
        # If so give double bonus points
        if ( (is_cell_left and is_cell_left_x2) or
             (is_cell_right and is_cell_right_x2) or
             (is_cell_below and is_cell_below_x2) or
             (is_cell_above and is_cell_above_x2) ):
            return 4.0

        # Secondly check if just one cell surrounding possible move
        if (is_cell_left or
            is_cell_right or
            is_cell_below or
            is_cell_above):
            return 2.0

    return 1.0

def custom_score(game, player) -> float:
    """ heuristic_1_center
    Evaluation function outputs a
    score equal to the Center Available Factor
    that has higher weight when center square still available on any move

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    center_available_factor = get_center_available_factor(game, player)

    # Heuristic score output
    return float(center_available_factor)

def custom_score_2(game, player) -> float:
    """ heuristic_2_reflection
    Heuristic 2's Reflection Available Factor
    has higher weight when reflection of opposition player
    position is available on other side of board.
    i.e. In game tree, for all available opposition in coordinates,
    count how many available reflection moves (on opposite side of board)
    are available as a legal moves for the current player. These should result in
    higher weight if available

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    reflection_available_factor = get_reflection_available_factor(game, player)

    return float(reflection_available_factor)

def custom_score_3(game, player) -> float:
    """ heuristic_3_partition
    Heuristic 3's Partition Growth Factor
    has higher weight when available moves are
    vertically or horizontally (not diagonally) adjacent
    to a sequence of one or two blocked locations

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    partition_possible_factor = get_partition_possible_factor(game, player)

    return float(partition_possible_factor)



class IsolationPlayer:

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):

    def __init__(self,search_depth=3,score_fn=custom_score,timeout=10):
        super().__init__(search_depth=search_depth, score_fn=score_fn, timeout=timeout)
        self.schedule=[]

    def get_move(self, game, time_left):
        best_move = (-1, -1)
        
        self.time_left = time_left
        if self.time_left() <= 0.1:
            logging.warning("[get_move] - Terminated due to no time")
            return best_move
        legal_moves=game.get_legal_moves()
        if not legal_moves:
            logging.debug("[get_move] - Terminated due to no remaining legal moves")
            return best_move

        try:
            logging.debug("[get_move] - Performing Fixed-Depth Search to depth %r: ", self.search_depth)
            # logging.debug("Time left is: %r", self.time_left())
          # logging.debug(game.to_string())
            _, best_move = self.minimax(game, depth=self.search_depth,is_maximizer=True)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            logging.warning("[get_move] - Terminated due to no remaining time")

            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, is_maximizer=True):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Initialise variable for no legal moves
        best_score = float('-inf') if is_maximizer else float('inf')
        current_player = game.active_player if is_maximizer else game.inactive_player
        legal_moves = game.get_legal_moves(game.active_player)

        logging.debug("Current player is Maximizing: %r", is_maximizer)
        logging.debug("Current depth: %r", depth)
        logging.debug("Best utility: %r", best_score)
        logging.debug("Remaining legal moves: %r", legal_moves)

        # Recursion function termination conditions when legal moves exhausted or no plies left
        if not legal_moves:
            logging.debug("Recursion terminated due to no remaining legal moves")
            best_move = (-1, -1)
            return game.utility(current_player), best_move
        else:
            best_move=legal_moves[0]

        if depth == 0:
            logging.debug("Recursion terminated due to no more plies to search")
            best_move=legal_moves[0]
            return self.score(game, current_player), best_move

        # Recursively alternate between Maximise and Minimise calculations for decrementing depths
        for move in legal_moves:
            # logging.debug("Recursion with time left is: %r", self.time_left())
            logging.debug("Recursion with move: %r", move)
            logging.debug("Best utility: %r", best_score)
            logging.debug("Best move: %r", best_move)

            # Obtain successor of current state by creating copy of board and applying a move.
            next_state = game.forecast_move(move)
            score, _ = self.minimax(next_state, depth - 1, not is_maximizer)
            logging.debug("Forecast utility: %r", score)

            
            if is_maximizer:
                logging.debug("Checking move with Maximising player, score > best_score? : %r", (score > best_score))
                if score > best_score:
                    best_score, best_move = score, move
                    #best_score, best_move = max((forecast_score, move),(best_score, best_move))
            else:
                logging.debug("Checking move with Minimising player, score < best_score? : %r", (score > best_score))
                if score < best_score:
                    best_score, best_move = score, move

        return best_score, best_move

class AlphaBetaPlayer(IsolationPlayer):

    def __init__(self,search_depth=3,score_fn=custom_score,timeout=10):
        super().__init__(search_depth=search_depth, score_fn=score_fn, timeout=timeout)
        self.schedule=[]
        assert self.TIMER_THRESHOLD == timeout
#        logging.warning("[{}] with depth {} and timeout {}".format(self,self.search_depth,self.TIMER_THRESHOLD))
    def get_move(self, game, time_left):
        #best_move = (-1, -1)
        
        self.time_left = time_left
        if self.time_left() <= 0.1:
            logging.warning("[get_move] - Terminated due to no remaining time")
            return best_move

        legal_moves=game.get_legal_moves()
        if not legal_moves:
            best_move=(-1,-1)
            logging.debug("[get_move] - Terminated due to no remaining legal moves")
            return best_move

        try:

            logging.debug("[get_move] - Performing Fixed-Depth Search to depth %r: ", self.search_depth)
            # logging.debug("Time left is: %r", self.time_left())
          # logging.debug(game.to_string())
            _, best_move = self.alphabeta(game, depth=self.search_depth,is_maximizer=True)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            best_move=legal_moves[0]
            logging.warning("[get_move] - Terminated due to no remaining time")

            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), is_maximizer=True):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #no_legal_moves = (-1, -1)
        #best_move = no_legal_moves
        best_score = float('-inf') if is_maximizer else float('inf')
        current_player = game.active_player if is_maximizer else game.inactive_player
        legal_moves = game.get_legal_moves(game.active_player)

     #   logging.debug("Current player is Maximizing: %r", is_maximizer)
     #   logging.debug("Current depth: %r", depth)
     #   logging.debug("Best utility: %r", best_score)
     #   logging.debug("Remaining legal moves: %r", legal_moves)

        # Recursion function termination conditions when legal moves exhausted or no plies left
        if not legal_moves:
      #      logging.debug("Recursion terminated due to no remaining legal moves")
            return game.utility(current_player), (-1,-1)
        else:
            best_move=legal_moves[0]
        if depth == 0:
     #       logging.debug("Recursion terminated due to no more plies to search")
            return self.score(game, current_player), best_move

        # Recursively alternate between Maximise and Minimise calculations for decrementing depths
        for move in legal_moves:
            # logging.debug("Recursion with time left is: %r", self.time_left())
       #     logging.debug("Recursion with move: %r", move)
       #     logging.debug("Best utility: %r", best_score)
       #     logging.debug("Best move: %r", best_move)

            # Obtain successor of current state by creating copy of board and applying a move.
            next_state = game.forecast_move(move)
            score, _ = self.alphabeta(next_state, depth - 1, alpha, beta, not is_maximizer)
       #     logging.debug("Forecast utility: %r", score)

            if is_maximizer:
        #        logging.debug("Checking move with Maximising player, score > best_score? : %r", (score > best_score))
                if score > best_score:
                    best_score, best_move = score, move

                    # Prune next successor node if possible
                    if best_score >= beta:
                        break
                    alpha = max(alpha, best_score)
            else:
        #        logging.debug("Checking move with Minimising player, score < best_score? : %r", (score > best_score))
                if score < best_score:
                    best_score, best_move = score, move

                    # Prune next successor node if possible
                    if best_score <= alpha:
                        break
                    beta = min(beta, best_score)

        return best_score, best_move


def run():
    import logging
    from logging.config import dictConfig

    logging_config = dict(
        version = 1,
        formatters = {
            'f': {'format':
                  '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
            },
        handlers = {
         'h': {'class': 'logging.StreamHandler',
               'formatter': 'f',
               'level': logging.DEBUG}
         },
        root = {
            'handlers': ['h'],
            'level': logging.DEBUG,
            },
#        filename='example.log', filemode='w', level=logging.DEBUG)
    )

    dictConfig(logging_config)
#    logger = logging.getLogger()
    logging.getLogger(__name__).addHandler(logging.NullHandler())


    try:
        # Copy of minimax Unit Test for debugging only
        import isolation
        h, w = 3, 4
        test_depth = 2
        starting_location = (1, 1)
        adversary_location = (0, 0)
        iterative_search = False
        search_method = "minimax"
        heuristic = lambda g, p: 0.
        agentUT = MinimaxPlayer(
            search_depth=test_depth, score_fn=open_move_score,timeout = 10)

        agentAB = AlphaBetaPlayer(
            search_depth=test_depth, score_fn=open_move_score,timeout = 10)

#        agentUT.time_left = lambda: 99
        board = isolation.Board(agentUT, agentAB, w, h)
        board.apply_move(starting_location)
        board.apply_move(adversary_location)
        legal_moves = board.get_legal_moves()
        #print(board.to_string())
        #print(legal_moves)
        # for move in legal_moves:
        #    next_state = board.forecast_move(move)
        #    v, _ = agentUT.minimax(next_state, test_depth)
        #    assert type(v) is float, "Minimax function should return a floating point value approximating the score for the branch being searched."

        move = agentUT.get_move(board, lambda: 99)
        
#        print(board.__get_moves__(move))
        assert move in legal_moves, "The get_move() function failed as player 1 on a game in progress. It should return coordinates on the game board for the location of the agent's next move. The move must be one of the legal moves on the current game board."

        return board
    except SystemExit:
        logging.exception('SystemExit occurred')
    except:
        logging.exception('Unknown exception occurred.')

if __name__ == '__main__':
    logger=logging.getLogger()
    b=run()
    winner,move_history,result=b.play()
    
