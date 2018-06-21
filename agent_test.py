"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
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

def heuristic_1_center(game, player) -> float:
    """
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

def heuristic_2_reflection(game, player) -> float:
    """
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

def heuristic_3_partition(game, player) -> float:
    """
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

def heuristic_combined_1_2(game, player) -> float:
    """
    Combines Heuristics 1 and 2

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
    reflection_available_factor = get_reflection_available_factor(game, player)

    return float(center_available_factor + reflection_available_factor)

def heuristic_combined_1_3(game, player) -> float:
    """
    Combines Heuristics 1 and 3

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
    partition_possible_factor = get_partition_possible_factor(game, player)

    return float(center_available_factor + partition_possible_factor)

def heuristic_combined_2_3(game, player) -> float:
    """
    Combines Heuristics 2 and 3

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
    partition_possible_factor = get_partition_possible_factor(game, player)

    return float(reflection_available_factor + partition_possible_factor)

def heuristic_combined_1_2_3(game, player) -> float:
    """
    Combines Heuristics 1, 2 and 3

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
    reflection_available_factor = get_reflection_available_factor(game, player)
    partition_possible_factor = get_partition_possible_factor(game, player)

    return float(center_available_factor +
                 reflection_available_factor +
                 partition_possible_factor)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    heuristics_options = {
        "null_score": null_score,
        "open_move_score": open_move_score,
        "improved_score": improved_score,
        "heuristic_1_center": heuristic_1_center,
        "heuristic_2_reflection": heuristic_2_reflection,
        "heuristic_3_partition": heuristic_3_partition,
        "heuristic_combined_1_2": heuristic_combined_1_2,
        "heuristic_combined_1_3": heuristic_combined_1_3,
        "heuristic_combined_2_3": heuristic_combined_2_3,
        "heuristic_combined_1_2_3": heuristic_combined_1_2_3
    }

    return heuristics_options["heuristic_2_reflection"](game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate successors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        remaining_legal_moves = legal_moves
        no_legal_moves = (-1, -1)
        best_move = no_legal_moves
        if not remaining_legal_moves:
            logging.debug("Get Moves - Terminated due to no remaining legal moves")
            return no_legal_moves

        # Flag indicating Iterative Deepening Search - Initialise Depth at 0 (to later be incremented)
        #   - Reference: https://github.com/aimacode/aima-pseudocode/blob/master/md/Iterative-Deepening-Search.md
        # Flag otherwise indicates Fixed-Depth Search (FDS) - Set to Search Depth parameter (only for FDS)
        depth = 0 if self.iterative else self.search_depth

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # Flag indicates perform Iterative Deepening Search
            if self.iterative:
                logging.debug("Get Moves - Performing Iterative Deepening Search to depth %r: ", depth)
                while True:
                    # logging.debug("Time left is: %r", self.time_left())
                    depth += 1
                    if self.method == 'minimax':
                        _, best_move = self.minimax(game, depth)
                    elif self.method == 'alphabeta':
                        _, best_move = self.alphabeta(game, depth)
                    else:
                        raise ValueError("Invalid method")

                    # Check remaining time between depth iterations and
                    # return the best move when less than 1ms to avoid
                    # running out of time and forfeiting the game
                    if self.time_left() <= 0.001:
                        return best_move

            # Flag indicates perform Fixed-Depth Search
            else:
                logging.debug("Get Moves - Performing Fixed-Depth Search to depth %r: ", depth)
                # logging.debug("Time left is: %r", self.time_left())
                if self.method == 'minimax':
                    _, best_move = self.minimax(game, depth)
                elif self.method == 'alphabeta':
                    _, best_move = self.alphabeta(game, depth)
                else:
                    raise ValueError("Invalid method")
                return best_move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            # logging.warning("Get Moves - Timeout reached")

            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Reference: https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        # Initialise variable for no legal moves
        no_legal_moves = (-1, -1)
        best_move = no_legal_moves
        best_utility = float('-inf') if maximizing_player else float('inf')
        current_player = game.active_player if maximizing_player else game.inactive_player
        remaining_legal_moves = game.get_legal_moves(game.active_player)

        logging.debug("Current player is Maximizing: %r", maximizing_player)
        logging.debug("Current depth: %r", depth)
        logging.debug("Best utility: %r", best_utility)
        logging.debug("Remaining legal moves: %r", remaining_legal_moves)

        # Recursion function termination conditions when legal moves exhausted or no plies left
        if not remaining_legal_moves:
            logging.debug("Recursion terminated due to no remaining legal moves")
            return game.utility(current_player), no_legal_moves
        elif depth == 0:
            logging.debug("Recursion terminated due to no more plies to search")
            return self.score(game, current_player), remaining_legal_moves[0]

        # Recursively alternate between Maximise and Minimise calculations for decrementing depths
        for move in remaining_legal_moves:
            # logging.debug("Recursion with time left is: %r", self.time_left())
            logging.debug("Recursion with move: %r", move)
            logging.debug("Best utility: %r", best_utility)
            logging.debug("Best move: %r", best_move)

            # Obtain successor of current state by creating copy of board and applying a move.
            next_state = game.forecast_move(move)
            forecast_utility, _ = self.minimax(next_state, depth - 1, not maximizing_player)
            logging.debug("Forecast utility: %r", forecast_utility)

            if maximizing_player:
                logging.debug("Checking move with Maximising player, forecast_utility > best_utility? : %r", (forecast_utility > best_utility))
                if forecast_utility > best_utility:
                    best_utility, best_move = forecast_utility, move
            else:
                logging.debug("Checking move with Minimising player, forecast_utility < best_utility? : %r", (forecast_utility > best_utility))
                if forecast_utility < best_utility:
                    best_utility, best_move = forecast_utility, move

        return best_utility, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO - Refactor duplicate from minimax and alphabeta into helper function
        # Reference: https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        # Initialise variable for no legal moves
        no_legal_moves = (-1, -1)
        best_move = no_legal_moves
        best_utility = float('-inf') if maximizing_player else float('inf')
        current_player = game.active_player if maximizing_player else game.inactive_player
        remaining_legal_moves = game.get_legal_moves(game.active_player)

        logging.debug("Current player is Maximizing: %r", maximizing_player)
        logging.debug("Current depth: %r", depth)
        logging.debug("Best utility: %r", best_utility)
        logging.debug("Remaining legal moves: %r", remaining_legal_moves)

        # Recursion function termination conditions when legal moves exhausted or no plies left
        if not remaining_legal_moves:
            logging.debug("Recursion terminated due to no remaining legal moves")
            return game.utility(current_player), no_legal_moves
        elif depth == 0:
            logging.debug("Recursion terminated due to no more plies to search")
            return self.score(game, current_player), remaining_legal_moves[0]

        # Recursively alternate between Maximise and Minimise calculations for decrementing depths
        for move in remaining_legal_moves:
            # logging.debug("Recursion with time left is: %r", self.time_left())
            logging.debug("Recursion with move: %r", move)
            logging.debug("Best utility: %r", best_utility)
            logging.debug("Best move: %r", best_move)

            # Obtain successor of current state by creating copy of board and applying a move.
            next_state = game.forecast_move(move)
            forecast_utility, _ = self.alphabeta(next_state, depth - 1, alpha, beta, not maximizing_player)
            logging.debug("Forecast utility: %r", forecast_utility)

            if maximizing_player:
                logging.debug("Checking move with Maximising player, forecast_utility > best_utility? : %r", (forecast_utility > best_utility))
                if forecast_utility > best_utility:
                    best_utility, best_move = forecast_utility, move

                    # Prune next successor node if possible
                    if best_utility >= beta:
                        break
                    alpha = max(alpha, best_utility)
            else:
                logging.debug("Checking move with Minimising player, forecast_utility < best_utility? : %r", (forecast_utility > best_utility))
                if forecast_utility < best_utility:
                    best_utility, best_move = forecast_utility, move

                    # Prune next successor node if possible
                    if best_utility <= alpha:
                        break
                    beta = min(beta, best_utility)

        return best_utility, best_move

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
        h, w = 7, 7
        test_depth = 1
        starting_location = (5, 3)
        adversary_location = (0, 0)
        iterative_search = False
        search_method = "minimax"
        heuristic = lambda g, p: 0.
        agentUT = CustomPlayer(
            test_depth, heuristic, iterative_search, search_method)
        agentUT.time_left = lambda: 99
        board = isolation.Board(agentUT, 'null_agent', w, h)
        board.apply_move(starting_location)
        board.apply_move(adversary_location)
        legal_moves = board.get_legal_moves()

        # for move in legal_moves:
        #    next_state = board.forecast_move(move)
        #    v, _ = agentUT.minimax(next_state, test_depth)
        #    assert type(v) is float, "Minimax function should return a floating point value approximating the score for the branch being searched."

        move = agentUT.get_move(board, legal_moves, lambda: 99)
        assert move in legal_moves, "The get_move() function failed as player 1 on a game in progress. It should return coordinates on the game board for the location of the agent's next move. The move must be one of the legal moves on the current game board."

        return
    except SystemExit:
        logging.exception('SystemExit occurred')
    except:
        logging.exception('Unknown exception occurred.')

if __name__ == '__main__':
    run()
