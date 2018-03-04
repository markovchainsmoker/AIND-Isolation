import random
import timeit

class SearchTimeout(Exception):
    """Added current game state as argument. """
    def __init__(self,gameState):
      self.gameState=gameState

def custom_score(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    own = len(game.get_legal_moves(player))
    opp = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own-opp)
    #return float(len(game.get_legal_moves(player))-len(game.get_legal_moves(player)))

def custom_score_2(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
        
    own_moves = len(game.get_legal_moves(player))
    return own_moves
    #opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    #return float(len(game.get_legal_moves(player))-len(game.get_legal_moves(game.get_opponent(player))))
    #return float(len(game.get_legal_moves(player))-0.5*len(game.get_legal_moves(player)))
    #mine=len(game.get_legal_moves(player))
    #their=len(game.get_legal_moves(game.get_opponent(player)))
    #return float(mine-their)
    #return float(len(game.get_legal_moves(player))-len(game.get_legal_moves(get.opponent(player)))

def custom_score_3(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #w, h = game.width / 2., game.height / 2.
    #y, x = game.get_player_location(player)
    #return float((h - y)**2 + (w - x)**2)

    own = len(game.get_legal_moves(player))
    opp = len(game.get_legal_moves(game.get_opponent(player)))
    diff= float(own-opp)
    #free_moves=float(len(game.get_legal_moves(player)))
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    #offset= float((h - y)**2 + (w - x)**2)
    offset=float(abs(h - y) + abs(w - x))
    return diff-offset
    #player_location=game.get_player_location(player)
    #x=abs(game.width-player_location[0]-2)
    #y=abs(game.height-player_location[1]-2)
    
    #print('{},{},{},{}'.format(player_location,x,y,x+y))
    #return free_moves-x-y
    
class IsolationPlayer:

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10):
      self.search_depth = search_depth
      self.score = score_fn
      self.time_left = None
      self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=140):
      """new init to specify timeout and extras"""
      super().__init__(search_depth=search_depth,score_fn=score_fn,timeout=timeout)
      self.mins=0
      self.maxs=0
      self.a=[]
      self.m=[]
      self.v=[]
    
    def exception_handler(self,e):
      """calculate the score of the game state that threw the timeout"""
      s=e.gameState
      #print('{}->{}:{}'.format(s.get_player_location(s.active_player),s.get_player_location(s.inactive_player),self.score(s,self)))
      #print(self.time_left())
      return self.score(s,self)
          	
    def get_move(self, game, time_left):
        self.time_left = time_left
        best_move = (-1, -1)
        try:
            best_move=self.minimax(game)
        except SearchTimeout as e:
          """if timed out at root we default to the greedy 1ply scoring"""
          legal_moves=game.get_legal_moves()
          if not legal_moves:
            print('timed out, no legal move')
          actions=[(self.score(game.forecast_move(m),self),m) for m in legal_moves]
          best_score,best_move=max(actions)
          #print('defaulting to 1plie score -> {} [{}]'.format(best_move,best_score))
        return best_move
        
        
    def minimax(self,gameState):
        if self.time_left() < self.TIMER_THRESHOLD:raise SearchTimeout(gameState)
        best_move=(-1,-1)
        try:
            legal_moves = gameState.get_legal_moves()
            if not legal_moves:
              return best_move
            actions=[(self.min_value(gameState.forecast_move(m),self.search_depth-1,self.time_left),m) for m in legal_moves]
            best_score,best_move=max(actions)
            self.a.append(actions)
            self.v.append(best_score)
            self.m.append(best_move)
        except SearchTimeout as e:
           #print('time out at depth 1 with {:2.1f}' .format(self.time_left()))
           raise e
        return best_move
        
    def min_value(self,s,d,time_left):
        self.mins+=1
        if time_left() < self.TIMER_THRESHOLD:raise SearchTimeout(s)
        try:    
            if d<1:return self.score(s,self)
            legal_moves = s.get_legal_moves()
            if not legal_moves:return s.utility(self)
            actions=[(self.max_value(s.forecast_move(m),d-1,time_left),m) for m in legal_moves]
            v,a=min(actions)
        except SearchTimeout as e:
            v=self.exception_handler(e)
            #print('timeout in min {}'.format(v))
        return v

    def max_value(self,s,d,time_left):
        self.maxs+=1
        if time_left() < self.TIMER_THRESHOLD:raise SearchTimeout(s)
        try:    
            if d<1:return self.score(s,self)
            legal_moves = s.get_legal_moves()
            if not legal_moves:return s.utility(self)
            actions=[(self.min_value(s.forecast_move(m),d-1,time_left),m) for m in legal_moves]
            v,a=max(actions)
        except SearchTimeout as e:
            v=self.exception_handler(e)
            #print('timeout in max {}'.format(v))
        return v

class AlphaBetaPlayer(IsolationPlayer):

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=50):
      """new init to specify timeout and extras"""
      super().__init__(search_depth=search_depth,score_fn=score_fn,timeout=timeout)
      self.mins=0
      self.maxs=0
      self.a=[]
      self.m=[]
      self.v=[]
    
    def exception_handler(self,e):
      """calculate the score of the game state that threw the timeout"""
      s=e.gameState
      print('{}->{}:{}'.format(s.get_player_location(s.active_player),s.get_player_location(s.inactive_player),self.score(s,self)))
      print(self.time_left())
      return self.score(s,self)
          	
    def get_move(self, game, time_left):
        self.time_left = time_left
        best_move = (-1, -1)
        try:
            best_move=self.alphabeta(game)
        except SearchTimeout as e:
          """if timed out at root we default to the greedy 1ply scoring"""
          legal_moves=game.get_legal_moves()
          if not legal_moves:
            print('timed out, no legal move')
          actions=[(self.score(game.forecast_move(m),self),m) for m in legal_moves]
          best_score,best_move=max(actions)
          #print('defaulting to 1plie score -> {} [{}]'.format(best_move,best_score))
        return best_move
        
        
    def alphabeta(self,gameState,alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:raise SearchTimeout(gameState)
        best_move=(-1,-1)
        try:
            legal_moves = gameState.get_legal_moves()
            if not legal_moves:
              return best_move
            #best_score=-float('inf')
            actions=[(self.min_value(gameState.forecast_move(m),self.search_depth-1,alpha,beta,self.time_left),m) for m in legal_moves]
            """for m in legal_moves:
              v=self.min_value(gameState.forecast_move(m),self.search_depth-1,alpha,beta,self.time_left)
              if v>=best_score:
              	print('{} is new best {} ({},{})'.format(m,v,'.','.'))
              	best_score=v
              	best_move=m
              alpha=max(v,alpha)
              if beta<=alpha:pass"""
            
            
            best_score,best_move=max(actions)
            self.a.append(actions)
            self.v.append(best_score)
            self.m.append(best_move)
        except SearchTimeout as e:
           #print('time out at depth 1 with {:2.1f}' .format(self.time_left()))
           raise e
        return best_move
        
    def min_value(self,s,d,a,b,time_left):
        self.mins+=1
        if time_left() < self.TIMER_THRESHOLD:raise SearchTimeout(s)
        try:    
            if d<1:return self.score(s,self)
            legal_moves = s.get_legal_moves()
            if not legal_moves:return s.utility(self)
            v=float('inf')
            #print('min',legal_moves)
            for m in legal_moves:
              v=min(v,self.max_value(s.forecast_move(m),d-1,a,b,time_left))
              b=min(v,b)
              #print('{}->{} {} ({},{})'.format(s.get_player_location(self),m,v,a,b))
              if b<=a:
              	#print('alpha-pruning {} {}'.format(m,v))
              	break
        except SearchTimeout as e:
          pass
            #v=self.exception_handler(e)
            #print('timeout in min {}'.format(v))
        return v

    def max_value(self,s,d,a,b,time_left):
        self.maxs+=1
        if time_left() < self.TIMER_THRESHOLD:raise SearchTimeout(s)
        try:    
            if d<1:return self.score(s,self)
            legal_moves = s.get_legal_moves()
            if not legal_moves:return s.utility(self)
            v=-float('inf')
            #print('max',legal_moves)
            for m in legal_moves:
              v=max(v,self.min_value(s.forecast_move(m),d-1,a,b,time_left))
              a=max(v,a)
              if b<=a:
              	#print(s.to_string())
              	#print('max {}: {} {} beta-pruning {}'.format(self.maxs,m,v,set(legal_moves)-set([m])))
              	break
        except SearchTimeout as e:
          pass
            #v=self.exception_handler(e)
            #print('timeout in max {}'.format(v))
        return v
        


def durations(f):
	start=timeit.default_timer()
	#v=float('inf')
	f()
	return (timeit.default_timer()-start)*1000
	
if __name__ == "__main__":
	from isolation import Board
	import timeit
	from sample_players import GreedyPlayer
	w,h=7,7
	alpha=AlphaBetaPlayer(score_fn=custom_score_3)
	mini=MinimaxPlayer(score_fn=custom_score_3)
	mini2=MinimaxPlayer(score_fn=custom_score_2)
	game=Board(alpha,mini,width=w,height=h)
	time_millis = lambda: 1000 * timeit.default_timer()
	time_limit=150
	move_start = time_millis()
	time_left = lambda : time_limit - (time_millis() - move_start)
	
	#start=time_millis()
	
	#res=game.active_player.min_value(game.forecast_move((1,1)),2,-float('inf'),float('inf'),time_left)
	
	res=game.active_player.get_move(game,time_left)
	duration=time_millis()-move_start
	#res1=game.active_player.get_move(game,time_left)
	#print(game.active_player.mins,game.active_player.maxs,duration,res,duration/game.active_player.mins)
	for i in range(0,0):
	  #game = Board(MinimaxPlayer(), AlphaBetaPlayer(),width=w,height=h)
	  #winner, history, outcome = game.play(time_limit=150)
	  #print('{} {}'.format(winner,outcome))
	  game = Board(AlphaBetaPlayer(),MinimaxPlayer(),width=w,height=h)
	  winner, history, outcome = game.play(time_limit=150)
	  print('{} {}'.format(winner,outcome))
	#for b in Board.getinstances():print(b.to_string())
	#for b in Board._boards[-4:]:print(b.to_string())
	  #[print(b.to_string()) for b in history]
	 # mini.minimax(gam)