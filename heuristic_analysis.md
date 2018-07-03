####  [AIND 2018 Term 1] Carl Wennstam - Isolation Project

## Heuristics Analysis

<br><br>

### Introduction

### Benchmarks
#### Statistical properties
In order to compare heuristics and algorithms first a baseline should be established. As a null hypothesis we might assume a binomial distribution with a 50/50 win rate, but there will be some degree of variance in a sample. The suggested 10 samples per match-up is too small to draw any reliable conclusions. WIth 100 samples, a 95% confidence interval is {40,60}. Thus, as a first sanity check of the setup, we would expect matches between players with the same algorithm and heuristic to show win rates that lie within this interval.
#### Baseline winrates
Starting with the 'fair' games as defined in the *tournament* file: games start with a random move and continue from there.
##### Minimax
  MM|open|center|improved
  - | -|-|-
  **open**    | 53|17|54|
  **center**  | 84|45|87|
  **improved**| 47|19|56|
First thing to note is that the win rates against same heuristic falls within the 95% confidence interval, as expected.  
##### AlphaBeta
  AB|open|center|improved
  - | -|-|-
  **open**    | 47|12|62|
  **center**  | 86|51|85|
  **improved**| 49|26|51|

The situation is similar for AlphaBeta vs itself.
##### Minimax vs AlphaBeta
  AB/MM|open|center|improved
  - | -|-|-
  **open**    | 53|14|49|
  **center**  | 82|42|86|
  **improved**| 45|11|51|
Minimax vs AlphaBeta again shows similar results, although the MM open vs AB improved win rate is quite a bit lower.
No timeouts occured during these simulations with a *TIMER_THRESHOLD*=20. 

#### Full game
The initial random move setup might not be completely realistic so I removed the initial randomization and ran the matchups again.
##### Minimax (*TIMER_THRESHOLD* = 20)
  MM|open|center|improved
  - | -|-|-
  **open**    | 53|9|49|
  **center**  | 95|53|97|
  **improved**| 56|4|52|
No we are seeing some interestingly abysmal results for **center** vs **open** and **improved**. There was also 7 timeouts, which may explain some of the poor performance. As the worst case search (the first move) is now a larger tree (after removing the opening randomizing feature), the **center** heuristic in particular seems to struggle as a result. Increasing *TIMER_THREHOLD* parameter from 20 to 90 helped get rid of the timeouts. 
##### Minimax (*TIMER_THREHOLD* = 90)
  MM|open|center|improved
  - | -|-|-
  **open**    | 45|19|48|
  **center**  | 80|51|91|
  **improved**| 47|13|53|
The statistics are now back to the 'fair' game, but this example illustrates how taxing the first move can be. Needless to say, by increasing *TIMER_THREHOLD* we are allowing less time to search the tree, which might restrict the play, forcing sup-optimal play.
#### Time complexity
##### Search times for *get_move*
  Method|d=1|d=2|d=3
  - | -|-|-
  **Random**    | 0.05|0.05|0.05|
  **MM_Center**  | 2.0|86|417|
  **AB_Center**| 2.1|4|30|
  **MM_Open**| 2.6|100|543|
  **AB_Open**| 2.7|18|137|
  **MM_Improved**| 3.5|129|694|
  **AB_Improved**| 3.6|95|147|


|
> Written with [StackEdit](https://stackedit.io/).
