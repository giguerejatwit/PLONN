Parlay Leg Opimizer using Neural Networks (PLONN)

Goals:

GoalA) The goal of this project is to implement a neural network which will be able to take a series of Sport bets and produce
a ranked list of different paralays for maxium ROI. (i.e.)

Bruins (-135) vs Avalanche (+ 200)
Ducks (-120) vs Leafs (+100)
Lighting (+136) vs Oilers (-162)

we should expect an optimizated parlays of:

1. [Bruins, Ducks, Lighting] (+ 461)
2. [Bruins, Ducks, Oilers] (+ 240)
3. [Avalanch, Ducks, Oilers] (+156)

Where 1 has the best chance of winning and ROI, although 2 has the best chance of winning however lacks the ROI, and 3 has an equal chance of winning as 2, however an even less ROI.

Goal B) Learn about neural networks, how they work, when they are the best solution, and why.
This should be considered a learning experience to get a better understanding of ML/AI.

Goal C) After several weeks of testing and approved accuracy, will lauch a dicord bot for a monthly subscription

Plan:

1. Target our bets based off of in-season and past-season data. Some teams may have trends of winning/lossing based off the coaching and player-to-player that have had a historical battle between each other. For example, 

Boston Bruins have won the past 10 games over the Montreal Canadians
Boston Bruins are 7-2 vs. Las Vegas Golden Nights


Data:

1. Historical Data over past 3 seasons/ team
    a. Find trends of winning/lossing vs a specific team

2. In-season data (Roster Strength, Goalie Performance, WL streaks, GF, GA, i.e., )

Development:

Using Python, PyTorch, to develop an implementation of a Neural Network, and fit the data to the NN. 

DB(?),