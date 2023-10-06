# Parlay Leg Optimizer using Neural Networks (PLONN)
## Introduction
PLONN is a machine learning model designed to optimize sports betting parlays for maximum Return on Investment (ROI). By analyzing historical and in-season data, PLONN ranks potential parlay combinations based on their likelihood of winning and their potential payout.

# Goals:
## Goal A: Parlay Optimization
Implement a neural network that takes a series of sports bets and produces a ranked list of different parlays for maximum ROI.

### Example:

Given the bets:

Bruins (-135) vs. Avalanche (+ 200)

Ducks (-120) vs. Leafs (+100)

Lighting (+136) vs. Oilers (-162)

### PLONN's predicted optimized parlays:

[Bruins, Ducks, Lighting] (+ 461),

[Bruins, Ducks, Oilers] (+ 240),

[Avalanche, Ducks, Oilers] (+156),

Where the first parlay has the best balance of win probability and ROI, the second has a higher win probability but lower ROI, and the third has an equal win probability to the second but an even lower ROI.

## Goal B: Learning
Understand the intricacies of neural networks, their applications, advantages, and limitations. This project serves as a hands-on learning experience in the realm of ML/AI.

## Goal C: Deployment
After rigorous testing and achieving satisfactory accuracy, the plan is to launch a Discord bot offering this service on a subscription basis.

## Plan:
Data Collection: Target bets based on in-season and past-season data. Some teams may exhibit trends based on historical matchups, coaching strategies, or player rivalries.

## Accuracy:
| Date  | TeamA                   | TeamB                   | Total Goals | O/U | Dataset A Pred.(in-season) | Predicted # of Goals A | Dataset B Pred. (historical) | Predicted # of Goals B | Notes           |
|-------|-------------------------|-------------------------|-------------|-----|---------------------------|------------------------|------------------------------|------------------------|-----------------|
| 4-Oct | Detroit Red Wings       | Pittsburgh Penguins     | 3           | 6.5 | 0                         |                        |                              |                        |                 |
| 4-Oct | New York Rangers        | New Jersey Devils       | 7           | 6.5 | 1                         |                        |                              |                        |                 |
| 4-Oct | Calgary Flames          | Edmonton Oilers         | 9           | 6.5 | 1                         |                        |                              |                        |                 |
| 4-Oct | Seattle Kraken          | Vancouver Canucks       | 3           | 6.5 | 1                         |                        |                              |                        |                 |
|       |                         |                         |             |     | Acc %                      | 75%                    | 0                            |                        |                 |
| 5-Oct | Washington Capitals     | Columbus Blue Jackets   | 6           | 6.5 | 0                         | 5.7                    | 1                            | 6.6239                 |                 |
| 5-Oct | Columbus Blue Jackets   | Toronto Maple Leafs     | 7           | 6.5 | 0                         | 5.99                   | 0                            | 5.9103                 | OT for 7th Goal |
| 5-Oct | Florida Panthers        | Tampa Bay Lightning     | 9           | 6.5 | 0                         | 6.9                    | 0                            | 6.26                   |                 |
| 5-Oct | Boston Bruins           | New York Rangers        | 4           | 6.5 | 0                         | 6.47                   | 1                            | 6.6                    |                 |
| 5-Oct | New York Islanders      | Philadelphia Flyers     | 7           | 6.5 | 0                         | 5.63                   | 0                            |                        |                 |
| 5-Oct | Carolina Hurricanes     | Nashville Predators     | 6           | 6.5 | 0                         | 5.9                    | 0                            | 5.18                   |                 |
| 5-Oct | Dallas Stars            | St. Louis Blues         | 4           | 6.5 | 1                         | 6.6                    | 0                            | 6.35                   |                 |
| 5-Oct | Winnipeg Jets           | Ottawa Senators         | 3           | 6.5 | 0                         | 6.16                   | 0                            | 5.4                    |                 |
| 5-Oct | Minnesota Wild          | Chicago Blackhawks      | 3           | 6.5 | 0                         | 5.37                   | 0                            | 4.83                   |                 |
| 5-Oct | San Jose Sharks         | Los Angeles Kings       | 7           | 6.5 | 0                         | 6.18                   | 0                            | 4.7686                 | OT for 7th Goal |
| 5-Oct | Arizona Coyotes         | Anaheim Ducks           | 0           | 6.5 | 0                         | 5.25                   | 0                            | 5.7548                 |                 |
| 5-Oct | Colorado Avalanche      | Vegas Golden Knights    | 1           | 6.5 | 1                         | 6.6                    | 0                            | 6.0907                 |                 |
|       |                         | Acc % excl. OT                        | 84%            |     | Acc % incl. OT                      | 66.6%      

Predicted 11/16 games accurately, 2/16 games were predicted accurately however when into over time for their 7th goal. 

```/Data/accuracy.xlsx``` will be updated along the course of the 2023-2024 season for more up-to-date accuracy
