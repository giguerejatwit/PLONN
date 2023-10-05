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
| Date  | TeamA                | TeamB                   | Total Goals | O/U | Predicted O/U |
|-------|----------------------|-------------------------|-------------|-----|---------------|
| 4-Oct | Detroit Red Wings    | Pittsburgh Penguins     | 3           | 6.5 | 0             |
| 4-Oct | New York Rangers     | New Jersey Devils       | 7           | 6.5 | 1             |
| 4-Oct | Calgary Flames       | Edmonton Oilers         | 9           | 6.5 | 1             |
| 4-Oct | Seattle Kraken       | Vancouver Canucks       | 3           | 6.5 | 1             |

Predicted 3/4 games 75% accurate.

/Data/accuracy.xlsx will be update along the course of the 2023-2024 season for more up-to-date accuracy
