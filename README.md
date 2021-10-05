# Sabermetrics - Quantitative Baseball Explorations
## A data scientist's musings about *The Game*

This is where I will store and document my exploration into "sabermetrics" quantitative analysis of the game of baseball. The data I'm using is from the [Chadwick Baseball Bureau Baseball DataBank](https://github.com/chadwickbureau/baseballdatabank).

## Notebooks
- [CreateDatabase.ipynb](./notebooks/CreateDatabase.ipynb): In this Jupyter notebook, a sqlite database is built from the core source files listed above. This database is what will be referenced for the modeling / analysis.

- [GPFeatureEngineering.ipynb](./notebooks/GPFeatureEngineering.ipynb): This Jupyter notebook demonstrates building functional trees and running a Genetic Programming algorithm on some data. In addition, the GP's use for feature engineering is demonstrated, by generating a good-fitting feature to some simulated data.

- [GPBaseball.ipynb](./notebooks/GPBaseball.ipynb): This Jupyber notebook applies genetic programming to a problem with the baseball data.

## View Queries
The `src` folder holds several view creation queries to be applied to the sqlite database. Analysis of the data will be based on these views.
- [NLALPlayersTeams.sql](./src/NLALPlayersTeams.sql): join teams and players

- [NLALRegularSeasonStats_byPosition.sql](./src/NLALRegularSeasonStats_byPosition.sql): regular season stats by player by position

- [NLALRegularSeasonStats_byPlayer.sql](./src/NLALRegularSeasonStats_byPosition.sql): regular season stats by player

- [NLALRegularSeasonStats_byTeam.sql](./src/NLALRegularSeasonStats_byPosition.sql): regular season stats aggregated by team; all stats are summed except ERA, which is averaged

- [NLALTeamsRanks.sql](./src/NLALTeamsRanks.sql): team rankings, wins, losses, games played by team by season

- [NLALRegularSeasonTeamStatsRanks.sql](./src/NLALRegularSeasonTeamStatsRanks.sql): joins NLALTeamsRanks and NLALRegularSeasonStats_byTeam views

## Source Code
The `src` folder contains the code used for running the genetic programming algorithm and evaluating results.
- [GP](./src/GP/): class and function definitions for functional trees, tree objective functions, GP operators and running function
- [util](./src/util): generally useful functions referenced by the above-mentioned functions


## Resources
[A Guide to Sabermetric Research](https://sabr.org/sabermetrics)
[Column Definitions](https://rdrr.io/cran/Lahman/man/Pitching.html)
