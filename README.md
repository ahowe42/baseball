# Sabermetrics - Quantitative Baseball Explorations
## A data scientist's musings about *The Game*

This is where I will store and document my exploration into "sabermetrics" quantitative analysis of the game of baseball. The data I'm using is from the [Chadwick Baseball Bureau Baseball DataBank](https://github.com/chadwickbureau/baseballdatabank).

## Files
- [CreateDatabase.ipynb](./src/CreateDatabase.ipynb): In this Jupyter notebook, a sqlite database is built from the core source files listed above. This database is what will be referenced for the modeling / analysis I do.

- [GPFeatureEngineering.ipynb](./src/GPFeatureEngineering.ipynb): This Jupyter notebook holds all the code for building functional trees and running a Genetic Programming algorithm. In addition, the GP's use for feature engineering is demonstrated, by generating a good-fitting feature to some simulated data.

- [NLALRegularSeasonStats.sql](./src/NLALRegularSeasonStats.sql): SQL code for a view in the sqlite database. More to be said later.

- [NLALPlayersTeams.sql.sql](./src/NLALPlayersTeams.sql.sql): SQL code for a view in the sqlite database. More to be said later.

## Resources
[A Guide to Sabermetric Research](https://sabr.org/sabermetrics)
[Column Definitions](https://rdrr.io/cran/Lahman/man/Pitching.html)
