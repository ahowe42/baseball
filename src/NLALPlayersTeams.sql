CREATE VIEW NLALPlayersTeams AS
    SELECT _yearID,
           _teamID,
           _playerID,
           _stint
      FROM Batting
     WHERE _lgID IN ('NL', 'AL') 
    UNION
    SELECT _yearID,
           _teamID,
           _playerID,
           _stint
      FROM Fielding
     WHERE _lgID IN ('NL', 'AL');
