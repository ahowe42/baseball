CREATE VIEW NLALTeamsRanks AS
    SELECT _yearID,
           _lgID,
           _divID,
           _teamID,
           _Rank,
           _DivWin,
           _LGWin,
           _G,
           _W,
           _L,
           1.0 * _W / _G AS WinPerc,
           1.0 * _W / _L AS WinLosPerc
      FROM Teams
     WHERE _lgID IN ('NL', 'AL') AND 
           _divID IN ('W', 'E') 
     ORDER BY _yearID,
              _lgID,
              _divID,
              _Rank;