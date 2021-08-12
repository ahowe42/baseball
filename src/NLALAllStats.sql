CREATE VIEW NLALAllStats AS
    SELECT PLA._yearID,
           TEM._name AS TEAM,
           PPL._nameFirst || ' ' || PPL._nameLast AS Player,
           PLA._stint,
           FLD.F_Game,
           FLD.F_GameStarted,
           FLD.F_OutPlayed,
           FLD.F_Putout,
           FLD.F_Assist,
           FLD.F_Error,
           FLD.F_DoublePlay,
           FLD.F_CatcherPassedBall,
           FLD.F_CatcherWildPitch,
           FLD.F_CatcherOppStolenBase,
           FLD.F_CatcherOppCaughtStealing,
           FLD.F_ZoneRating,
           BAT.B_Game,
           BAT.B_AtBats,
           BAT.B_Run,
           BAT.B_Hit,
           BAT.B_Double,
           BAT.B_Triple,
           BAT.B_Homer,
           BAT.B_RBI,
           BAT.B_StolenBase,
           BAT.B_CaughtStealing,
           BAT.B_Walk,
           BAT.B_StrikeOuts,
           BAT.B_IntentWalk,
           BAT.B_HitbyPitch,
           BAT.B_SacrificeHit,
           BAT.B_SacrificeFly,
           BAT.B_GroundDoublePlay,
           PIT.P_Win,
           PIT.P_Loss,
           PIT.P_Game,
           PIT.P_GameStarted,
           PIT.P_CompleteGame,
           PIT.P_Shutout,
           PIT.P_Save,
           PIT.P_OutsPitched,
           PIT.P_Hits,
           PIT.P_EarnedRun,
           PIT.P_HomeRun,
           PIT.P_Walk,
           PIT.P_StrikeOut,
           PIT.P_OppBattAvg,
           PIT.P_EarnedRunAvg,
           PIT.P_IntentWalk,
           PIT.P_WildPitch,
           PIT.P_HitBatter,
           PIT.P_Balk,
           PIT.P_BatterFaced,
           PIT.P_GameFinished,
           PIT.P_RunAllowed,
           PIT.P_OppSacrificeHit,
           PIT.P_OppSacrificeFly,
           PLA._playerID,
           PLA._teamID
      FROM NLALPLayersTeams AS PLA
           INNER JOIN
           People AS PPL ON PLA._playerID = PPL._playerID
           INNER JOIN
           Teams AS TEM ON PLA._teamID = TEM._teamID AND 
                           PLA._yearID = TEM._yearID
           LEFT JOIN
           (
               SELECT _playerID,
                      _yearID,
                      _stint,
                      _teamID,
                      _POS AS Position,
                      _G AS F_Game,
                      _GS AS F_GameStarted,
                      _InnOuts AS F_OutPlayed,
                      _PO AS F_Putout,
                      _A AS F_Assist,
                      _E AS F_Error,
                      _DP AS F_DoublePlay,
                      _PB AS F_CatcherPassedBall,
                      _WP AS F_CatcherWildPitch,
                      _SB AS F_CatcherOppStolenBase,
                      _CS AS F_CatcherOppCaughtStealing,
                      _ZR AS F_ZoneRating
                 FROM Fielding
                WHERE _lgID IN ('NL', 'AL') 
           )
           AS FLD ON PLA._playerID = FLD._playerID AND 
                     PLA._teamID = FLD._teamID AND 
                     PLA._yearID = FLD._yearID AND 
                     PLA._stint = FLD._stint
           LEFT JOIN
           (
               SELECT _playerID,
                      _yearID,
                      _stint,
                      _teamID,
                      _G AS B_Game,
                      _AB AS B_AtBats,
                      _R AS B_Run,
                      _H AS B_Hit,
                      _2B AS B_Double,
                      _3B AS B_Triple,
                      _HR AS B_Homer,
                      _RBI AS B_RBI,
                      _SB AS B_StolenBase,
                      _CS AS B_CaughtStealing,
                      _BB AS B_Walk,
                      _SO AS B_StrikeOuts,
                      _IBB AS B_IntentWalk,
                      _HBP AS B_HitbyPitch,
                      _SH AS B_SacrificeHit,
                      _SF AS B_SacrificeFly,
                      _GIDP AS B_GroundDoublePlay
                 FROM Batting
                WHERE _lgID IN ('NL', 'AL') 
           )
           AS BAT ON PLA._playerID = BAT._playerID AND 
                     PLA._teamID = BAT._teamID AND 
                     PLA._yearID = BAT._yearID AND 
                     PLA._stint = BAT._stint
           LEFT JOIN
           (
               SELECT _playerID,
                      _yearID,
                      _stint,
                      _teamID,
                      _W AS P_Win,
                      _L AS P_Loss,
                      _G AS P_Game,
                      _GS AS P_GameStarted,
                      _CG AS P_CompleteGame,
                      _SHO AS P_Shutout,
                      _SV AS P_Save,
                      _IPouts AS P_OutsPitched,
                      _H AS P_Hits,
                      _ER AS P_EarnedRun,
                      _HR AS P_HomeRun,
                      _BB AS P_Walk,
                      _SO AS P_StrikeOut,
                      _BAOpp AS P_OppBattAvg,
                      _ERA AS P_EarnedRunAvg,
                      _IBB AS P_IntentWalk,
                      _WP AS P_WildPitch,
                      _HBP AS P_HitBatter,
                      _BK AS P_Balk,
                      _BFP AS P_BatterFaced,
                      _GF AS P_GameFinished,
                      _R AS P_RunAllowed,
                      _SH AS P_OppSacrificeHit,
                      _SF AS P_OppSacrificeFly
                 FROM Pitching
                WHERE _lgID IN ('NL', 'AL') 
           )
           AS PIT ON PLA._playerID = PIT._playerID AND 
                     PLA._teamID = PIT._teamID AND 
                     PLA._yearID = PIT._yearID AND 
                     PLA._stint = PIT._stint;
