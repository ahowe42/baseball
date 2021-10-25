CREATE VIEW NLALRegularSeasonStats_byPosition AS
    SELECT PLA._yearID,
           PLA._teamID,
           TEM._name AS Team,
           PLA._playerID,
           PPL._nameFirst || ' ' || PPL._nameLast AS Player,
           PLA._stint,
           FLD.Position,
           ifnull(FLD.F_GamePlayed, 0) AS F_GamePlayed,
           ifnull(FLD.F_GameStarted, 0) AS F_GameStarted,
           ifnull(FLD.F_OutPlayed, 0) AS F_OutPlayed,
           ifnull(FLD.F_Putout, 0) AS F_Putout,
           ifnull(FLD.F_Assist, 0) AS F_Assist,
           ifnull(FLD.F_Error, 0) AS F_Error,
           ifnull(FLD.F_DoublePlay, 0) AS F_DoublePlay,
           ifnull(FLD.F_CatcherPassedBall, 0) AS F_CatcherPassedBall,
           ifnull(FLD.F_CatcherWildPitch, 0) AS F_CatcherWildPitch,
           ifnull(FLD.F_CatcherOppStolenBase, 0) AS F_CatcherOppStolenBase,
           ifnull(FLD.F_CatcherOppCaughtStealing, 0) AS F_CatcherOppCaughtStealing,
           ifnull(FLD.F_ZoneRating, 0) AS F_ZoneRating,
           ifnull(BAT.B_Game, 0) AS B_Game,
           ifnull(BAT.B_AtBats, 0) AS B_AtBats,
           ifnull(BAT.B_Run, 0) AS B_Run,
           ifnull(BAT.B_Hits, 0) AS B_Hits,
           ifnull(BAT.B_Double, 0) AS B_Double,
           ifnull(BAT.B_Triple, 0) AS B_Triple,
           ifnull(BAT.B_Homer, 0) AS B_Homer,
           ifnull(BAT.B_RBI, 0) AS B_RBI,
           ifnull(BAT.B_StolenBase, 0) AS B_StolenBase,
           ifnull(BAT.B_CaughtStealing, 0) AS B_CaughtStealing,
           ifnull(BAT.B_Walk, 0) AS B_Walk,
           ifnull(BAT.B_StrikeOuts, 0) AS B_StrikeOuts,
           ifnull(BAT.B_IntentWalk, 0) AS B_IntentWalk,
           ifnull(BAT.B_HitbyPitch, 0) AS B_HitbyPitch,
           ifnull(BAT.B_SacrificeHit, 0) AS B_SacrificeHit,
           ifnull(BAT.B_SacrificeFly, 0) AS B_SacrificeFly,
           ifnull(BAT.B_GroundDoublePlay, 0) AS B_GroundDoublePlay,
           ifnull(PIT.P_Win, 0) AS P_Win,
           ifnull(PIT.P_Loss, 0) AS P_Loss,
           ifnull(PIT.P_GamesPlayed, 0) AS P_GamesPlayed,
           ifnull(PIT.P_GameStarted, 0) AS P_GameStarted,
           ifnull(PIT.P_CompleteGame, 0) AS P_CompleteGame,
           ifnull(PIT.P_Shutout, 0) AS P_Shutout,
           ifnull(PIT.P_Save, 0) AS P_Save,
           ifnull(PIT.P_OutsPitched, 0) AS P_OutsPitched,
           ifnull(PIT.P_Hits, 0) AS P_Hits,
           ifnull(PIT.P_EarnedRun, 0) AS P_EarnedRun,
           ifnull(PIT.P_HomeRun, 0) AS P_HomeRun,
           ifnull(PIT.P_Walk, 0) AS P_Walk,
           ifnull(PIT.P_StrikeOut, 0) AS P_StrikeOut,
           ifnull(PIT.P_OppBattAvg, 0) AS P_OppBattAvg,
           ifnull(PIT.P_ERA, 0) AS P_ERA,
           ifnull(PIT.P_IntentWalk, 0) AS P_IntentWalk,
           ifnull(PIT.P_WildPitch, 0) AS P_WildPitch,
           ifnull(PIT.P_HitBatter, 0) AS P_HitBatter,
           ifnull(PIT.P_Balk, 0) AS P_Balk,
           ifnull(PIT.P_BatterFaced, 0) AS P_BatterFaced,
           ifnull(PIT.P_GameFinished, 0) AS P_GameFinished,
           ifnull(PIT.P_RunAllowed, 0) AS P_RunAllowed,
           ifnull(PIT.P_OppSacrificeHit, 0) AS P_OppSacrificeHit,
           ifnull(PIT.P_OppSacrificeFly, 0) AS P_OppSacrificeFly
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
                      _G AS F_GamePlayed,
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
                      _H AS B_Hits,
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
                      _G AS P_GamesPlayed,
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
                      _ERA AS P_ERA,
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
