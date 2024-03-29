CREATE VIEW NLALRegularSeasonStats_byTeamPrevYear AS
    SELECT _yearID,
           _lgID,
           _divID,
           _teamID,
           _Rank,
           _DivWin,
           _LgWin,
           _G,
           _W,
           _L,
           WinPerc,
           WinLosPerc,
           SUM(F_GamePlayed) AS F_GamePlayed,
           SUM(F_GameStarted) AS F_GameStarted,
           SUM(F_OutPlayed) AS F_OutPlayed,
           SUM(F_Putout) AS F_Putout,
           SUM(F_Assist) AS F_Assist,
           SUM(F_Error) AS F_Error,
           SUM(F_DoublePlay) AS F_DoublePlay,
           SUM(F_CatcherPASsedBall) AS F_CatcherPASsedBall,
           SUM(F_CatcherWildPitch) AS F_CatcherWildPitch,
           SUM(F_CatcherOppStolenBASe) AS F_CatcherOppStolenBASe,
           SUM(F_CatcherOppCaughtStealing) AS F_CatcherOppCaughtStealing,
           SUM(F_ZoneRating) AS F_ZoneRating,
           SUM(B_Game) AS B_Game,
           SUM(B_AtBats) AS B_AtBats,
           SUM(B_Run) AS B_Run,
           SUM(B_Hits) AS B_Hits,
           SUM(B_Double) AS B_Double,
           SUM(B_Triple) AS B_Triple,
           SUM(B_Homer) AS B_Homer,
           SUM(B_RBI) AS B_RBI,
           SUM(B_StolenBASe) AS B_StolenBASe,
           SUM(B_CaughtStealing) AS B_CaughtStealing,
           SUM(B_Walk) AS B_Walk,
           SUM(B_StrikeOuts) AS B_StrikeOuts,
           SUM(B_IntentWalk) AS B_IntentWalk,
           SUM(B_HitbyPitch) AS B_HitbyPitch,
           SUM(B_SacrificeHit) AS B_SacrificeHit,
           SUM(B_SacrificeFly) AS B_SacrificeFly,
           SUM(B_GroundDoublePlay) AS B_GroundDoublePlay,
           SUM(P_Win) AS P_Win,
           SUM(P_Loss) AS P_Loss,
           SUM(P_GamesPlayed) AS P_GamesPlayed,
           SUM(P_GameStarted) AS P_GameStarted,
           SUM(P_CompleteGame) AS P_CompleteGame,
           SUM(P_Shutout) AS P_Shutout,
           SUM(P_Save) AS P_Save,
           SUM(P_OutsPitched) AS P_OutsPitched,
           SUM(P_Hits) AS P_Hits,
           SUM(P_EarnedRun) AS P_EarnedRun,
           SUM(P_HomeRun) AS P_HomeRun,
           SUM(P_Walk) AS P_Walk,
           SUM(P_StrikeOut) AS P_StrikeOut,
           SUM(P_OppBattAvg) AS P_OppBattAvg,
           avg(P_ERA) AS P_ERA,
           SUM(P_IntentWalk) AS P_IntentWalk,
           SUM(P_WildPitch) AS P_WildPitch,
           SUM(P_HitBatter) AS P_HitBatter,
           SUM(P_Balk) AS P_Balk,
           SUM(P_BatterFaced) AS P_BatterFaced,
           SUM(P_GameFinished) AS P_GameFinished,
           SUM(P_RunAllowed) AS P_RunAllowed,
           SUM(P_OppSacrificeHit) AS P_OppSacrificeHit,
           SUM(P_OppSacrificeFly) AS P_OppSacrificeFly
      FROM (
               SELECT TEM._yearID,
                      TEM._lgID,
                      TEM._divID,
                      TEM._teamID,
                      TEM._Rank,
                      TEM._DivWin,
                      TEM._LgWin,
                      TEM._G,
                      TEM._W,
                      TEM._L,
                      TEM.WinPerc,
                      TEM.WinLosPerc,
                      S.F_GamePlayed,
                      S.F_GameStarted,
                      S.F_OutPlayed,
                      S.F_Putout,
                      S.F_Assist,
                      S.F_Error,
                      S.F_DoublePlay,
                      S.F_CatcherPassedBall,
                      S.F_CatcherWildPitch,
                      S.F_CatcherOppStolenBase,
                      S.F_CatcherOppCaughtStealing,
                      S.F_ZoneRating,
                      S.B_Game,
                      S.B_AtBats,
                      S.B_Run,
                      S.B_Hits,
                      S.B_Double,
                      S.B_Triple,
                      S.B_Homer,
                      S.B_RBI,
                      S.B_StolenBase,
                      B_CaughtStealing,
                      S.B_Walk,
                      S.B_StrikeOuts,
                      S.B_IntentWalk,
                      S.B_HitbyPitch,
                      S.B_SacrificeHit,
                      S.B_SacrificeFly,
                      S.B_GroundDoublePlay,
                      S.P_Win,
                      S.P_Loss,
                      S.P_GamesPlayed,
                      S.P_GameStarted,
                      S.P_CompleteGame,
                      S.P_Shutout,
                      S.P_Save,
                      S.P_OutsPitched,
                      S.P_Hits,
                      S.P_EarnedRun,
                      P_HomeRun,
                      S.P_Walk,
                      S.P_StrikeOut,
                      S.P_OppBattAvg,
                      S.P_ERA,
                      S.P_IntentWalk,
                      S.P_WildPitch,
                      S.P_HitBatter,
                      S.P_Balk,
                      S.P_BatterFaced,
                      P_GameFinished,
                      S.P_RunAllowed,
                      S.P_OppSacrificeHit,
                      S.P_OppSacrificeFly
                 FROM (
                          SELECT DISTINCT R.*,
                                          P._playerID,
                                          R._yearID - 1 AS statYear
                            FROM NLALTeamsRanks AS R
                                 INNER JOIN
                                 NLALPlayersTeams AS P ON R._yearID = P._yearID AND 
                                                          R._teamID = P._teamID
                      )
                      AS TEM
                      INNER JOIN
                      NLALRegularSeasonStats_byPlayer AS S ON TEM.statYear = S._yearID AND 
                                                              TEM._playerID = S._playerID
           )
           AS D
     GROUP BY _yearID,
              _lgID,
              _divID,
              _teamID,
              _Rank,
              _DivWin,
              _LgWin,
              _G,
              _W,
              _L,
              WinPerc,
              WinLosPerc;
