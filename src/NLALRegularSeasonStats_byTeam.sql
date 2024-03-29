CREATE VIEW NLALRegularSeasonStats_byTeam AS
    SELECT _yearID,
           Team,
           _teamID,
           sum(F_GamePlayed) AS F_GamePlayed,
           sum(F_GameStarted) AS F_GameStarted,
           sum(F_OutPlayed) AS F_OutPlayed,
           sum(F_Putout) AS F_Putout,
           sum(F_Assist) AS F_Assist,
           sum(F_Error) AS F_Error,
           sum(F_DoublePlay) AS F_DoublePlay,
           sum(F_CatcherPassedBall) AS F_CatcherPassedBall,
           sum(F_CatcherWildPitch) AS F_CatcherWildPitch,
           sum(F_CatcherOppStolenBase) AS F_CatcherOppStolenBase,
           sum(F_CatcherOppCaughtStealing) AS F_CatcherOppCaughtStealing,
           sum(F_ZoneRating) AS F_ZoneRating,
           sum(B_AtBats) AS B_AtBats,
           sum(B_Run) AS B_Run,
           sum(B_Hits) AS B_Hits,
           sum(B_Double) AS B_Double,
           sum(B_Triple) AS B_Triple,
           sum(B_Homer) AS B_Homer,
           sum(B_RBI) AS B_RBI,
           sum(B_StolenBase) AS B_StolenBase,
           sum(B_CaughtStealing) AS B_CaughtStealing,
           sum(B_Walk) AS B_Walk,
           sum(B_StrikeOuts) AS B_StrikeOuts,
           sum(B_IntentWalk) AS B_IntentWalk,
           sum(B_HitbyPitch) AS B_HitbyPitch,
           sum(B_SacrificeHit) AS B_SacrificeHit,
           sum(B_SacrificeFly) AS B_SacrificeFly,
           sum(B_GroundDoublePlay) AS B_GroundDoublePlay,
           sum(P_Win) AS P_Win,
           sum(P_Loss) AS P_Loss,
           sum(P_GamesPlayed) AS P_GamesPlayed,
           sum(P_GameStarted) AS P_GameStarted,
           sum(P_CompleteGame) AS P_CompleteGame,
           sum(P_Shutout) AS P_Shutout,
           sum(P_Save) AS P_Save,
           sum(P_OutsPitched) AS P_OutsPitched,
           sum(P_Hits) AS P_Hits,
           sum(P_EarnedRun) AS P_EarnedRun,
           sum(P_HomeRun) AS P_HomeRun,
           sum(P_Walk) AS P_Walk,
           sum(P_StrikeOut) AS P_StrikeOut,
           sum(P_OppBattAvg) AS P_OppBattAvg,
           avg(P_ERA) AS P_ERA,
           sum(P_IntentWalk) AS P_IntentWalk,
           sum(P_WildPitch) AS P_WildPitch,
           sum(P_HitBatter) AS P_HitBatter,
           sum(P_Balk) AS P_Balk,
           sum(P_BatterFaced) AS P_BatterFaced,
           sum(P_GameFinished) AS P_GameFinished,
           sum(P_RunAllowed) AS P_RunAllowed,
           sum(P_OppSacrificeHit) AS P_OppSacrificeHit,
           sum(P_OppSacrificeFly) AS P_OppSacrificeFly
      FROM NLALRegularSeasonStats_byPlayerStintTeam
     GROUP BY _yearID,
              Team,
              _teamID;
