"use client";

import { useEffect, useMemo, useState } from "react";
import { ImpactBars, ImpactDatum, impactLegend } from "../charts/bars";
import { Figure, Legend } from "../charts/frame";
import {
  PlayerProjection,
  TeamProjection,
  loadPlayerProjections,
  loadTeamProjections,
} from "../lib/data";
import {
  fmtRating,
  linear,
  niceTicks,
  polarity,
  symmetricBound,
} from "../lib/viz";

export function ProjectionsView({
  onPlayer,
}: {
  onPlayer: (id: number) => void;
}) {
  const [teams, setTeams] = useState<TeamProjection[]>([]);
  const [players, setPlayers] = useState<PlayerProjection[]>([]);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    let live = true;
    Promise.all([loadTeamProjections(), loadPlayerProjections()])
      .then(([nextTeams, nextPlayers]) => {
        if (!live) return;
        setTeams(nextTeams);
        setPlayers(nextPlayers);
        setState("ready");
      })
      .catch(() => live && setState("error"));
    return () => {
      live = false;
    };
  }, []);

  const rankedTeams = useMemo(
    () =>
      [...teams].sort(
        (a, b) => b.projected_net_rating - a.projected_net_rating,
      ),
    [teams],
  );
  const topPlayers = useMemo<ImpactDatum[]>(
    () =>
      [...players]
        .sort((a, b) => b.projected_net - a.projected_net)
        .slice(0, 15)
        .map((row) => ({
          id: row.PLAYER_ID,
          name: row.PLAYER_NAME,
          team: row.TEAM_ABBREVIATION,
          season: row.projected_net === undefined ? 0 : 2027,
          offense: row.projected_offense,
          defense: row.projected_defense,
          net: row.projected_net,
          poss: 0,
        })),
    [players],
  );

  const season = teams[0]?.projection_season ?? 2027;

  if (state === "error") {
    return (
      <div className="empty">
        <b>Projections unavailable.</b>
        <span>This snapshot has no projection shards.</span>
      </div>
    );
  }

  return (
    <>
      <div className="page-head">
        <div>
          <p className="kicker">{season} baseline</p>
          <h1>Projections</h1>
        </div>
        <span className="meta">Research baseline · not a forecast</span>
      </div>

      <p className="lede" style={{ marginTop: 0, marginBottom: 20 }}>
        This ages each returning player’s latent rating one year and holds every
        team’s roster and minutes fixed. Trades, rookies, injuries, and schedule
        are absent, so treat it as a baseline the models can be measured against
        — not a prediction of the standings.
      </p>

      <div className="grid">
        <Figure
          kicker="Teams"
          title="Projected net rating"
          note="Rosters and minutes are frozen at last season’s returning players. Season 2027 stays reserved for confirmation, so nothing here is scored against it."
          table={
            <table className="mini">
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Team</th>
                  <th scope="col">Net</th>
                  <th scope="col">Win pace</th>
                  <th scope="col">Players</th>
                </tr>
              </thead>
              <tbody>
                {rankedTeams.map((row, position) => (
                  <tr key={row.TEAM_ABBREVIATION}>
                    <td>{position + 1}</td>
                    <td>{row.TEAM_ABBREVIATION}</td>
                    <td>{fmtRating(row.projected_net_rating)}</td>
                    <td>{row.projected_win_pace.toFixed(1)}</td>
                    <td>{row.players}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        >
          <TeamColumns rows={rankedTeams} />
        </Figure>

        <Figure
          kicker="Players"
          title="Fifteen highest projected"
          legend={<Legend items={impactLegend} />}
          note="Aged one year from the latent rating. Select a row to open the player’s history."
          table={
            <table className="mini">
              <thead>
                <tr>
                  <th scope="col">Player</th>
                  <th scope="col">Team</th>
                  <th scope="col">Age</th>
                  <th scope="col">Off</th>
                  <th scope="col">Def</th>
                  <th scope="col">Net</th>
                </tr>
              </thead>
              <tbody>
                {[...players]
                  .sort((a, b) => b.projected_net - a.projected_net)
                  .slice(0, 60)
                  .map((row) => (
                    <tr key={row.PLAYER_ID}>
                      <td>{row.PLAYER_NAME}</td>
                      <td>{row.TEAM_ABBREVIATION}</td>
                      <td>{Math.round(row.AGE)}</td>
                      <td>{fmtRating(row.projected_offense)}</td>
                      <td>{fmtRating(row.projected_defense)}</td>
                      <td>{fmtRating(row.projected_net)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          }
        >
          <ImpactBars rows={topPlayers} onSelect={(row) => onPlayer(row.id)} />
        </Figure>
      </div>
    </>
  );
}

/**
 * Team columns around zero. One series, so the diverging scale carries polarity
 * and no legend box is needed — the title says what is plotted.
 */
function TeamColumns({ rows }: { rows: TeamProjection[] }) {
  if (!rows.length) return <div className="empty">No team projections.</div>;

  const width = 880;
  const height = 300;
  const pad = { left: 42, right: 16, top: 20, bottom: 44 };
  const bound = symmetricBound(
    rows.map((row) => row.projected_net_rating),
    2,
  );
  const y = linear([-bound, bound], [height - pad.bottom, pad.top]);
  const slot = (width - pad.left - pad.right) / rows.length;
  const barWidth = Math.min(24, slot - 2); // cap the mark; the leftover is air

  return (
    <div className="chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Projected net rating by team"
      >
        {niceTicks(-bound, bound, 5).map((tick) => (
          <g key={tick}>
            <line
              className={tick === 0 ? "zero-line" : "grid-line"}
              x1={pad.left}
              x2={width - pad.right}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text
              className="tick"
              x={pad.left - 9}
              y={y(tick) + 4}
              textAnchor="end"
            >
              {tick > 0 ? `+${tick}` : tick}
            </text>
          </g>
        ))}
        {rows.map((row, index) => {
          const value = row.projected_net_rating;
          const centre = pad.left + slot * (index + 0.5);
          const top = Math.min(y(value), y(0));
          const size = Math.max(1, Math.abs(y(value) - y(0)));
          const radius = Math.min(4, size);
          const up = value >= 0;
          return (
            <g key={row.TEAM_ABBREVIATION}>
              <path
                className="bar"
                fill={polarity(value, bound)}
                d={
                  up
                    ? `M${centre - barWidth / 2},${top + size} v-${size - radius} a${radius},${radius} 0 0 1 ${radius},-${radius} h${barWidth - radius * 2} a${radius},${radius} 0 0 1 ${radius},${radius} v${size - radius} z`
                    : `M${centre - barWidth / 2},${top} v${size - radius} a${radius},${radius} 0 0 0 ${radius},${radius} h${barWidth - radius * 2} a${radius},${radius} 0 0 0 ${radius},-${radius} v-${size - radius} z`
                }
              >
                <title>{`${row.TEAM_ABBREVIATION}: ${fmtRating(value)} net, ${row.projected_win_pace.toFixed(1)} win pace`}</title>
              </path>
              <text
                className="tick"
                x={centre}
                y={up ? y(0) + 15 : y(0) - 7}
                textAnchor="middle"
                style={{ fontSize: 9 }}
              >
                {row.TEAM_ABBREVIATION}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
