"use client";

import { useMemo, useState } from "react";
import { Figure, ScaleLegend } from "../charts/frame";
import { Landscape, LandscapeDatum, netBoundFor } from "../charts/scatter";
import {
  LeaderboardRow,
  ModelId,
  possessions,
  rating,
  resolveModel,
} from "../lib/data";
import { fmtInt, fmtRating } from "../lib/viz";
import { MinPossField, ModelField, SeasonField, TeamField } from "./controls";

export function LandscapeView({
  rows,
  seasons,
  season,
  onSeason,
  model,
  onModel,
  minPoss,
  onMinPoss,
  onPlayer,
  selected,
}: {
  rows: LeaderboardRow[];
  seasons: number[];
  season: number;
  onSeason: (season: number) => void;
  model: ModelId;
  onModel: (model: ModelId) => void;
  minPoss: number;
  onMinPoss: (value: number) => void;
  onPlayer: (id: number) => void;
  selected?: number;
}) {
  const [team, setTeam] = useState("All");
  const active = resolveModel(rows, model);

  const teams = useMemo(
    () =>
      [
        ...new Set(
          rows
            .map((row) => row.TEAM_ABBREVIATION)
            .filter((v): v is string => Boolean(v)),
        ),
      ].sort(),
    [rows],
  );

  const points = useMemo<LandscapeDatum[]>(
    () =>
      rows
        .filter((row) => possessions(row) >= Math.max(minPoss, 300))
        .filter((row) => team === "All" || row.TEAM_ABBREVIATION === team)
        .flatMap((row) => {
          const offense = rating(row, active.prefix, "offense");
          const defense = rating(row, active.prefix, "defense");
          const net = rating(row, active.prefix, "net");
          return offense === undefined ||
            defense === undefined ||
            net === undefined
            ? []
            : [
                {
                  id: row.PLAYER_ID,
                  name: row.PLAYER_NAME,
                  team: row.TEAM_ABBREVIATION,
                  season: row.Season,
                  offense,
                  defense,
                  net,
                  poss: possessions(row),
                },
              ];
        }),
    [rows, minPoss, team, active.prefix],
  );

  const quadrant = (row: LandscapeDatum) =>
    row.offense >= 0
      ? row.defense >= 0
        ? "Two-way"
        : "Offense first"
      : row.defense >= 0
        ? "Defense first"
        : "Below average";

  const counts = useMemo(() => {
    const tally = new Map<string, number>();
    for (const row of points)
      tally.set(quadrant(row), (tally.get(quadrant(row)) ?? 0) + 1);
    return tally;
  }, [points]);

  return (
    <>
      <div className="page-head">
        <div>
          <p className="kicker">Landscape</p>
          <h1>Offense against defense</h1>
        </div>
      </div>

      <div className="filters">
        <SeasonField seasons={seasons} value={season} onChange={onSeason} />
        <ModelField rows={rows} value={model} onChange={onModel} />
        <TeamField teams={teams} value={team} onChange={setTeam} />
        <MinPossField value={minPoss} onChange={onMinPoss} />
      </div>

      <div className="kpi-row" style={{ marginBottom: 14 }}>
        {["Two-way", "Offense first", "Defense first", "Below average"].map(
          (label) => (
            <div className="tile" key={label}>
              <div className="tile-label">{label}</div>
              <div className="tile-value">{counts.get(label) ?? 0}</div>
              <div className="tile-sub">
                {points.length
                  ? Math.round(((counts.get(label) ?? 0) / points.length) * 100)
                  : 0}
                % of field
              </div>
            </div>
          ),
        )}
      </div>

      <Figure
        kicker={active.label}
        title="The two-way plane"
        legend={
          <ScaleLegend
            caption="Net"
            low={`−${netBoundFor(points).toFixed(1)}`}
            high={`+${netBoundFor(points).toFixed(1)}`}
          />
        }
        note={
          <>
            Each dot is one player. Dot size is possessions on the
            smaller side, so thin samples look thin. Colour restates net — the
            sum of the two axes — which is what makes the two-way corner read at
            a glance. Only the highest-net names are labelled; hover or open the
            table for the rest. A 300-possession floor applies here so the plane
            is not dominated by noise.
          </>
        }
        table={
          <table className="mini">
            <thead>
              <tr>
                <th scope="col">Player</th>
                <th scope="col">Group</th>
                <th scope="col">Off</th>
                <th scope="col">Def</th>
                <th scope="col">Net</th>
                <th scope="col">Poss</th>
              </tr>
            </thead>
            <tbody>
              {[...points]
                .sort((a, b) => b.net - a.net)
                .map((row) => (
                  <tr key={row.id}>
                    <td>{row.name}</td>
                    <td>{quadrant(row)}</td>
                    <td>{fmtRating(row.offense)}</td>
                    <td>{fmtRating(row.defense)}</td>
                    <td>{fmtRating(row.net)}</td>
                    <td>{fmtInt(row.poss)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        }
      >
        <Landscape
          rows={points}
          highlight={selected}
          onSelect={(row) => onPlayer(row.id)}
        />
      </Figure>
    </>
  );
}
