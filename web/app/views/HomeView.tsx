"use client";

import { useMemo } from "react";
import { ImpactBars, ImpactDatum, impactLegend } from "../charts/bars";
import { Figure, Legend } from "../charts/frame";
import { LeaderboardRow, possessions, rating } from "../lib/data";
import { fmtRating } from "../lib/viz";

export function HomeView({
  rows,
  onGo,
  onPlayer,
}: {
  rows: LeaderboardRow[];
  onGo: (tab: "ratings" | "landscape" | "research") => void;
  onPlayer: (id: number) => void;
}) {
  const displayedSeason = rows[0]?.Season;

  const leaders = useMemo<ImpactDatum[]>(
    () =>
      rows
        .filter((row) => possessions(row) >= 1000)
        .map((row) => ({
          id: row.PLAYER_ID,
          name: row.PLAYER_NAME,
          team: row.TEAM_ABBREVIATION,
          season: row.Season,
          offense: rating(row, "aio_", "offense") ?? 0,
          defense: rating(row, "aio_", "defense") ?? 0,
          net: rating(row, "aio_", "net") ?? 0,
          poss: possessions(row),
        }))
        .sort((a, b) => b.net - a.net)
        .slice(0, 10),
    [rows],
  );

  return (
    <>
      <section
        style={{
          padding: "26px 0 34px",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <p className="kicker">NBA Impact Lab</p>
        <h1 style={{ maxWidth: "18ch", fontSize: "clamp(38px, 6.4vw, 74px)" }}>
          One number.
          <br />
          More context.
        </h1>
        <div
          style={{ display: "flex", gap: 8, marginTop: 24, flexWrap: "wrap" }}
        >
          <button type="button" className="btn" onClick={() => onGo("ratings")}>
            View the ratings
          </button>
          <button
            type="button"
            className="btn ghost"
            onClick={() => onGo("landscape")}
          >
            Explore the landscape
          </button>
          <button
            type="button"
            className="btn ghost"
            onClick={() => onGo("research")}
          >
            Read what held up
          </button>
        </div>
      </section>

      <Figure
        kicker={`AIO · ${displayedSeason ?? ""}`}
        title="This season’s ten most valuable"
        legend={<Legend items={impactLegend} />}
        note="Players with 1,000 or more possessions on the smaller side, ranked by AIO net. Select a row to open the player page."
        table={
          <table className="mini">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Player</th>
                <th scope="col">Off</th>
                <th scope="col">Def</th>
                <th scope="col">Net</th>
              </tr>
            </thead>
            <tbody>
              {leaders.map((row, position) => (
                <tr key={row.id}>
                  <td>{position + 1}</td>
                  <td>{row.name}</td>
                  <td>{fmtRating(row.offense)}</td>
                  <td>{fmtRating(row.defense)}</td>
                  <td>{fmtRating(row.net)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        }
      >
        <ImpactBars rows={leaders} onSelect={(row) => onPlayer(row.id)} />
      </Figure>
    </>
  );
}
