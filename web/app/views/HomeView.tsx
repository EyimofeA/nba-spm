"use client";

import { useMemo } from "react";
import { ImpactBars, ImpactDatum, impactLegend } from "../charts/bars";
import { Figure, Legend } from "../charts/frame";
import { Catalog, LeaderboardRow, possessions, rating } from "../lib/data";
import { fmtRating } from "../lib/viz";

export function HomeView({
  catalog,
  rows,
  onGo,
  onPlayer,
}: {
  catalog: Catalog;
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
        <p className="lede">
          Every rating on this site is points per 100 possessions. Offense and
          defense are estimated separately, and both are positive when the
          player helps. The ratings describe seasons that already happened —
          they are not forecasts.
        </p>
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

      <div className="section-head">
        <div>
          <p className="kicker">Three models</p>
          <h2>Built in this order</h2>
        </div>
        <span className="meta">{catalog.methods.aio_equation}</span>
      </div>
      <div className="grid three">
        <article className="card">
          <p className="kicker">Step one · SPM</p>
          <h3>Statistics</h3>
          <p className="note" style={{ marginTop: 8 }}>
            A statistical model that predicts one season of impact from box,
            tracking, playtype, and matchup features. The rated season is never
            in its own training labels.
          </p>
        </article>
        <article className="card">
          <p className="kicker">Reference · RAPM</p>
          <h3>Lineups</h3>
          <p className="note" style={{ marginTop: 8 }}>
            A ridge regression on possessions that separates each player from
            the other nine on the floor. RAPM uses no prior, so it is the
            independent reference rather than part of AIO.
          </p>
        </article>
        <article className="card">
          <p className="kicker">Step two · AIO</p>
          <h3>Combined</h3>
          <p className="note" style={{ marginTop: 8 }}>
            The all-in-one rating. SPM sets the center, then one centered ridge
            fit updates it with that season’s possessions.
          </p>
        </article>
      </div>

    </>
  );
}
