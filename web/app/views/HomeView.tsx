"use client";

import { useMemo } from "react";
import { ImpactBars, ImpactDatum, impactLegend } from "../charts/bars";
import { Figure, Legend } from "../charts/frame";
import { Catalog, LeaderboardRow, possessions, rating } from "../lib/data";
import { fmtInt, fmtRating } from "../lib/viz";

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
  const seasons = catalog.catalog.seasons;
  const span = seasons.length
    ? `${seasons[0] - 1}–${String(seasons[seasons.length - 1]).slice(2)}`
    : "—";

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

  const totalPoss = rows.reduce((sum, row) => sum + possessions(row), 0);
  const netForward = catalog.validation.walk_forward.find(
    (row) => row.component === "net",
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

      <div className="kpi-row" style={{ margin: "22px 0" }}>
        <div className="tile">
          <div className="tile-label">Seasons covered</div>
          <div className="tile-value">{seasons.length}</div>
          <div className="tile-sub">{span}</div>
        </div>
        <div className="tile">
          <div className="tile-label">Rated players</div>
          <div className="tile-value">{fmtInt(rows.length)}</div>
          <div className="tile-sub">Latest season</div>
        </div>
        <div className="tile">
          <div className="tile-label">Possessions</div>
          <div className="tile-value">
            {totalPoss ? `${Math.round(totalPoss / 1000)}K` : "—"}
          </div>
          <div className="tile-sub">Smaller side, summed</div>
        </div>
        <div className="tile">
          <div className="tile-label">Held-out net r</div>
          <div className="tile-value">
            {netForward ? netForward.correlation.toFixed(2) : "—"}
          </div>
          <div className="tile-sub">
            Walk-forward, {netForward?.folds ?? 0} folds
          </div>
        </div>
      </div>

      <Figure
        kicker={`AIO · ${seasons[seasons.length - 1] ?? ""}`}
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
            the other nine on the floor. Normal RAPM uses no prior, so it is the
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

      <div className="section-head">
        <div>
          <p className="kicker">How to read a rating</p>
          <h2>Four rules</h2>
        </div>
      </div>
      <section className="card">
        <ol className="def-list">
          <li>
            <b>Scale</b>
            <span>
              +3.0 means the model estimates about three more points per 100
              possessions for the team with that player on the floor, after
              separating teammates and opponents.
            </span>
          </li>
          <li>
            <b>Sides</b>
            <span>
              Offense and defense add to net. A positive defensive rating means
              the player prevents points.
            </span>
          </li>
          <li>
            <b>Sample</b>
            <span>
              The possession figure is the smaller of offensive and defensive
              exposure. Small samples pull toward the model’s center, so read
              low-possession rows with care.
            </span>
          </li>
          <li>
            <b>Scope</b>
            <span>
              These ratings describe past seasons. They are not forecasts, and
              no rating here carries a published interval yet.
            </span>
          </li>
        </ol>
      </section>
    </>
  );
}
