"use client";

import { useMemo, useState } from "react";
import { Figure } from "../charts/frame";
import { MultiLine } from "../charts/lines";
import { ScoreStateBubble } from "../charts/scatter";
import { MatchupLabPayload, RapmLabPayload } from "../lib/data";
import { fmtInt, fmtRating } from "../lib/viz";
import { MatchupsView } from "./MatchupsView";

type Project = "tests" | "replications" | "rubberband" | "age" | "matchups";
type RubberbandBasis = "actual_clock" | "possession_progress";
type RubberbandModel = "normal" | "clock" | "possession";

const statusLabel: Record<string, string> = {
  won: "Won",
  lost: "Lost",
  built: "Built",
  estimate: "Estimate",
};

function formatLabValue(key: string, value: string | number | null) {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  if (/season|year|window_(start|end)/i.test(key)) {
    return String(Math.trunc(value));
  }
  if (/games|rank|possessions|exposure/i.test(key)) {
    return fmtInt(value);
  }
  return fmtRating(value);
}

export function RapmLabView({
  lab,
  matchupLab,
}: {
  lab: RapmLabPayload | null;
  matchupLab: MatchupLabPayload | null;
}) {
  const [project, setProject] = useState<Project>("tests");
  const [replicationBoardId, setReplicationBoardId] = useState("darko-wowy-2026");
  const [timeBucket, setTimeBucket] = useState(7);
  const [rubberbandBasis, setRubberbandBasis] =
    useState<RubberbandBasis>("actual_clock");
  const [rubberbandModel, setRubberbandModel] =
    useState<RubberbandModel>("clock");
  const [leaderboardId, setLeaderboardId] = useState("rubberband-ratings");
  const [leaderboardSortKey, setLeaderboardSortKey] = useState<string | null>(
    null,
  );
  const [leaderboardSortOrder, setLeaderboardSortOrder] =
    useState<"asc" | "desc">("desc");

  const comparisonCoefficients = useMemo(
    () =>
      (lab?.rubberband.comparison_coefficients ?? []).filter(
        (row) => row.basis === rubberbandBasis,
      ),
    [lab, rubberbandBasis],
  );
  const coefficient = comparisonCoefficients.find(
    (row) => row.time_bucket === timeBucket,
  );
  const curve = useMemo(
    () => {
      if (!coefficient || !lab) return [];
      return Array.from({ length: 13 }, (_, index) => -30 + index * 5).map(
        (margin) => ({
          x: margin,
          y:
            coefficient.slope_points_per_100_per_margin_point *
            Math.max(
              -lab.rubberband.margin_clip,
              Math.min(lab.rubberband.margin_clip, margin),
            ),
        }),
      );
    },
    [coefficient, lab],
  );
  const slopeComparison = useMemo(
    () =>
      (["actual_clock", "possession_progress"] as const).map((basis) => ({
        label: basis === "actual_clock" ? "Actual clock" : "Possession progress",
        color:
          basis === "actual_clock" ? "var(--series-1)" : "var(--series-2)",
        points: (lab?.rubberband.comparison_coefficients ?? [])
          .filter((row) => row.basis === basis)
          .map((row) => ({
            x: row.time_bucket + 1,
            y: row.slope_points_per_100_per_margin_point,
          })),
      })),
    [lab],
  );
  const selectedLeaderboard = lab?.leaderboards.find(
    (row) => row.id === leaderboardId,
  );
  const selectedReplicationBoard = lab?.replication_leaderboards.find(
    (row) => row.id === replicationBoardId,
  );
  const sortedLeaderboardRows = useMemo(() => {
    if (!selectedLeaderboard) return [];
    if (!leaderboardSortKey) return selectedLeaderboard.rows;
    const direction = leaderboardSortOrder === "asc" ? 1 : -1;
    return selectedLeaderboard.rows
      .map((row, index) => ({ row, index }))
      .sort((left, right) => {
        const a = left.row[leaderboardSortKey];
        const b = right.row[leaderboardSortKey];
        if (a == null && b == null) return left.index - right.index;
        if (a == null) return 1;
        if (b == null) return -1;
        const comparison =
          typeof a === "number" && typeof b === "number"
            ? a - b
            : String(a).localeCompare(String(b));
        return comparison * direction || left.index - right.index;
      })
      .map(({ row }) => row);
  }, [leaderboardSortKey, leaderboardSortOrder, selectedLeaderboard]);
  const neutralNormal = lab?.rubberband.rapm_evaluation.find(
    (row) => row.variant === "normal" && row.prediction_mode === "neutral_player_only",
  );
  const neutralClock = lab?.rubberband.rapm_evaluation.find(
    (row) => row.variant === "clock" && row.prediction_mode === "neutral_player_only",
  );
  const neutralPossession = lab?.rubberband.rapm_evaluation.find(
    (row) =>
      row.variant === "possession_progress" &&
      row.prediction_mode === "neutral_player_only",
  );
  const conditionalClock = lab?.rubberband.rapm_evaluation.find(
    (row) => row.variant === "clock" && row.prediction_mode === "conditional_score_path",
  );
  const conditionalPossession = lab?.rubberband.rapm_evaluation.find(
    (row) =>
      row.variant === "possession_progress" &&
      row.prediction_mode === "conditional_score_path",
  );
  const jeNormal = lab?.rubberband.je?.evaluation.find(
    (row) => row.variant === "normal",
  );
  const jeNeutral = lab?.rubberband.je?.evaluation.find(
    (row) =>
      row.variant === "je_categorical" &&
      row.prediction_mode === "neutral_player_only",
  );
  const jeConditional = lab?.rubberband.je?.evaluation.find(
    (row) =>
      row.variant === "je_categorical" &&
      row.prediction_mode === "conditional_score_path",
  );
  const rubberbandRatings = useMemo(() => {
    if (!lab) return [];
    return [...lab.rubberband.ratings]
      .sort((a, b) => {
        const aNet =
          rubberbandModel === "normal"
            ? a.normal_net
            : rubberbandModel === "clock"
              ? a.clock_net
              : a.possession_net;
        const bNet =
          rubberbandModel === "normal"
            ? b.normal_net
            : rubberbandModel === "clock"
              ? b.clock_net
              : b.possession_net;
        return bNet - aNet;
      })
      .slice(0, 100);
  }, [lab, rubberbandModel]);
  const segmentLabel = coefficient
    ? rubberbandBasis === "actual_clock"
      ? `${coefficient.minutes_elapsed_start?.toFixed(0)} to ${coefficient.minutes_elapsed_end?.toFixed(0)} minutes elapsed`
      : timeBucket === 7
        ? "175 plus completed possessions before the current possession"
        : `${timeBucket * 25} to ${timeBucket * 25 + 24} completed possessions before the current possession`
    : "Segment unavailable";

  function chooseLeaderboard(nextId: string) {
    setLeaderboardId(nextId);
    setLeaderboardSortKey(null);
    setLeaderboardSortOrder("desc");
  }

  function sortLeaderboard(key: string) {
    if (!selectedLeaderboard) return;
    if (leaderboardSortKey === key) {
      setLeaderboardSortOrder(
        leaderboardSortOrder === "desc" ? "asc" : "desc",
      );
      return;
    }
    const sample = selectedLeaderboard.rows.find((row) => row[key] != null)?.[
      key
    ];
    setLeaderboardSortKey(key);
    setLeaderboardSortOrder(typeof sample === "number" ? "desc" : "asc");
  }

  function leaderboardAriaSort(
    key: string,
  ): "ascending" | "descending" | undefined {
    if (leaderboardSortKey !== key) return undefined;
    return leaderboardSortOrder === "asc" ? "ascending" : "descending";
  }

  if (!lab) return <div className="empty">RAPM Lab data unavailable.</div>;

  return (
    <>
      <div className="page-head">
        <div>
          <p className="kicker">Local research</p>
          <h1>RAPM Lab</h1>
        </div>
      </div>

      <div className="filters">
        <div className="field">
          <span>Project</span>
          <div className="segmented" role="group" aria-label="RAPM Lab project">
            {([
              ["tests", "Tests"],
              ["replications", "Replications"],
              ["rubberband", "Rubber band"],
              ["age", "Age"],
              ["matchups", "Matchups"],
            ] as const).map(([id, label]) => (
              <button
                type="button"
                key={id}
                aria-pressed={project === id}
                onClick={() => setProject(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {project === "tests" && (
        <section>
          <div className="section-head lab-leaderboard-head">
            <div>
              <p className="kicker">Saved ratings</p>
              <h2>{selectedLeaderboard?.title ?? "Leaderboard"}</h2>
            </div>
            <label className="field compact-field">
              <span>Leaderboard</span>
              <select
                value={leaderboardId}
                onChange={(event) => chooseLeaderboard(event.target.value)}
              >
                {lab.leaderboards.map((board) => (
                  <option key={board.id} value={board.id}>{board.title}</option>
                ))}
              </select>
            </label>
          </div>
          {selectedLeaderboard && (
            <div className="table-wrap lab-leaderboard">
              <table className="data">
                <caption className="visually-hidden">
                  {selectedLeaderboard.title}
                </caption>
                <thead>
                  <tr>
                    {selectedLeaderboard.columns.map((column, index) => (
                      <th
                        key={column.key}
                        className={index === 0 ? "left" : undefined}
                        scope="col"
                        aria-sort={leaderboardAriaSort(column.key)}
                      >
                        <button
                          type="button"
                          onClick={() => sortLeaderboard(column.key)}
                        >
                          {column.label}
                          {leaderboardSortKey === column.key && (
                            <span className="arrow">
                              {leaderboardSortOrder === "asc" ? "▲" : "▼"}
                            </span>
                          )}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedLeaderboardRows.map((row, index) => (
                    <tr key={`${selectedLeaderboard.id}-${index}`}>
                      {selectedLeaderboard.columns.map((column, columnIndex) => (
                        <td
                          key={column.key}
                          className={columnIndex === 0 ? "left name" : undefined}
                        >
                          {formatLabValue(column.key, row[column.key])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="section-head lab-results-head">
            <div>
              <p className="kicker">Saved experiments</p>
              <h2>Test, result, decision</h2>
            </div>
          </div>
          <div className="table-wrap rapm-lab-results">
            <table className="data">
              <caption className="visually-hidden">
                RAPM research tests, results, and decisions
              </caption>
              <thead>
                <tr>
                  <th className="left" scope="col"><button type="button" disabled>Experiment</button></th>
                  <th className="left" scope="col"><button type="button" disabled>Testing</button></th>
                  <th className="left" scope="col"><button type="button" disabled>Result</button></th>
                  <th className="left" scope="col"><button type="button" disabled>Decision</button></th>
                </tr>
              </thead>
              <tbody>
                {lab.experiments.map((row) => {
                  const board = lab.leaderboards.find(
                    (candidate) => candidate.experiment_id === row.id,
                  );
                  return (
                    <tr
                      key={row.id}
                      className={board ? "clickable" : undefined}
                      onClick={() => board && chooseLeaderboard(board.id)}
                    >
                      <td className="left name">
                        {row.title}
                        <span className={`lab-status ${row.status}`}>
                          {statusLabel[row.status]}
                        </span>
                      </td>
                      <td className="left lab-copy">{row.test}</td>
                      <td className="left lab-copy">{row.result}</td>
                      <td className="left lab-copy">{row.decision}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {project === "replications" && (
        <div className="grid">
        <section>
          <div className="section-head">
            <div>
              <p className="kicker">Independent reference checks</p>
              <h2>Replication status</h2>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data">
              <caption className="visually-hidden">
                Public metric replication and reference status
              </caption>
              <thead>
                <tr>
                  <th className="left" scope="col"><button type="button" disabled>Metric</button></th>
                  <th className="left" scope="col"><button type="button" disabled>Build</button></th>
                  <th className="left" scope="col"><button type="button" disabled>Status</button></th>
                  <th scope="col"><button type="button" disabled>Matched</button></th>
                  <th scope="col"><button type="button" disabled>Pearson</button></th>
                  <th className="left" scope="col"><button type="button" disabled>Decision</button></th>
                </tr>
              </thead>
              <tbody>
                {lab.replications.map((row) => (
                  <tr key={`${row.metric}-${row.build}`}>
                    <td className="left name">{row.metric}</td>
                    <td className="left lab-copy">{row.build}</td>
                    <td className="left">{row.status.replaceAll("_", " ")}</td>
                    <td>{fmtInt(row.matched_rows)}</td>
                    <td>{row.pearson == null ? "—" : row.pearson.toFixed(3)}</td>
                    <td className="left lab-copy">{row.decision}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        <section>
          <div className="section-head">
            <div>
              <p className="kicker">Published ratings</p>
              <h2>{selectedReplicationBoard?.title ?? "Leaderboard"}</h2>
            </div>
            <label className="field compact-field">
              <span>Metric</span>
              <select
                value={replicationBoardId}
                onChange={(event) => setReplicationBoardId(event.target.value)}
              >
                {lab.replication_leaderboards.map((board) => (
                  <option key={board.id} value={board.id}>{board.title}</option>
                ))}
              </select>
            </label>
          </div>
          {selectedReplicationBoard && (
            <div className="table-wrap lab-leaderboard">
              <table className="data">
                <thead>
                  <tr>
                    <th className="left">Player</th>
                    <th className="left">Team</th>
                    <th>Offense</th>
                    <th>Defense</th>
                    <th>Net</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedReplicationBoard.rows.map((row, index) => (
                    <tr key={`${row.player}-${index}`}>
                      <td className="left name">{row.player}</td>
                      <td className="left">{row.team ?? "—"}</td>
                      <td>{fmtRating(row.offense)}</td>
                      <td>{fmtRating(row.defense)}</td>
                      <td className="headline">{fmtRating(row.net)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
        </div>
      )}

      {project === "rubberband" && (
        <div className="grid">
          <Figure
            kicker="Five-point buckets · 2014 to 2026"
            title="Scoring changes with the lead"
            note={`The player-rating candidate changed held-out RMSE by ${fmtRating(lab.rubberband.five_point.diagnostic.rmse_delta_candidate_minus_baseline)}. The paired 95% interval was ${fmtRating(lab.rubberband.five_point.diagnostic.paired_game_bootstrap.lower_95)} to ${fmtRating(lab.rubberband.five_point.diagnostic.paired_game_bootstrap.upper_95)}.`}
            table={
              <table className="mini">
                <thead>
                  <tr><th>Lead</th><th>Effect / 100</th><th>Possessions</th></tr>
                </thead>
                <tbody>
                  {lab.rubberband.five_point.curve.map((row) => (
                    <tr key={row.margin}>
                      <td>{row.margin > 0 ? `+${row.margin}` : row.margin}</td>
                      <td>{fmtRating(row.effect_points_per_100_vs_tie)}</td>
                      <td>{fmtInt(row.possessions)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <ScoreStateBubble rows={lab.rubberband.five_point.curve} />
          </Figure>

          {lab.rubberband.je && (
            <>
              <Figure
                kicker="JE replication · fit 2014 to 2025"
                title="Score margin effect inside RAPM"
                note={`2026 neutral player-only RMSE: ${jeNormal?.margin_rmse.toFixed(3)} normal, ${jeNeutral?.margin_rmse.toFixed(3)} adjusted. Conditional observed-path RMSE: ${jeConditional?.margin_rmse.toFixed(3)}.`}
                table={
                  <table className="mini">
                    <thead>
                      <tr>
                        <th>Lead</th>
                        <th>Effect / 100</th>
                        <th>Possessions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {lab.rubberband.je.curve.map((row) => (
                        <tr key={row.margin}>
                          <td>
                            {row.margin > 0 ? `+${row.margin}` : row.margin}
                          </td>
                          <td>
                            {fmtRating(row.effect_points_per_100_vs_tie)}
                          </td>
                          <td>{fmtInt(row.possessions)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                }
              >
                <ScoreStateBubble rows={lab.rubberband.je.curve} />
              </Figure>

              <section className="card">
                <div className="card-head">
                  <div>
                    <p className="kicker">Held-out check · 2026</p>
                    <h2>The effect exists; the rating adjustment loses</h2>
                  </div>
                </div>
                <table className="mini rubberband-summary">
                  <tbody>
                    <tr>
                      <th>Trailing by 20</th>
                      <td>{fmtRating(lab.rubberband.je.effects["-20"])}</td>
                    </tr>
                    <tr>
                      <th>Trailing by 10</th>
                      <td>{fmtRating(lab.rubberband.je.effects["-10"])}</td>
                    </tr>
                    <tr>
                      <th>Leading by 10</th>
                      <td>{fmtRating(lab.rubberband.je.effects["10"])}</td>
                    </tr>
                    <tr>
                      <th>Leading by 20</th>
                      <td>{fmtRating(lab.rubberband.je.effects["20"])}</td>
                    </tr>
                    <tr>
                      <th>Neutral RMSE</th>
                      <td>
                        {jeNormal?.margin_rmse.toFixed(3)} →{" "}
                        {jeNeutral?.margin_rmse.toFixed(3)}
                      </td>
                    </tr>
                    <tr>
                      <th>Paired 95% change</th>
                      <td>
                        {fmtRating(
                          lab.rubberband.je.bootstrap_vs_normal
                            .neutral_player_only.lower_95,
                        )}{" "}
                        to{" "}
                        {fmtRating(
                          lab.rubberband.je.bootstrap_vs_normal
                            .neutral_player_only.upper_95,
                        )}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </section>
            </>
          )}

          <section className="card">
            <div className="card-head">
              <div>
                <p className="kicker">2024 to 2026</p>
                <h2>Clock and possession-progress adjustment</h2>
              </div>
              <span className="meta">
                {fmtInt(lab.rubberband.possessions)} possessions
              </span>
            </div>
            <table className="mini rubberband-summary">
              <tbody>
                <tr>
                  <th>Curve agreement</th>
                  <td>{lab.rubberband.context_effect.slope_correlation.toFixed(3)}</td>
                </tr>
                <tr>
                  <th>Clock residual RMSE</th>
                  <td>
                    {lab.rubberband.context_effect.clock.baseline_rmse.toFixed(6)} → {lab.rubberband.context_effect.clock.candidate_rmse.toFixed(6)}
                  </td>
                </tr>
                <tr>
                  <th>Possession residual RMSE</th>
                  <td>
                    {lab.rubberband.context_effect.possession_progress.baseline_rmse.toFixed(6)} → {lab.rubberband.context_effect.possession_progress.candidate_rmse.toFixed(6)}
                  </td>
                </tr>
                <tr>
                  <th>Neutral RAPM RMSE</th>
                  <td>
                    {neutralNormal?.margin_rmse.toFixed(3)} normal · {neutralClock?.margin_rmse.toFixed(3)} clock · {neutralPossession?.margin_rmse.toFixed(3)} possession
                  </td>
                </tr>
                <tr>
                  <th>Neutral RAPM correlation</th>
                  <td>
                    {neutralNormal?.margin_correlation.toFixed(3)} normal · {neutralClock?.margin_correlation.toFixed(3)} clock · {neutralPossession?.margin_correlation.toFixed(3)} possession
                  </td>
                </tr>
                <tr>
                  <th>Mean rating movement</th>
                  <td>
                    {lab.rubberband.rating_effect.clock_mean_absolute_net_change.toFixed(3)} clock · {lab.rubberband.rating_effect.possession_mean_absolute_net_change.toFixed(3)} possession
                  </td>
                </tr>
              </tbody>
            </table>
            <p className="note">{lab.rubberband.decision}</p>
          </section>

          <Figure
            kicker="Empirical effect"
            title="Margin slope grows late in the game"
            note={`The two eight-segment curves correlate ${lab.rubberband.context_effect.slope_correlation.toFixed(3)}. Segment 8 is 42 to 48 minutes or combined possession 176 onward.`}
            table={
              <table className="mini">
                <thead>
                  <tr><th>Segment</th><th>Clock</th><th>Possession</th></tr>
                </thead>
                <tbody>
                  {Array.from({ length: 8 }, (_, index) => {
                    const clock = lab.rubberband.comparison_coefficients.find(
                      (row) => row.basis === "actual_clock" && row.time_bucket === index,
                    );
                    const possession = lab.rubberband.comparison_coefficients.find(
                      (row) => row.basis === "possession_progress" && row.time_bucket === index,
                    );
                    return (
                      <tr key={index}>
                        <td>{index + 1}</td>
                        <td>{clock ? fmtRating(clock.slope_points_per_100_per_margin_point) : "—"}</td>
                        <td>{possession ? fmtRating(possession.slope_points_per_100_per_margin_point) : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            }
          >
            <MultiLine
              series={slopeComparison}
              xFormat={(value) => String(value)}
              yFormat={fmtRating}
              xTitle="Game segment, early to late"
              zeroBaseline
            />
          </Figure>

          <Figure
            kicker="Adjustment curve"
            title={segmentLabel}
            controls={
              <div className="rubberband-controls">
                <label className="field compact-field">
                  <span>Progress</span>
                  <select
                    value={rubberbandBasis}
                    onChange={(event) =>
                      setRubberbandBasis(event.target.value as RubberbandBasis)
                    }
                  >
                    <option value="actual_clock">Actual clock</option>
                    <option value="possession_progress">Possession count</option>
                  </select>
                </label>
                <label className="field compact-field">
                  <span>Segment</span>
                  <select
                    value={timeBucket}
                    onChange={(event) => setTimeBucket(Number(event.target.value))}
                  >
                    {comparisonCoefficients.map((row) => (
                      <option key={row.time_bucket} value={row.time_bucket}>
                        Segment {row.time_bucket + 1}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            }
            note="Positive margin means the offense is leading. Negative adjustment means fewer points than its cross-fitted lineup expectation."
            table={
              <table className="mini">
                <thead>
                  <tr><th>Margin</th><th>Adjustment / 100</th></tr>
                </thead>
                <tbody>
                  {curve.map((row) => (
                    <tr key={row.x}><td>{row.x > 0 ? `+${row.x}` : row.x}</td><td>{fmtRating(row.y)}</td></tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <MultiLine
              series={[{
                label: rubberbandBasis === "actual_clock" ? "Actual clock" : "Possession progress",
                color: "var(--series-1)",
                points: curve,
              }]}
              xFormat={(value) => (value > 0 ? `+${value}` : String(value))}
              yFormat={fmtRating}
              xTitle="Offense margin before possession"
              zeroBaseline
            />
          </Figure>

          <section>
            <div className="section-head">
              <div>
                <p className="kicker">2024 and 2025 fit</p>
                <h2>Margin slope by {rubberbandBasis === "actual_clock" ? "game time" : "possession count"}</h2>
              </div>
            </div>
            <div className="table-wrap">
              <table className="data">
                <caption className="visually-hidden">Rubber-band slope estimates and game-cluster intervals</caption>
                <thead>
                  <tr>
                    <th className="left" scope="col"><button type="button" disabled>Segment</button></th>
                    <th scope="col"><button type="button" disabled>Slope / margin point</button></th>
                    <th scope="col"><button type="button" disabled>95% interval</button></th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonCoefficients.map((row) => (
                    <tr key={row.time_bucket} aria-selected={row.time_bucket === timeBucket} onClick={() => setTimeBucket(row.time_bucket)}>
                      <td className="left name">
                        {row.basis === "actual_clock"
                          ? `${row.minutes_elapsed_start?.toFixed(0)} to ${row.minutes_elapsed_end?.toFixed(0)} min`
                          : row.time_bucket === 7
                            ? "175 plus completed"
                            : `${row.time_bucket * 25} to ${row.time_bucket * 25 + 24} completed`}
                      </td>
                      <td className="headline">{fmtRating(row.slope_points_per_100_per_margin_point)}</td>
                      <td>{fmtRating(row.lower_95)} to {fmtRating(row.upper_95)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <div>
                <p className="kicker">Why it is not promoted</p>
                <h2>Observed score-path addback fails</h2>
              </div>
            </div>
            <table className="mini rubberband-summary">
              <tbody>
                <tr>
                  <th>Normal RMSE</th>
                  <td>{neutralNormal?.margin_rmse.toFixed(3)}</td>
                </tr>
                <tr>
                  <th>Clock addback RMSE</th>
                  <td>{conditionalClock?.margin_rmse.toFixed(3)}</td>
                </tr>
                <tr>
                  <th>Possession addback RMSE</th>
                  <td>{conditionalPossession?.margin_rmse.toFixed(3)}</td>
                </tr>
              </tbody>
            </table>
            <p className="note">
              Score margin is partly created by player performance. Adding its fitted response across the observed game path conditions on that endogenous history and destroys final-margin calibration.
            </p>
          </section>

          <section>
            <div className="section-head">
              <div>
                <p className="kicker">Three-year ratings</p>
                <h2>{rubberbandModel === "normal" ? "Normal" : rubberbandModel === "clock" ? "Clock-adjusted" : "Possession-adjusted"} RAPM</h2>
              </div>
              <label className="field compact-field">
                <span>Model</span>
                <select
                  value={rubberbandModel}
                  onChange={(event) =>
                    setRubberbandModel(event.target.value as RubberbandModel)
                  }
                >
                  <option value="normal">Normal</option>
                  <option value="clock">Clock-adjusted</option>
                  <option value="possession">Possession-adjusted</option>
                </select>
              </label>
            </div>
            <div className="table-wrap">
              <table className="data">
                <caption className="visually-hidden">Rubber-band adjusted RAPM leaderboard</caption>
                <thead>
                  <tr>
                    <th className="left" scope="col"><button type="button" disabled>Player</button></th>
                    <th scope="col"><button type="button" disabled>Offense</button></th>
                    <th scope="col"><button type="button" disabled>Defense</button></th>
                    <th scope="col"><button type="button" disabled>Net</button></th>
                    <th scope="col"><button type="button" disabled>Change</button></th>
                  </tr>
                </thead>
                <tbody>
                  {rubberbandRatings.map((row) => {
                    const offense = rubberbandModel === "normal" ? row.normal_offense : rubberbandModel === "clock" ? row.clock_offense : row.possession_offense;
                    const defense = rubberbandModel === "normal" ? row.normal_defense : rubberbandModel === "clock" ? row.clock_defense : row.possession_defense;
                    const net = rubberbandModel === "normal" ? row.normal_net : rubberbandModel === "clock" ? row.clock_net : row.possession_net;
                    const change = rubberbandModel === "normal" ? 0 : rubberbandModel === "clock" ? row.clock_net_change : row.possession_net_change;
                    return (
                      <tr key={row.player_id}>
                        <td className="left name">{row.player_name}</td>
                        <td>{fmtRating(offense)}</td>
                        <td>{fmtRating(defense)}</td>
                        <td className="headline">{fmtRating(net)}</td>
                        <td>{fmtRating(change)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}

      {project === "age" && (
        <div className="grid">
          <Figure
            kicker="2014 to 2026 · reference age 27"
            title="Estimated lineup-age effect"
            note="Categorical age controls, not a smoothed biological aging curve. Each value compares one player at that age with one age-27 player."
            table={
              <table className="mini">
                <thead>
                  <tr><th>Age</th><th>Offense</th><th>Defense</th><th>Net</th></tr>
                </thead>
                <tbody>
                  {lab.age.curve.map((row) => (
                    <tr key={row.age}>
                      <td>{row.age}</td>
                      <td>{fmtRating(row.offense)}</td>
                      <td>{fmtRating(row.defense)}</td>
                      <td>{fmtRating(row.net)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <MultiLine
              series={[
                { label: "Offense", color: "var(--series-1)", points: lab.age.curve.map((row) => ({ x: row.age, y: row.offense })) },
                { label: "Defense", color: "var(--series-2)", points: lab.age.curve.map((row) => ({ x: row.age, y: row.defense })) },
                { label: "Net", color: "var(--series-3)", points: lab.age.curve.map((row) => ({ x: row.age, y: row.net })) },
              ]}
              xTitle="Age"
              xFormat={(value) => String(value)}
              yFormat={fmtRating}
              markEvery={2}
            />
          </Figure>

          <section className="card">
            <div className="card-head">
              <div>
                <p className="kicker">Held-out check · reused 2026</p>
                <h2>Age context helps; age-27 ratings do not</h2>
              </div>
            </div>
            <table className="mini rubberband-summary">
              <thead>
                <tr><th>Prediction</th><th>RMSE</th><th>Correlation</th></tr>
              </thead>
              <tbody>
                {lab.age.evaluation.map((row) => (
                  <tr key={row.variant}>
                    <th>{row.variant === "normal" ? "Normal" : row.variant === "same_age_27" ? "Same age 27" : "Actual lineup ages"}</th>
                    <td>{row.margin_rmse.toFixed(3)}</td>
                    <td>{row.margin_correlation.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="note">
              Same-age RMSE change {fmtRating(lab.age.bootstrap_vs_normal.same_age_27.observed_rmse_delta)}. Actual-age change {fmtRating(lab.age.bootstrap_vs_normal.age_conditional.observed_rmse_delta)}; paired 95% interval {fmtRating(lab.age.bootstrap_vs_normal.age_conditional.lower_95)} to {fmtRating(lab.age.bootstrap_vs_normal.age_conditional.upper_95)}.
            </p>
          </section>

          <section className="card">
            <div className="card-head">
              <div>
                <p className="kicker">Interpretation</p>
                <h2>What changed</h2>
              </div>
            </div>
            <table className="mini rubberband-summary">
              <tbody>
                <tr><th>Age-slot coverage</th><td>{(100 * lab.age.quality.age_slot_coverage).toFixed(2)}%</td></tr>
                <tr><th>Rating correlation</th><td>{lab.age.rating_effect.net_correlation_with_normal.toFixed(3)}</td></tr>
                <tr><th>Mean absolute change</th><td>{lab.age.rating_effect.mean_absolute_net_change.toFixed(3)}</td></tr>
                <tr><th>Qualified leaderboard</th><td>{fmtInt(lab.age.quality.qualified_players)} players</td></tr>
              </tbody>
            </table>
            <button type="button" className="text-button" onClick={() => { chooseLeaderboard("same-age-ratings"); setProject("tests"); }}>
              Open age-27 leaderboard
            </button>
          </section>
        </div>
      )}

      {project === "matchups" && (
        <MatchupsView lab={matchupLab} />
      )}
    </>
  );
}
