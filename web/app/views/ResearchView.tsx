"use client";

import { Catalog } from "../lib/data";

const LABELS: Record<string, string> = {
  prior: "PULSE prior",
  pulse: "PULSE",
  rapm: "RAPM",
  box15_9y_normal_aio: "PULSE",
  zero_prior_rapm: "RAPM",
  rich_spm_9y_normal_aio: "Rich SPM + lineup update",
};

const fmt = (value: number | undefined, digits = 3) =>
  typeof value === "number" ? value.toFixed(digits) : "—";

export function ResearchView({ catalog }: { catalog: Catalog }) {
  const pulse = catalog.methods.pulse;
  const benchmark = catalog.validation.metric_comparison;
  const common = (benchmark?.summary ?? [])
    .filter((row) => row.scope === "strict_common_2017_2020")
    .sort((a, b) => a.mean_mse - b.mean_mse);
  const maximal = (benchmark?.summary ?? [])
    .filter((row) => row.scope === "all_available_2017_2024")
    .sort((a, b) => a.mean_mse - b.mean_mse);
  const correlations = (benchmark?.correlations ?? [])
    .filter((row) => row.left_candidate === "Box15 + RAPM" || row.right_candidate === "Box15 + RAPM")
    .sort((a, b) => b.pearson - a.pearson);

  return (
    <section aria-labelledby="about-pulse-heading">
      <header className="page-head">
        <div><p className="kicker">Method</p><h1 id="about-pulse-heading">About PULSE</h1></div>
      </header>

      <div className="grid">
        <section className="card prose-grid">
          <div>
            <p className="kicker">Definition</p>
            <h2>PULSE prior + lineup update = PULSE</h2>
            <p className="note">
              The prior estimates longer-run impact from one season of box data.
              The lineup update adds what that season’s possession lineups say after accounting
              for teammates and opponents.
            </p>
          </div>
          <div className="tag-grid">
            <span>Offense and defense are fit together</span>
            <span>Net equals offense plus defense</span>
            <span>Ratings describe the selected season</span>
            <span>Validation uses only earlier training seasons</span>
          </div>
        </section>

        <section className="card prose-grid">
          <div>
            <p className="kicker">Training</p><h2>What the model learns</h2>
            <p className="note">{pulse?.prior}</p>
            <p className="note">{pulse?.lineup_update}</p>
          </div>
          <div>
            <p className="kicker">Validation</p>
            <p className="note">{pulse?.validation}</p>
            <p className="note">
              Lower MSE and RMSE are better. Correlation measures ordering. A calibration slope
              near 1 means the predicted spread has the right amplitude.
            </p>
          </div>
        </section>

        {(pulse?.comparison.length ?? 0) > 0 && (
          <section className="card">
            <div className="card-head"><div><p className="kicker">Next-season games</p><h2>Frozen internal comparison</h2></div></div>
            <div className="table-wrap"><table className="data">
              <thead><tr><th className="left">Model</th><th>Folds</th><th>MSE</th><th>RMSE</th><th>r</th><th>Calibration</th></tr></thead>
              <tbody>{pulse?.comparison.map((row) => (
                <tr key={row.candidate}>
                  <th className="left">{LABELS[row.candidate] ?? row.candidate}</th>
                  <td>{row.folds}</td><td>{fmt(row.equal_season_mse)}</td>
                  <td>{fmt(row.equal_season_rmse)}</td><td>{fmt(row.mean_correlation)}</td>
                  <td>{fmt(row.mean_calibration_slope)}</td>
                </tr>
              ))}</tbody>
            </table></div>
          </section>
        )}

        <section className="card">
          <div className="card-head"><div><p className="kicker">PULSE prior</p><h2>The 15 box inputs</h2></div></div>
          <div className="tag-grid">
            {(pulse?.box15_inputs ?? []).map((feature) => <span key={feature}>{feature} per 100</span>)}
          </div>
          <p className="note">
            Possessions weight noisy training labels. Possessions are not a player-value input.
            Rich tracking, playtype and matchup features remain a research comparison because they
            did not improve the downstream lineup-updated model.
          </p>
        </section>

        {common.length > 0 && <MetricTable title="Strict common coverage" rows={common} />}
        {maximal.length > 0 && <MetricTable title="Each metric’s available coverage" rows={maximal} />}

        {correlations.length > 0 && (
          <section className="card">
            <div className="card-head"><div><p className="kicker">Player seasons</p><h2>Agreement with the earlier Box15 model</h2></div></div>
            <div className="table-wrap"><table className="data">
              <thead><tr><th className="left">Metric</th><th>Matched</th><th>Pearson</th><th>Spearman</th></tr></thead>
              <tbody>{correlations.map((row) => {
                const other = row.left_candidate === "Box15 + RAPM" ? row.right_candidate : row.left_candidate;
                return <tr key={other}><th className="left">{other}</th><td>{row.matched_player_seasons}</td><td>{fmt(row.pearson)}</td><td>{fmt(row.spearman)}</td></tr>;
              })}</tbody>
            </table></div>
          </section>
        )}

        <section className="card prose-grid">
          <div><p className="kicker">Limits</p><h2>What PULSE does not claim</h2></div>
          <div className="tag-grid">
            <span>It is not a preseason forecast</span>
            <span>It is not causal credit for every play</span>
            <span>External metrics have different coverage and information timing</span>
            <span>Historical source quality varies by era</span>
            <span>Event-derived factors depend on the events available in each era</span>
          </div>
          <p className="note">
            The canonical lineup ledger reconciles more than 99.9% of game scores. It excludes every
            identified technical free throw from player impact and reconciles those points separately.
          </p>
          {benchmark?.note && <p className="note">{benchmark.note}</p>}
        </section>
      </div>
    </section>
  );
}

function MetricTable({
  title,
  rows,
}: {
  title: string;
  rows: NonNullable<Catalog["validation"]["metric_comparison"]>["summary"];
}) {
  return (
    <section className="card">
      <div className="card-head"><div><p className="kicker">External benchmark</p><h2>{title}</h2></div></div>
      <div className="table-wrap"><table className="data">
        <thead><tr><th className="left">Metric</th><th>Seasons</th><th>RMSE</th><th>r</th><th>Calibration</th></tr></thead>
        <tbody>{rows.map((row) => (
          <tr key={`${row.scope}-${row.candidate}`}>
            <th className="left">{row.candidate}</th>
            <td>{row.folds}</td><td>{fmt(row.aggregate_rmse)}</td><td>{fmt(row.mean_correlation)}</td><td>{fmt(row.mean_calibration_slope)}</td>
          </tr>
        ))}</tbody>
      </table></div>
    </section>
  );
}
