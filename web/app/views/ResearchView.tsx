"use client";

import { useEffect, useState } from "react";
import { MultiLine } from "../charts/lines";
import { Figure } from "../charts/frame";
import { Catalog } from "../lib/data";

const LABELS: Record<string, string> = {
  prior: "PULSE prior",
  pulse: "PULSE",
  rapm: "Lineup-only RAPM",
  box15_9y_normal_aio: "PULSE",
  zero_prior_rapm: "Lineup-only RAPM",
  rich_spm_9y_normal_aio: "Rich SPM + lineup update",
};

const fmt = (value: number | undefined, digits = 3) =>
  typeof value === "number" ? value.toFixed(digits) : "—";

type BenchmarkRow = {
  candidate: string;
  folds: number;
  aggregate_rmse: number;
  mean_correlation: number;
  mean_calibration_slope: number;
};

type ResearchCurves = {
  age: { age: number; offense: number; defense: number; net: number }[];
  score_state: { margin: number; effect_points_per_100_vs_tie: number }[];
  wp_rapm_vs_pulse: {
    comparison: string;
    target: string;
    outcome_seasons: number[];
    games: number;
    summary: {
      model: string;
      folds: number;
      games: number;
      mean_correlation: number;
      equal_season_rmse: number;
    }[];
    paired_comparisons: {
      draws: number;
      left: string;
      right: string;
      mean_mse_delta: number;
      lower_95: number;
      upper_95: number;
      probability_left_better: number;
    }[];
    warning: string;
  };
};

export function ResearchView({ catalog }: { catalog: Catalog }) {
  const [curves, setCurves] = useState<ResearchCurves | null>(null);
  const pulse = catalog.methods.pulse;

  useEffect(() => {
    fetch("/data/rapm/research-curves.json")
      .then((response) => {
        if (!response.ok) throw new Error("Research curves unavailable");
        return response.json() as Promise<ResearchCurves>;
      })
      .then(setCurves)
      .catch(() => setCurves(null));
  }, []);

  return (
    <article className="about-article" aria-labelledby="about-pulse-heading">
      <header>
        <p className="kicker">Method</p>
        <h1 id="about-pulse-heading">About PULSE</h1>
        <p className="article-lede">
          PULSE estimates a player’s impact per 100 possessions. It combines a
          statistical prior with one season of lineup evidence.
        </p>
      </header>

      <section>
        <h2>How PULSE works</h2>
        <p>
          The prior learns longer-run impact from 15 box-score rates. The lineup
          update measures how the player’s teams performed in each possession
          after accounting for every teammate and opponent. The two parts add
          exactly: PULSE prior + lineup update = PULSE.
        </p>
        <p>
          Offense and defense are estimated separately. Net equals offense plus
          defense. The model uses possessions to weight noisy training labels.
          Possession count does not enter as player value.
        </p>
      </section>

      <section>
        <h2>The box prior</h2>
        <p>{pulse?.prior}</p>
        <ul className="article-columns">
          {(pulse?.box15_inputs ?? []).map((feature) => <li key={feature}>{feature} per 100</li>)}
        </ul>
        <p>
          Rich SPM uses tracking, playtype, matchup, and stabilized shooting
          inputs. It reconstructs stable RAPM better on its own. It did not
          improve the final lineup-updated rating, so PULSE keeps the simpler
          box prior. Rich SPM remains available as a comparison.
        </p>
      </section>

      <section>
        <h2>Benchmarking process</h2>
        <p>
          Each historical fold trains only on earlier rating seasons. It creates
          a prior for season t, updates that prior with season-t possessions,
          and predicts season t+1 games. Every head-to-head comparison uses the
          same games and player coverage. Lower MSE and RMSE are better.
          Correlation measures ranking. A calibration slope near 1 means the
          predicted margin spread has the right size.
        </p>
        {(pulse?.comparison.length ?? 0) > 0 && (
          <BenchmarkTable title="PULSE validation" rows={pulse?.comparison.map((row) => ({
            candidate: LABELS[row.candidate] ?? row.candidate,
            folds: row.folds,
            aggregate_rmse: row.equal_season_rmse,
            mean_correlation: row.mean_correlation,
            mean_calibration_slope: row.mean_calibration_slope,
          })) ?? []} />
        )}
      </section>

      <section>
        <h2>Win-probability research</h2>
        <p>
          On the native win-probability target, plain WP-RAPM scored 0.981
          RMSE. Adding the PULSE box prior raised RMSE to 1.130; using Rich SPM
          as the prior raised it to 1.119. Those scores are not comparable to
          point-margin RMSE.
        </p>
        {curves?.wp_rapm_vs_pulse && <WpRapmValidation comparison={curves.wp_rapm_vs_pulse} />}
        <p>
          The current target is conserved change in win probability. Change in
          log odds would be a different, unbounded estimand and needs its own
          clipped, out-of-sample test before it replaces that target.
        </p>
      </section>

      {curves && <section>
        <h2>Context curves</h2>
        <p>
          These curves describe the rejected age and score-state adjustments.
          They remain research context and do not change the published ratings.
        </p>
        <div className="grid two">
          <Figure title="Age curve" note="Estimated change relative to age 27.">
            <MultiLine
              series={[
                { label: "Net", color: "var(--series-1)", points: curves.age.map((row) => ({ x: row.age, y: row.net })) },
                { label: "Offense", color: "var(--series-2)", points: curves.age.map((row) => ({ x: row.age, y: row.offense })) },
                { label: "Defense", color: "var(--series-3)", points: curves.age.map((row) => ({ x: row.age, y: row.defense })) },
              ]}
              xTitle="Age"
            />
          </Figure>
          <Figure title="Score-state curve" note="Adjustment relative to a tied game.">
            <MultiLine
              series={[{
                label: "Score state",
                color: "var(--series-1)",
                points: curves.score_state.map((row) => ({ x: row.margin, y: row.effect_points_per_100_vs_tie })),
              }]}
              xTitle="Offense margin before possession"
            />
          </Figure>
        </div>
      </section>}

      <section>
        <h2>Limits</h2>
        <p>
          PULSE is a retrospective season rating. It is not a preseason
          forecast or causal credit for every play. Historical event coverage
          changes by era. External metrics also use different inputs and
          information dates.
        </p>
      </section>
    </article>
  );
}

function WpRapmValidation({ comparison }: {
  comparison: ResearchCurves["wp_rapm_vs_pulse"];
}) {
  const wpVsRapm = comparison.paired_comparisons.find((row) => row.left === "WP-RAPM" && row.right === "RAPM");
  const seasons = comparison.outcome_seasons;
  return <div className="article-table">
    <h3>Next-season point-margin validation</h3>
    <p>
      All three models use an affine map fitted only on earlier outcome seasons and
      are scored on the same {comparison.games.toLocaleString()} games from {Math.min(...seasons)}–{Math.max(...seasons)}.
    </p>
    <div className="table-wrap"><table className="data compact-data">
      <thead><tr><th className="left">Metric</th><th>Seasons</th><th>Games</th><th>RMSE</th><th>r</th></tr></thead>
      <tbody>{comparison.summary.map((row) => <tr key={row.model}>
        <th className="left">{row.model}</th><td>{row.folds}</td><td>{row.games.toLocaleString()}</td>
        <td>{fmt(row.equal_season_rmse)}</td><td>{fmt(row.mean_correlation)}</td>
      </tr>)}</tbody>
    </table></div>
    {wpVsRapm && <p>
      WP-RAPM minus ordinary RAPM MSE was {fmt(wpVsRapm.mean_mse_delta)} with a
      95% interval of [{fmt(wpVsRapm.lower_95)}, {fmt(wpVsRapm.upper_95)}]. WP-RAPM
      won {Math.round(wpVsRapm.probability_left_better * wpVsRapm.draws)} of {wpVsRapm.draws.toLocaleString()} paired bootstrap draws.
    </p>}
    <p>
      This comparison uses one-season zero-prior WP-RAPM and observed next-season
      lineups. It is a reused historical diagnostic, not a deployable forecast.
    </p>
  </div>;
}

function BenchmarkTable({ title, rows }: { title?: string; rows: BenchmarkRow[] }) {
  return <div className="article-table">
    {title && <h3>{title}</h3>}
    <div className="table-wrap"><table className="data compact-data">
      <thead><tr><th className="left">Metric</th><th>Seasons</th><th>RMSE</th><th>r</th><th>Calibration</th></tr></thead>
      <tbody>{rows.map((row) => <tr key={row.candidate}>
        <th className="left">{row.candidate}</th><td>{row.folds}</td>
        <td>{fmt(row.aggregate_rmse)}</td><td>{fmt(row.mean_correlation)}</td>
        <td>{fmt(row.mean_calibration_slope)}</td>
      </tr>)}</tbody>
    </table></div>
  </div>;
}
