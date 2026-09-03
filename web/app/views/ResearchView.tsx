"use client";

import { useEffect, useState } from "react";
import { MultiLine } from "../charts/lines";
import { Figure } from "../charts/frame";
import { Catalog } from "../lib/data";
import { WP_RELEASE_ID } from "../lib/rapmRelease";

const LABELS: Record<string, string> = {
  prior: "PULSE prior",
  pulse: "PULSE",
  rapm: "Lineup-only RAPM",
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

const EXTERNAL_RUN = "pulse_external_common_v1_c500545ce4";
type ExternalBenchmark = {
  run_id: string;
  panels: { scope: string; outcome_start: number; outcome_end: number; games: number;
    rows: BenchmarkRow[]; matched_exposure_min: number; matched_exposure_max: number }[];
  rich_prior_test: {
    run_id: string;
    outcome_start: number;
    outcome_end: number;
    games: number;
    rows: BenchmarkRow[];
    pulse_minus_rich_mse: number;
    lower_95: number;
    upper_95: number;
  };
  source_rating_years: { model: string; start: number; end: number }[];
};

type ResearchCurves = {
  age: { age: number; offense: number; defense: number; net: number }[];
  score_state: { margin: number; effect_points_per_100_vs_tie: number }[];
  wp_rapm_vs_pulse: {
    run_id: string;
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
    published_lambda: number;
    warning: string;
  };
};

export function ResearchView({ catalog }: { catalog: Catalog }) {
  const [curves, setCurves] = useState<ResearchCurves | null>(null);
  const [external, setExternal] = useState<ExternalBenchmark | null>(null);
  const [externalFailed, setExternalFailed] = useState(false);
  const pulse = catalog.methods.pulse;

  useEffect(() => {
    fetch(`/data/external-benchmark.json?v=${EXTERNAL_RUN}`)
      .then((response) => {
        if (!response.ok) throw new Error("External benchmark unavailable");
        return response.json() as Promise<ExternalBenchmark>;
      })
      .then((data) => {
        if (data.run_id !== EXTERNAL_RUN) throw new Error("Stale external benchmark");
        setExternal(data);
      })
      .catch(() => setExternalFailed(true));
    fetch(`/data/rapm/research-curves.json?v=${WP_RELEASE_ID}`)
      .then((response) => {
        if (!response.ok) throw new Error("Research curves unavailable");
        return response.json() as Promise<ResearchCurves>;
      })
      .then((data) => {
        if (data.wp_rapm_vs_pulse?.run_id !== WP_RELEASE_ID) throw new Error("Stale WP research release");
        setCurves(data);
      })
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
          Each PULSE historical fold trains only on earlier rating seasons. It creates
          a prior for season t, updates that prior with season-t possessions,
          and predicts season t+1 games. Every head-to-head comparison uses the
          same games and player coverage. Lower MSE and RMSE are better.
          Correlation measures association with game margins. A calibration slope near 1 means the
          predicted margin spread has the right size.
        </p>
        <p>
          This is the primary site test. It uses every player carried by the saved
          PULSE folds. It does not restrict PULSE to the coverage of external models.
          The target excludes technical free throws.
        </p>
        {(pulse?.comparison.length ?? 0) > 0 && (
          <BenchmarkTable title="Full-coverage PULSE test, 2015–2026 games" rows={pulse?.comparison.map((row) => ({
            candidate: LABELS[row.candidate] ?? row.candidate,
            folds: row.folds,
            aggregate_rmse: row.equal_season_rmse,
            mean_correlation: row.mean_correlation,
            mean_calibration_slope: row.mean_calibration_slope,
          })) ?? []} />
        )}
        {external?.rich_prior_test && <div>
          <BenchmarkTable
            title={`Prior swap on shared ${external.rich_prior_test.outcome_start}–${external.rich_prior_test.outcome_end} games`}
            rows={external.rich_prior_test.rows}
          />
          <p>
            Both priors train on rating seasons earlier than season t. Each then receives
            the same season-t lineup update and predicts season t+1 on the same
            {external.rich_prior_test.games.toLocaleString()} games. PULSE minus Rich SPM
            MSE is {fmt(external.rich_prior_test.pulse_minus_rich_mse)}, with a 95% whole-game
            interval of [{fmt(external.rich_prior_test.lower_95)}, {fmt(external.rich_prior_test.upper_95)}].
            Rich SPM is better alone, but the Box15 prior is better after the lineup update.
          </p>
          <p>This is reused historical evidence, not an untouched confirmation.</p>
        </div>}
      </section>

      <section aria-labelledby="external-benchmark-heading">
        <h2 id="external-benchmark-heading">Against other models</h2>
        <p>
          Ratings from 2015–2025 predict the following season’s games from 2016–2026.
          Every model uses the same games and matched player coverage. The target is
          the official final point margin, including technical free throws.
          Lower RMSE is better. No calibration correction is applied.
        </p>
        {external?.panels.filter((panel) => panel.scope === "main").map((panel) => <div key={panel.scope}>
          <BenchmarkTable title={`${panel.outcome_start}–${panel.outcome_end} games`} rows={panel.rows} />
          <p>{panel.games.toLocaleString()} games. Matched ratings cover {(100 * panel.matched_exposure_min).toFixed(1)}–{(100 * panel.matched_exposure_max).toFixed(1)}% of lineup exposure across seasons.
            Unmatched players receive zero for every model. Ratings share source-season centering and the same RAPM home adjustment.</p>
        </div>)}
        {!external && <p role="status">{externalFailed ? "Benchmark results are unavailable. Reload to try again." : "Loading benchmark results…"}</p>}
        <p>
          PULSE uses past-only prior fits, followed by the rated season’s lineup update.
          PIPM and RAPTOR are CourtSignal reconstructions used as-is. RAPTOR’s mapping
          was fitted across 2014–2022, so its earlier results are not an out-of-time test.
          External files do not establish their original training cutoffs.
        </p>
        <p>
          These historical seasons have informed model development. Results are diagnostic,
          not untouched confirmation. All predictions use observed next-season lineups.
        </p>
        {external && <details>
          <summary>Source coverage and MAMBA</summary>
          <p>Available rating years, using each season’s ending year. A 2026 rating would predict 2027, which is not evaluated here.</p>
          <div className="table-wrap"><table className="data compact-data benchmark-data">
            <thead><tr><th className="left">Metric</th><th>Rating years</th></tr></thead>
            <tbody>{external.source_rating_years.map((row) => <tr key={row.model}>
              <th className="left">{row.model}</th><td>{row.start}–{row.end}</td>
            </tr>)}</tbody>
          </table></div>
          <p>MAMBA stops at the 2023–24 rating season. Its comparison therefore ends with 2024–25 games.</p>
          {external.panels.filter((panel) => panel.scope === "with_mamba").map((panel) => <BenchmarkTable
            key={panel.scope} title={`${panel.outcome_start}–${panel.outcome_end} games, including MAMBA`} rows={panel.rows} />)}
        </details>}
      </section>

      <section>
        <h2>Win-probability research</h2>
        <p>
          WP-RAPM assigns lineup-adjusted credit for changes in win probability.
          The log-odds version uses changes in clipped home-win log odds. Neither
          scale is points per 100 possessions.
        </p>
        {curves?.wp_rapm_vs_pulse && <WpRapmValidation comparison={curves.wp_rapm_vs_pulse} />}
        <p>
          Historical possession ordering and winner labels have been corrected.
          Earlier WP penalty comparisons and box-prior results are withdrawn.
          Published WP boards are descriptive research ratings,
          not forecasts or causal credit.
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
  const seasons = comparison.outcome_seasons;
  return <div className="article-table">
    <h3>Next-season point-margin validation</h3>
    <p>
      All models use an affine map fitted only on earlier outcome seasons and
      are scored on the same {comparison.games.toLocaleString()} games from {Math.min(...seasons)}–{Math.max(...seasons)}.
    </p>
    <p>Target: {comparison.target}. This rescoring differs from the PULSE validation target above.</p>
    <div className="table-wrap"><table className="data compact-data">
      <thead><tr><th className="left">Metric</th><th>Seasons</th><th>Games</th><th>RMSE</th><th>r</th></tr></thead>
      <tbody>{comparison.summary.map((row) => <tr key={row.model}>
        <th className="left">{row.model}</th><td>{row.folds}</td><td>{row.games.toLocaleString()}</td>
        <td>{fmt(row.equal_season_rmse)}</td><td>{fmt(row.mean_correlation)}</td>
      </tr>)}</tbody>
    </table></div>
    <p>{comparison.warning}</p>
  </div>;
}

function BenchmarkTable({ title, rows }: { title?: string; rows: BenchmarkRow[] }) {
  return <div className="article-table">
    {title && <h3>{title}</h3>}
    <div className="table-wrap"><table className="data compact-data benchmark-data">
      <thead><tr><th className="left">Metric</th><th>RMSE</th><th>r</th><th><abbr title="Calibration slope">Cal.</abbr></th></tr></thead>
      <tbody>{rows.map((row) => <tr key={row.candidate}>
        <th className="left">{row.candidate}</th>
        <td>{fmt(row.aggregate_rmse)}</td><td>{fmt(row.mean_correlation)}</td>
        <td>{fmt(row.mean_calibration_slope)}</td>
      </tr>)}</tbody>
    </table></div>
  </div>;
}
