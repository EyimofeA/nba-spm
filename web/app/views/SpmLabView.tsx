"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { Component, SpmLabPayload } from "../lib/data";
import { fmtRating } from "../lib/viz";

type Metric = "spm" | "aio";
type View = "box-aio" | "comparison" | "ratings" | "features";
type Correlation = "pearson" | "spearman";
type SortKey = "selected_net" | "selected_offense" | "selected_defense" | "delta_net";
type BoxSortKey = "prior_offense" | "prior_defense" | "prior_net" | "posterior_offense" | "posterior_defense" | "posterior_net" | "change_net";
type FeatureSide = "all" | "offense" | "defense";

const pct = (value: number) => `${(100 * value).toFixed(1)}%`;

export function SpmLabView({ lab }: { lab: SpmLabPayload | null }) {
  const [view, setView] = useState<View>("box-aio");
  const [season, setSeason] = useState(2026);
  const [metric, setMetric] = useState<Metric>("aio");
  const [sort, setSort] = useState<SortKey>("selected_net");
  const [ascending, setAscending] = useState(false);
  const [component, setComponent] = useState<Component>("net");
  const [correlation, setCorrelation] = useState<Correlation>("pearson");
  const [featureSide, setFeatureSide] = useState<FeatureSide>("all");
  const [featureSearch, setFeatureSearch] = useState("");
  const [boxSort, setBoxSort] = useState<BoxSortKey>("posterior_net");
  const [boxAscending, setBoxAscending] = useState(false);

  const rows = useMemo(
    () =>
      (lab?.ratings ?? [])
        .filter((row) => row.Season === season && row.metric === metric)
        .sort((a, b) => (a[sort] - b[sort]) * (ascending ? 1 : -1)),
    [ascending, lab, metric, season, sort],
  );
  const includedMetrics = useMemo(
    () => (lab?.comparison?.definitions ?? []).filter((row) => row.included),
    [lab],
  );
  const correlationRows = useMemo(
    () => (lab?.comparison?.pairwise_correlations ?? []).filter((row) => row.component === component),
    [component, lab],
  );
  const correlationLookup = useMemo(
    () => new Map(correlationRows.map((row) => [`${row.left_metric}:${row.right_metric}`, row])),
    [correlationRows],
  );
  const featureRows = useMemo(() => {
    const query = featureSearch.trim().toLowerCase();
    return lab?.weighting.feature_catalog.filter((row) => {
      const matchesSide = featureSide === "all" || row.side === "both" || row.side === featureSide;
      const matchesSearch = !query || row.feature.toLowerCase().includes(query) || row.description.toLowerCase().includes(query);
      return matchesSide && matchesSearch;
    }) ?? [];
  }, [featureSearch, featureSide, lab]);
  const boxRows = useMemo(
    () =>
      (lab?.box15.leaderboard ?? [])
        .filter((row) => row.Season === season)
        .sort((a, b) => (a[boxSort] - b[boxSort]) * (boxAscending ? 1 : -1)),
    [boxAscending, boxSort, lab, season],
  );
  const boxCorrelations = useMemo(
    () => (lab?.box15.correlations ?? []).filter((row) => row.component === component),
    [component, lab],
  );

  const sortBy = (key: SortKey) => {
    if (sort === key) setAscending((value) => !value);
    else {
      setSort(key);
      setAscending(false);
    }
  };
  const sortBoxBy = (key: BoxSortKey) => {
    if (boxSort === key) setBoxAscending((value) => !value);
    else {
      setBoxSort(key);
      setBoxAscending(false);
    }
  };
  if (!lab) return <p className="empty">SPM research data unavailable.</p>;
  if (!lab.comparison) return <p className="empty">SPM comparison data unavailable.</p>;

  return (
    <div className="grid">
      <div className="page-head">
        <div><p className="kicker">Local research</p><h1>SPM Lab</h1></div>
        <div className="segmented" aria-label="SPM Lab view">
          {(["box-aio", "comparison", "ratings", "features"] as View[]).map((value) => (
            <button key={value} type="button" aria-pressed={view === value} onClick={() => setView(value)}>{value === "box-aio" ? "Box AIO" : value[0].toUpperCase() + value.slice(1)}</button>
          ))}
        </div>
      </div>

      {view === "box-aio" && <>
        <section className="card">
          <div className="card-head">
            <div><p className="kicker">Box15 prior + possession update</p><h2>Box AIO leaderboard</h2></div>
            <label className="field"><span>Season</span><select value={season} onChange={(event) => setSeason(Number(event.target.value))}>{lab.box15.seasons.toReversed().map((value) => <option key={value}>{value}</option>)}</select></label>
          </div>
          <div className="table-wrap comparison-table">
            <table className="data">
              <thead>
                <tr><th className="left" rowSpan={2}><button type="button" disabled>Player</button></th><th colSpan={3}>Prior</th><th colSpan={3}>AIO posterior</th><th rowSpan={2}><button type="button" onClick={() => sortBoxBy("change_net")}>Change</button></th></tr>
                <tr><th><button type="button" onClick={() => sortBoxBy("prior_offense")}>Off</button></th><th><button type="button" onClick={() => sortBoxBy("prior_defense")}>Def</button></th><th><button type="button" onClick={() => sortBoxBy("prior_net")}>Net</button></th><th><button type="button" onClick={() => sortBoxBy("posterior_offense")}>Off</button></th><th><button type="button" onClick={() => sortBoxBy("posterior_defense")}>Def</button></th><th><button type="button" onClick={() => sortBoxBy("posterior_net")}>Net</button></th></tr>
              </thead>
              <tbody>{boxRows.map((row) => <tr key={`${row.Season}-${row.PLAYER_ID}`}><td className="left name">{row.PLAYER_NAME}</td><td>{fmtRating(row.prior_offense)}</td><td>{fmtRating(row.prior_defense)}</td><td>{fmtRating(row.prior_net)}</td><td>{fmtRating(row.posterior_offense)}</td><td>{fmtRating(row.posterior_defense)}</td><td className="headline">{fmtRating(row.posterior_net)}</td><td>{fmtRating(row.change_net)}</td></tr>)}</tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <div className="card-head">
            <div><p className="kicker">Matched 2021–24 player-seasons</p><h2>Agreement with other metrics</h2></div>
            <div className="filters compact-filters">
              <div className="segmented">{(["net", "offense", "defense"] as Component[]).map((value) => <button key={value} type="button" aria-pressed={component === value} onClick={() => setComponent(value)}>{value}</button>)}</div>
              <div className="segmented">{(["pearson", "spearman"] as Correlation[]).map((value) => <button key={value} type="button" aria-pressed={correlation === value} onClick={() => setCorrelation(value)}>{value === "spearman" ? "Rank" : "Pearson"}</button>)}</div>
            </div>
          </div>
          <div className="table-wrap comparison-table">
            <table className="data">
              <thead><tr><th className="left"><button type="button" disabled>Metric</button></th><th><button type="button" disabled>Prior</button></th><th><button type="button" disabled>AIO posterior</button></th><th><button type="button" disabled>Players</button></th></tr></thead>
              <tbody>{boxCorrelations.map((row) => <tr key={`${row.component}-${row.metric}`}><td className="left name">{row.metric_label}</td><td>{row[`prior_${correlation}`].toFixed(3)}</td><td className="headline">{row[`posterior_${correlation}`].toFixed(3)}</td><td>{Math.min(row.prior_rows, row.posterior_rows).toLocaleString()}</td></tr>)}</tbody>
            </table>
          </div>
        </section>
      </>}

      {view === "comparison" && <>
        <section className="card">
          <div className="card-head">
            <div><p className="kicker">Next-season team test</p><h2>Public metric comparison</h2></div>
            <span className="meta">{lab.comparison.common_seasons[0]}–{lab.comparison.common_seasons.at(-1)}</span>
          </div>
          <p className="note">Year Y ratings are weighted by observed player minutes for each team in Y+1, then compared with Y+1 win percentage. This tests ratings with perfect knowledge of next season&apos;s minutes; it is not a preseason forecast.</p>
          <div className="table-wrap comparison-table">
            <table className="data">
              <thead><tr><th className="left"><button type="button" disabled>Metric</button></th><th><button type="button" disabled>Mean R²</button></th><th><button type="button" disabled>Pearson</button></th><th><button type="button" disabled>Rank corr.</button></th><th><button type="button" disabled>Coverage</button></th></tr></thead>
              <tbody>{lab.comparison.team_win_summary.map((row) => <tr key={row.metric}><td className="left name">{row.metric_label}</td><td className="headline">{row.mean_r_squared.toFixed(3)}</td><td>{row.mean_pearson.toFixed(3)}</td><td>{row.mean_spearman.toFixed(3)}</td><td>{pct(row.minimum_minute_coverage)}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="note">Players below {lab.comparison.minimum_metric_year_minutes} minutes in year Y use a {fmtRating(lab.comparison.replacement_value)} replacement rating. Four seasons is useful evidence, not a promotion verdict.</p>
        </section>

        <section className="card">
          <div className="card-head">
            <div><p className="kicker">Same players · same seasons</p><h2>Metric agreement</h2></div>
            <div className="filters compact-filters">
              <div className="segmented">{(["net", "offense", "defense"] as Component[]).map((value) => <button key={value} type="button" aria-pressed={component === value} onClick={() => setComponent(value)}>{value}</button>)}</div>
              <div className="segmented">{(["pearson", "spearman"] as Correlation[]).map((value) => <button key={value} type="button" aria-pressed={correlation === value} onClick={() => setCorrelation(value)}>{value === "spearman" ? "Rank" : "Pearson"}</button>)}</div>
            </div>
          </div>
          <div className="table-wrap correlation-matrix">
            <table className="data">
              <thead><tr><th className="left"><button type="button" disabled>Metric</button></th>{includedMetrics.map((item) => <th key={item.metric} title={item.metric_label}><button type="button" disabled>{item.metric_label}</button></th>)}</tr></thead>
              <tbody>{includedMetrics.map((left) => <tr key={left.metric}>
                <td className="left name">{left.metric_label}</td>
                {includedMetrics.map((right) => {
                  const row = correlationLookup.get(`${left.metric}:${right.metric}`);
                  const value = row?.[correlation];
                  return <td key={right.metric} className="correlation-cell" style={value == null ? undefined : { "--strength": Math.max(0, value) } as CSSProperties} title={row ? `${row.rows.toLocaleString()} matched player-seasons` : "No common observations"}>{value == null ? "—" : value.toFixed(2)}</td>;
                })}
              </tr>)}</tbody>
            </table>
          </div>
          <p className="note">Pairwise correlations use only player-seasons shared by each pair, after the same 250-minute screen. High agreement is not proof of accuracy.</p>
        </section>

        <section className="card">
          <div className="card-head"><div><p className="kicker">Definitions</p><h2>What each metric does</h2></div></div>
          <div className="metric-definitions">
            {lab.comparison.definitions.map((row) => <article key={row.metric} className="metric-definition">
              <div><h3>{row.metric_label}</h3><span className="meta">{row.included ? row.kind : "Not scored"}</span></div>
              <p>{row.how_it_works}</p><p className="note">{row.interpretation}</p>
            </article>)}
          </div>
        </section>

        <section className="card">
          <div className="card-head"><div><p className="kicker">Season-by-season</p><h2>R² by outcome season</h2></div></div>
          <div className="table-wrap"><table className="data">
            <thead><tr><th className="left"><button type="button" disabled>Metric</button></th>{lab.comparison.common_seasons.map((value) => <th key={value}><button type="button" disabled>{value + 1}</button></th>)}</tr></thead>
            <tbody>{lab.comparison.team_win_summary.map((summary) => <tr key={summary.metric}><td className="left name">{summary.metric_label}</td>{lab.comparison.common_seasons.map((value) => { const fold = lab.comparison.team_win_folds.find((row) => row.metric === summary.metric && row.rating_season === value); return <td key={value}>{fold ? fold.r_squared.toFixed(3) : "—"}</td>; })}</tr>)}</tbody>
          </table></div>
        </section>
      </>}

      {view === "ratings" && <section className="card">
        <div className="filters">
          <label className="field"><span>Season</span><select value={season} onChange={(event) => setSeason(Number(event.target.value))}>{lab.seasons.toReversed().map((value) => <option key={value}>{value}</option>)}</select></label>
          <div className="field"><span>Rating</span><div className="segmented"><button type="button" aria-pressed={metric === "spm"} onClick={() => setMetric("spm")}>SPM</button><button type="button" aria-pressed={metric === "aio"} onClick={() => setMetric("aio")}>AIO</button></div></div>
        </div>
        <div className="table-wrap"><table className="data"><thead><tr><th className="left"><button type="button" disabled>Player</button></th><th><button type="button" onClick={() => sortBy("selected_offense")}>Offense</button></th><th><button type="button" onClick={() => sortBy("selected_defense")}>Defense</button></th><th><button type="button" onClick={() => sortBy("selected_net")}>Net</button></th><th><button type="button" onClick={() => sortBy("delta_net")}>Change</button></th></tr></thead><tbody>{rows.map((row) => <tr key={`${row.metric}-${row.Season}-${row.PLAYER_ID}`}><td className="left name">{row.PLAYER_NAME}</td><td>{fmtRating(row.selected_offense)}</td><td>{fmtRating(row.selected_defense)}</td><td>{fmtRating(row.selected_net)}</td><td>{fmtRating(row.delta_net)}</td></tr>)}</tbody></table></div>
      </section>}

      {view === "features" && <>
        <section className="card">
          <div className="card-head"><div><p className="kicker">13 held-out seasons</p><h2>Does possession weighting help?</h2></div></div>
          <div className="table-wrap"><table className="data"><thead><tr><th className="left">Evaluation</th><th>Component</th><th>Weighted fit</th><th>No weight</th><th>Change</th></tr></thead><tbody>
            {(["sqrt_possessions", "equal_players"] as const).flatMap((evaluation) => (["offense", "defense", "net"] as const).map((side) => {
              const weighted = lab.weighting.summary.find((row) => row.evaluation === evaluation && row.component === side && row.variant === "sqrt_possessions");
              const unweighted = lab.weighting.summary.find((row) => row.evaluation === evaluation && row.component === side && row.variant === "unweighted");
              if (!weighted || !unweighted) return null;
              return <tr key={`${evaluation}-${side}`}><td className="left name">{evaluation === "sqrt_possessions" ? "Possession-weighted RMSE" : "Equal-player RMSE"}</td><td>{side}</td><td>{weighted.mean_rmse.toFixed(3)}</td><td>{unweighted.mean_rmse.toFixed(3)}</td><td>{fmtRating(unweighted.mean_rmse - weighted.mean_rmse)}</td></tr>;
            }))}
          </tbody></table></div>
          <p className="note">Positive change means removing the weight made RMSE worse. Both fits use the same current runtime, features, RAPM labels and leave-one-season-out folds.</p>
        </section>

        <section className="card">
          <div className="card-head"><div><p className="kicker">127 offense · 68 defense</p><h2>Website SPM inputs</h2></div></div>
          <div className="filters">
            <label className="field"><span>Find a feature</span><input value={featureSearch} onChange={(event) => setFeatureSearch(event.target.value)} placeholder="spacing, matchup, turnovers…" /></label>
            <div className="field"><span>Side</span><div className="segmented">{(["all", "offense", "defense"] as FeatureSide[]).map((value) => <button key={value} type="button" aria-pressed={featureSide === value} onClick={() => setFeatureSide(value)}>{value}</button>)}</div></div>
          </div>
          <div className="table-wrap"><table className="data"><thead><tr><th className="left">Feature</th><th>Side</th><th className="left">Meaning</th></tr></thead><tbody>{featureRows.map((row) => <tr key={row.feature}><td className="left name"><code>{row.feature}</code></td><td>{row.side}</td><td className="left">{row.description}</td></tr>)}</tbody></table></div>
        </section>

        <section className="card">
          <div className="card-head"><div><p className="kicker">Forward selection</p><h2>Feature families</h2></div></div>
          <div className="table-wrap"><table className="data"><thead><tr><th className="left"><button type="button" disabled>Family</button></th><th>Side</th><th>Features</th><th>RMSE change</th><th>Changer change</th><th>Decision</th></tr></thead><tbody>{lab.decisions.map((row) => <tr key={row.group}><td className="left name">{row.group.replaceAll("_", " ")}</td><td>{row.side}</td><td>{row.feature_count}</td><td>{fmtRating(row.development_mean_rmse_delta)}</td><td>{fmtRating(row.team_changer_mean_rmse_delta)}</td><td>{row.selected ? "Add" : "Reject"}</td></tr>)}</tbody></table></div>
        </section>
        <section className="card">
          <div className="card-head"><div><p className="kicker">Next-season games</p><h2>AIO validation</h2></div></div>
          <table className="mini"><thead><tr><th>Season</th><th>Baseline</th><th>Selected</th><th>Change</th></tr></thead><tbody>{lab.validation.map((row) => <tr key={row.test_season}><td>{row.test_season}</td><td>{row.baseline_rmse.toFixed(3)}</td><td>{row.selected_rmse.toFixed(3)}</td><td>{fmtRating(row.rmse_delta)}</td></tr>)}</tbody></table>
          <p className="note">Season stabilization uses only that season. Five-year pooling happens after each annual estimate is frozen.</p>
        </section>
      </>}
    </div>
  );
}
