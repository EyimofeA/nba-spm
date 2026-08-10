"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type AnnualRating = {
  Season: number;
  Poss_Off: number;
  Poss_Def: number;
  aio_net: number;
  aio_offense: number;
  aio_defense: number;
  spm_center_net: number;
  normal_rapm_net: number;
  rapm_update_net: number;
};

type RollingRating = {
  window_start: number;
  window_end: number;
  window_seasons: number;
  Poss_Off: number;
  Poss_Def: number;
  offense: number;
  defense: number;
  net: number;
};

type PeakRating = RollingRating & {
  peak_component: "offense" | "defense" | "net";
  peak_value: number;
  all_time_rank: number;
};

type Player = {
  PLAYER_ID: number;
  PLAYER_NAME: string;
  annual: AnnualRating[];
  rolling: RollingRating[];
  peaks: PeakRating[];
};

type SearchResult = { PLAYER_ID: number; PLAYER_NAME: string };
type View = "annual" | "3y" | "5y";

const API = process.env.NEXT_PUBLIC_RATINGS_API_URL ?? "http://127.0.0.1:8765";
const QUICK_PLAYERS = [
  [2544, "LeBron"],
  [201939, "Curry"],
  [203999, "Jokić"],
  [708, "Garnett"],
] as const;

const formatRating = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;

function TrajectoryChart({ player, view }: { player: Player; view: View }) {
  const series = useMemo(() => {
    if (view === "annual") {
      return player.annual.map((row) => ({
        label: String(row.Season),
        offense: row.aio_offense,
        defense: row.aio_defense,
        net: row.aio_net,
      }));
    }
    const length = view === "3y" ? 3 : 5;
    return player.rolling
      .filter((row) => row.window_seasons === length)
      .map((row) => ({
        label: String(row.window_end),
        offense: row.offense,
        defense: row.defense,
        net: row.net,
      }));
  }, [player, view]);

  if (!series.length) {
    return <div className="empty-chart">No {view.toUpperCase()} ratings for this player.</div>;
  }

  const width = 900;
  const height = 330;
  const pad = { left: 48, right: 24, top: 24, bottom: 42 };
  const values = series.flatMap((row) => [row.offense, row.defense, row.net]);
  const bound = Math.max(4, Math.ceil(Math.max(...values.map(Math.abs)) / 2) * 2);
  const x = (index: number) =>
    pad.left + (index * (width - pad.left - pad.right)) / Math.max(1, series.length - 1);
  const y = (value: number) =>
    pad.top + ((bound - value) * (height - pad.top - pad.bottom)) / (bound * 2);
  const line = (key: "offense" | "defense" | "net") =>
    series.map((row, index) => `${x(index)},${y(row[key])}`).join(" ");
  const ticks = [-bound, -bound / 2, 0, bound / 2, bound];
  const labelStep = Math.max(1, Math.ceil(series.length / 8));

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${view} player impact trajectory`}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className={tick === 0 ? "zero-line" : "grid-line"} x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} />
            <text className="axis-label" x={pad.left - 10} y={y(tick) + 4} textAnchor="end">{tick > 0 ? `+${tick}` : tick}</text>
          </g>
        ))}
        <polyline className="trajectory offense-line" points={line("offense")} />
        <polyline className="trajectory defense-line" points={line("defense")} />
        <polyline className="trajectory net-line" points={line("net")} />
        {series.map((row, index) => (
          <g key={row.label}>
            <circle className="net-dot" cx={x(index)} cy={y(row.net)} r="4" />
            {(index % labelStep === 0 || index === series.length - 1) && (
              <text className="year-label" x={x(index)} y={height - 14} textAnchor="middle">{row.label}</text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}

function RatingBar({ label, value }: { label: string; value: number }) {
  const width = Math.min(100, Math.abs(value) * 11);
  return (
    <div className="rating-row">
      <span>{label}</span>
      <div className="rating-track" aria-hidden="true">
        <div className={value >= 0 ? "rating-fill positive" : "rating-fill negative"} style={{ width: `${width}%` }} />
      </div>
      <strong>{formatRating(value)}</strong>
    </div>
  );
}

export function PlayerLab() {
  const [player, setPlayer] = useState<Player | null>(null);
  const [view, setView] = useState<View>("annual");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState("Loading pinned research ratings…");

  async function loadPlayer(id: number) {
    setStatus("Loading player…");
    try {
      const response = await fetch(`${API}/v1/players/${id}`);
      if (!response.ok) throw new Error("Player not found");
      setPlayer(await response.json());
      setResults([]);
      setQuery("");
      setStatus("");
    } catch {
      setStatus("Ratings API is offline. Start the local API and reload this page.");
    }
  }

  useEffect(() => {
    loadPlayer(2544);
  }, []);

  async function search(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    try {
      const response = await fetch(`${API}/v1/players/search?q=${encodeURIComponent(query)}&limit=8`);
      if (!response.ok) throw new Error("Search failed");
      const payload = await response.json();
      setResults(payload.results);
      setStatus(payload.results.length ? "" : "No matching player.");
    } catch {
      setStatus("Ratings API is offline. Start the local API and try again.");
    }
  }

  const latest = player?.annual.at(-1);
  const netPeaks = player?.peaks.filter((peak) => peak.peak_component === "net") ?? [];

  return (
    <main>
      <header className="site-header">
        <a className="wordmark" href="#top" aria-label="NBA Impact Lab home"><span>NBA</span> IMPACT LAB</a>
        <div className="research-badge">RESEARCH BUILD · DATA THROUGH 2024</div>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">PLAYER TRAJECTORY / 01</p>
          <h1>{player?.PLAYER_NAME ?? "Player impact, without the black box."}</h1>
          <p className="lede">Annual AIO explains the statistical center and lineup adjustment. Rolling normal RAPM shows what the possession data says on its own.</p>
        </div>
        <form className="player-search" onSubmit={search}>
          <label htmlFor="player-query">Find a player</label>
          <div className="search-row">
            <input id="player-query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. Tim Duncan" autoComplete="off" />
            <button type="submit">Search</button>
          </div>
          {results.length > 0 && <div className="search-results">{results.map((result) => <button type="button" key={result.PLAYER_ID} onClick={() => loadPlayer(result.PLAYER_ID)}>{result.PLAYER_NAME}</button>)}</div>}
          <div className="quick-players" aria-label="Quick player selections">
            {QUICK_PLAYERS.map(([id, name]) => <button type="button" key={id} onClick={() => loadPlayer(id)}>{name}</button>)}
          </div>
        </form>
      </section>

      {status && <div className="status" role="status">{status}</div>}

      {player && latest && <>
        <section className="score-strip" aria-label={`${latest.Season} rating summary`}>
          <div><span>{latest.Season} AIO</span><strong>{formatRating(latest.aio_net)}</strong></div>
          <div><span>OFFENSE</span><strong>{formatRating(latest.aio_offense)}</strong></div>
          <div><span>DEFENSE</span><strong>{formatRating(latest.aio_defense)}</strong></div>
          <div><span>MIN. SIDE POSS.</span><strong>{Math.min(latest.Poss_Off, latest.Poss_Def).toLocaleString()}</strong></div>
        </section>

        <section className="trajectory-section">
          <div className="section-heading">
            <div><p className="eyebrow">IMPACT OVER TIME</p><h2>Three views. One player.</h2></div>
            <div className="view-tabs" role="tablist" aria-label="Trajectory window">
              {(["annual", "3y", "5y"] as View[]).map((option) => <button role="tab" aria-selected={view === option} className={view === option ? "active" : ""} key={option} onClick={() => setView(option)}>{option === "annual" ? "Annual AIO" : `${option.toUpperCase()} RAPM`}</button>)}
            </div>
          </div>
          <div className="legend"><span className="net-key">Net</span><span className="offense-key">Offense</span><span className="defense-key">Defense</span><small>points per 100 possessions</small></div>
          <TrajectoryChart player={player} view={view} />
        </section>

        <section className="detail-grid">
          <article className="decomposition-card">
            <p className="eyebrow">{latest.Season} AIO DECOMPOSITION</p><h2>Center, then update.</h2>
            <p className="formula">SPM center <b>{formatRating(latest.spm_center_net)}</b> + RAPM update <b>{formatRating(latest.rapm_update_net)}</b> = AIO <b>{formatRating(latest.aio_net)}</b></p>
            <RatingBar label="SPM center" value={latest.spm_center_net} />
            <RatingBar label="RAPM update" value={latest.rapm_update_net} />
            <RatingBar label="Final AIO" value={latest.aio_net} />
            <p className="method-note">The SPM maps same-season statistical features to impact. Possession RAPM then adjusts that center for who shared the floor and who they faced.</p>
          </article>
          <article className="peak-card">
            <p className="eyebrow">NORMAL RAPM PEAKS</p><h2>Best sustained windows.</h2>
            {netPeaks.sort((a, b) => a.window_seasons - b.window_seasons).map((peak) => <div className="peak-row" key={peak.window_seasons}>
              <div><span>{peak.window_seasons}-YEAR</span><strong>{peak.window_start}–{String(peak.window_end).slice(-2)}</strong></div>
              <div><span>NET</span><strong>{formatRating(peak.net)}</strong></div>
              <div><span>ALL-TIME</span><strong>#{peak.all_time_rank}</strong></div>
            </div>)}
            <p className="method-note">Zero-prior normal RAPM, regular season only. Peak selection is descriptive and winner’s-curse biased.</p>
          </article>
        </section>

        <aside className="caveat-band"><strong>Read this correctly.</strong><span>Annual AIO and rolling RAPM answer different questions. The archive is stale after 2024, the 2024 cache is one game short, and uncertainty is not estimated yet.</span></aside>
      </>}

      <footer>NBA IMPACT LAB <span>·</span> MODEL VERSION IS PART OF THE RESULT</footer>
    </main>
  );
}
