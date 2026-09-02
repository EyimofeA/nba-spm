"use client";

import { useEffect, useMemo, useState } from "react";
import { Figure, Legend } from "../charts/frame";
import { MultiLine } from "../charts/lines";
import { Pizza, RadarComparison, SKILLS, Slice, pizzaLegend } from "../charts/pizza";
import {
  Catalog,
  COMPONENT_LABEL,
  Component,
  LeaderboardRow,
  ModelId,
  LocalPlayerSkills,
  LocalSkillIndex,
  Player,
  PlayerIndex,
  RapmLabPayload,
  rating,
  availableModels,
  loadSeason,
  resolveModel,
} from "../lib/data";
import { COMPONENT_COLOR, fmtRating, ordinalSuffix } from "../lib/viz";
import { ComparisonCards, RoleComparisonPlayer } from "./PlayerComparison";
import { PlayerSkills } from "./PlayerSkills";

type PlayerViewProps = {
  catalog: Catalog;
  season: number;
  onSeason: (season: number) => void;
  model: ModelId;
  onModel: (model: ModelId) => void;
  index: PlayerIndex[];
  comparePlayer: Player | null;
  localSkillIndex: LocalSkillIndex | null;
  localSkills: LocalPlayerSkills | null;
  compareLocalSkills: LocalPlayerSkills | null;
  rapmLab: RapmLabPayload | null;
  onCompare: (id: number) => void;
};
type ReconstructionModel = "raptor" | "pipm" | "darko_wowy";
const RECONSTRUCTION_LABELS: Record<ReconstructionModel, string> = {
  raptor: "RAPTOR",
  pipm: "PIPM",
  darko_wowy: "DARKO WOWY",
};
const RECONSTRUCTION_METRICS: Record<ReconstructionModel, string> = {
  raptor: "CourtSignal RAPTOR reconstruction",
  pipm: "CourtSignal PIPM reconstruction",
  darko_wowy: "CourtSignal DARKO WOWY reconstruction",
};

const foldName = (value: string) =>
  value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase();

/** Guard only. The body below always has a player, so its hooks are unconditional. */
export function PlayerView({
  player,
  ...props
}: PlayerViewProps & { player: Player | null }) {
  if (!player) {
    return (
      <div className="empty">
        <b>No player selected.</b>
        <span>Search by name above, or pick a row on the ratings board.</span>
      </div>
    );
  }
  return <PlayerBody {...props} player={player} />;
}

function PlayerBody({
  catalog,
  player,
  season,
  onSeason,
  model,
  onModel,
  index,
  comparePlayer,
  localSkillIndex,
  localSkills,
  compareLocalSkills,
  rapmLab,
  onCompare,
}: PlayerViewProps & { player: Player }) {
  const [compareQuery, setCompareQuery] = useState("");
  const [reconstructionModel, setReconstructionModel] = useState<ReconstructionModel | null>(null);
  const [seasonPool, setSeasonPool] = useState<LeaderboardRow[]>([]);
  const seasons = player.annual.map((row) => row.Season);
  const current =
    player.annual.find((row) => row.Season === season) ?? player.annual.at(-1);
  const currentSeason = current?.Season ?? season;
  const currentRows = current ? [current] : player.annual;
  const active = resolveModel(currentRows, model);
  const reconstructionBoards = rapmLab?.replication_leaderboards ?? [];
  const activeLabel = reconstructionModel ? RECONSTRUCTION_LABELS[reconstructionModel] : active.label;
  useEffect(() => {
    let live = true;
    loadSeason(currentSeason).then((rows) => {
      if (live) setSeasonPool(rows);
    }).catch(() => {
      if (live) setSeasonPool([]);
    });
    return () => { live = false; };
  }, [currentSeason]);
  const displayValue = (key: Component, row = current) => {
    if (!reconstructionModel) return rating(row, active.prefix, key);
    const boardRow = reconstructionBoards
      .find((board) => board.season === row?.Season && board.metric === RECONSTRUCTION_METRICS[reconstructionModel])
      ?.rows.find((candidate) => foldName(candidate.player) === foldName(player.PLAYER_NAME));
    const value = boardRow?.[key];
    return typeof value === "number" ? value : undefined;
  };
  const profile = player.profiles.find((row) => row.Season === season);
  const roleSeason = [...player.roles]
    .filter((row) => row.Season <= currentSeason)
    .sort((a, b) => b.Season - a.Season)[0];
  const roles = roleSeason;
  const compareMatches = useMemo(() => {
    const needle = foldName(compareQuery.trim());
    if (needle.length < 2) return [];
    return index
      .filter((item) => item.id !== player.PLAYER_ID && foldName(item.name).includes(needle))
      .slice(0, 6);
  }, [compareQuery, index, player.PLAYER_ID]);


  const series = (
    [
      ["net", COMPONENT_COLOR.net],
      ["offense", COMPONENT_COLOR.offense],
      ["defense", COMPONENT_COLOR.defense],
    ] as [Component, string][]
  ).map(([key, color]) => ({
    label: COMPONENT_LABEL[key],
    color,
    points: player.annual.flatMap((row) => {
      const value = displayValue(key, row);
      return value === undefined ? [] : [{ x: row.Season, y: value }];
    }),
  }));

  const slices = useMemo<Slice[]>(
    () =>
      profile
        ? SKILLS.flatMap((skill) => {
            const value = profile[skill.key];
            return typeof value === "number" ? [{ ...skill, value }] : [];
          })
        : [],
    [profile],
  );
  const comparisonProfile = comparePlayer?.profiles.find((row) => row.Season === season);
  const hasLocalSkills = Boolean(
    localSkillIndex && localSkills?.id === player.PLAYER_ID,
  );
  const comparisonSlices = useMemo<Slice[]>(
    () =>
      comparisonProfile
        ? SKILLS.flatMap((skill) => {
            const value = comparisonProfile[skill.key];
            return typeof value === "number" ? [{ ...skill, value }] : [];
          })
        : [],
    [comparisonProfile],
  );

  // Left unmemoized on purpose: the compiler handles it, and a hand-written dep
  // list on `active.prefix` is something it cannot verify.
  const currentRatings = (['net', 'offense', 'defense'] as Component[]).map(
    (key) => ({ key, value: displayValue(key) }),
  );
  const selectedValue = displayValue("net");
  const standingFor = (key: Component, value: number | undefined) => {
    if (value === undefined) return null;
    const values = reconstructionModel
      ? reconstructionBoards
          .find((board) => board.season === currentSeason && board.metric === RECONSTRUCTION_METRICS[reconstructionModel])
          ?.rows.flatMap((row) => typeof row[key] === "number" ? [row[key] as number] : []) ?? []
      : seasonPool
          .filter((row) => Math.min(Number(row.Poss_Off), Number(row.Poss_Def)) >= 100)
          .flatMap((row) => {
            const candidate = rating(row, active.prefix, key);
            return typeof candidate === "number" ? [candidate] : [];
          });
    if (!values.length) return null;
    const rank = 1 + values.filter((candidate) => candidate > value).length;
    const percentile = values.length === 1
      ? 100
      : Math.round(((values.length - rank) / (values.length - 1)) * 99 + 1);
    return { rank, percentile, total: values.length };
  };
  const selectedStanding = standingFor("net", selectedValue);
  const pulseEquation = current && active.id === "pulse" && !reconstructionModel
    ? [
        { label: "PULSE prior", value: current.pulse_prior_net },
        { label: "Lineup update", value: current.lineup_update_net },
        { label: "PULSE", value: current.pulse_net },
      ].map((item) => ({
        ...item,
        value: typeof item.value === "number" ? item.value : undefined,
      }))
    : [];
  const decomposition = current && active.id === "pulse" && !reconstructionModel ? [
    {
      side: "Offense",
      parts: [
        ["Shooting", current.pulse_offense_shooting_ts_contribution],
        ["Turnovers", current.pulse_offense_turnover_value_contribution],
        ["Off. rebounding", current.pulse_offense_offensive_rebound_value_contribution],
        ["Other", current.pulse_offense_residual],
      ],
    },
    {
      side: "Defense",
      parts: [
        ["Shooting", current.pulse_defense_shooting_ts_contribution],
        ["Turnovers", current.pulse_defense_turnover_value_contribution],
        ["Opp. OREB prevention", current.pulse_defense_opponent_oreb_prevention_contribution],
        ["Other", current.pulse_defense_residual],
      ],
    },
  ].filter((group) => group.parts.some(([, value]) => typeof value === "number")) : [];

  return (
    <section aria-labelledby="player-heading">
      <header className="player-hero">
        <div className="player-identity">
          <div>
            <p className="kicker">Player report · {activeLabel}</p>
            <h1 id="player-heading">{player.PLAYER_NAME}</h1>
            <div className="id-line">
              <span className="chip">{current?.TEAM_ABBREVIATION ?? "—"}</span>
              <span className="chip">
                {currentSeason - 1}–{String(currentSeason).slice(2)}
              </span>
              {roles?.offense && (
                <span className="chip">{roles.offense.primary_role}</span>
              )}
            </div>
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div
            className={`hero-figure ${selectedValue !== undefined && selectedValue < 0 ? "neg" : "pos"}`}
          >
            {fmtRating(selectedValue)}
          </div>
          <div className="hero-caption">
            Net · {activeLabel}{selectedStanding ? ` · #${selectedStanding.rank}/${selectedStanding.total} · ${ordinalSuffix(selectedStanding.percentile)}` : ""}
          </div>
        </div>
      </header>

      <div className="player-switcher" aria-label="Player report controls">
        <nav className="season-strip" aria-label="Player season">
          {[...seasons].reverse().map((year) => (
            <button key={year} type="button" aria-pressed={year === currentSeason} onClick={() => onSeason(year)}>
              {year - 1}–{String(year).slice(2)}
            </button>
          ))}
        </nav>
        <div className="model-tabs" role="tablist" aria-label="Player model">
          {availableModels(currentRows).filter((item) => item.available).map((item) => (
            <button key={item.id} type="button" role="tab" aria-selected={!reconstructionModel && active.id === item.id} onClick={() => { setReconstructionModel(null); onModel(item.id); }}>{item.label}</button>
          ))}
          {(Object.keys(RECONSTRUCTION_LABELS) as ReconstructionModel[]).filter((item) =>
            reconstructionBoards.some((board) => board.season === currentSeason && board.metric === RECONSTRUCTION_METRICS[item] && board.rows.some((row) => foldName(row.player) === foldName(player.PLAYER_NAME))),
          ).map((item) => <button key={item} type="button" role="tab" aria-selected={reconstructionModel === item} onClick={() => setReconstructionModel(item)}>{RECONSTRUCTION_LABELS[item]}</button>)}
        </div>
      </div>

      <section aria-labelledby="current-impact-heading">
        <div className="section-head" style={{ marginTop: 18 }}>
          <div>
            <p className="kicker">Current impact</p>
            <h2 id="current-impact-heading">
              {currentSeason - 1}–{String(currentSeason).slice(2)} ratings
            </h2>
          </div>
          <span className="meta">{activeLabel} · per 100 possessions</span>
        </div>
        <div className="kpi-row">
          {currentRatings.map(({ key, value }) => {
            const standing = standingFor(key, value);
            return (
            <article className="tile" key={key}>
              <div className="tile-label">{COMPONENT_LABEL[key]}</div>
              <div
                className="tile-value"
                style={{ color: value !== undefined && value < 0 ? "var(--neg)" : "var(--pos)" }}
              >
                {fmtRating(value)}
              </div>
              <div className="tile-sub">{activeLabel}{standing ? ` · #${standing.rank}/${standing.total} · ${ordinalSuffix(standing.percentile)}` : ""}</div>
            </article>
          );})}
        </div>
        {pulseEquation.every((item) => item.value !== undefined) && (
          <div className="pulse-equation" aria-label="PULSE prior plus lineup update equals PULSE">
            {pulseEquation.map((item, index) => (
              <div key={item.label}>
                {index > 0 && <i aria-hidden="true">{index === 1 ? "+" : "="}</i>}
                <span><small>{item.label}</small><b>{fmtRating(item.value)}</b></span>
              </div>
            ))}
          </div>
        )}
      </section>

      {decomposition.length > 0 && (
        <section aria-labelledby="pulse-decomposition-heading">
          <div className="section-head" style={{ marginTop: 18 }}>
            <div><p className="kicker">PULSE decomposition</p><h2 id="pulse-decomposition-heading">Where the rating comes from</h2></div>
            <span className="meta">Components reconcile to PULSE</span>
          </div>
          <div className="decomposition-grid">
            {decomposition.map((group) => (
              <article className="card decomposition-card" key={group.side}>
                <h3>{group.side}</h3>
                {group.parts.map(([label, value]) => (
                  <div className="decomposition-row" key={String(label)}>
                    <span>{label}</span><b>{fmtRating(typeof value === "number" ? value : undefined)}</b>
                  </div>
                ))}
              </article>
            ))}
          </div>
        </section>
      )}

      <div className="grid" style={{ marginTop: 14 }}>
        <Figure
          kicker="Career"
          title="Season by season"
          legend={
            <Legend
              items={series.map((s) => ({ label: s.label, color: s.color }))}
              shape="key"
            />
          }
          note={
            <>
              Select a season on the chart to move the whole page to it. Offense
              and defense add to net.
            </>
          }
        >
          <MultiLine
            series={series}
            xFormat={(value) => String(value)}
            xTitle="Season"
            selected={season}
            onSelectX={onSeason}
          />
        </Figure>

        {hasLocalSkills && localSkillIndex && localSkills && (
          <PlayerSkills
            index={localSkillIndex}
            player={localSkills}
            comparison={compareLocalSkills}
            season={season}
          />
        )}

        <div className={hasLocalSkills ? "grid" : "grid two"}>
          {!hasLocalSkills && <Figure
            kicker={`Skill profile · ${season}`}
            title="Season-relative percentile"
            legend={<Legend items={pizzaLegend} />}
            note="Percentiles compare this player to the rest of the season."
          >
            <Pizza slices={slices} />
          </Figure>}

          <section className="card compare-card" aria-labelledby="comparison-heading">
            <div className="card-head">
              <div>
                <p className="kicker">Peer comparison</p>
                <h2 id="comparison-heading">Same-season comparison</h2>
              </div>
            </div>
            <div className="compare-search">
              <label className="visually-hidden" htmlFor="player-comparison-search">
                Find a player to compare
              </label>
              <input
                id="player-comparison-search"
                placeholder="Find a player to compare"
                value={compareQuery}
                onChange={(event) => setCompareQuery(event.target.value)}
              />
              {compareMatches.length > 0 && (
                <div className="compare-results">
                  {compareMatches.map((item) => (
                    <button
                      type="button"
                      key={item.id}
                      onClick={() => {
                        onCompare(item.id);
                        setCompareQuery("");
                      }}
                    >
                      {item.name}
                    </button>
                  ))}
                </div>
              )}
            </div>
            {comparePlayer ? (
              <ComparisonCards left={player} right={comparePlayer} model={model} season={season} />
            ) : (
              <p className="note">Compare offense, defense, and net against a player from the same season.</p>
            )}
          </section>

        </div>

        {!hasLocalSkills && comparePlayer && slices.length >= 3 && comparisonSlices.length >= 3 && (
          <Figure
            kicker={`Skill comparison · ${season}`}
            title={`${player.PLAYER_NAME} vs ${comparePlayer.PLAYER_NAME}`}
            legend={<Legend items={[
              { label: player.PLAYER_NAME, color: "var(--series-1)" },
              { label: comparePlayer.PLAYER_NAME, color: "var(--series-2)" },
            ]} />}
            note="Each spoke is the same season-relative skill percentile used above."
          >
            <RadarComparison left={slices} right={comparisonSlices} leftName={player.PLAYER_NAME} rightName={comparePlayer.PLAYER_NAME} />
          </Figure>
        )}

        <section className="grid two" aria-label="Role detail">
          <RoleComparisonPlayer name={player.PLAYER_NAME} roles={roles} roleSeason={roleSeason?.Season} catalog={catalog} />
          {comparePlayer ? (
            <RoleComparisonPlayer
              name={comparePlayer.PLAYER_NAME}
              roles={[...comparePlayer.roles].filter((row) => row.Season <= currentSeason).sort((a, b) => b.Season - a.Season)[0]}
              roleSeason={[...comparePlayer.roles].filter((row) => row.Season <= currentSeason).sort((a, b) => b.Season - a.Season)[0]?.Season}
              catalog={catalog}
            />
          ) : null}
        </section>
      </div>
    </section>
  );
}
