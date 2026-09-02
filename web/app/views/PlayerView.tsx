"use client";

import { useMemo, useState } from "react";
import { Figure, Legend } from "../charts/frame";
import { MultiLine } from "../charts/lines";
import { Pizza, RadarComparison, SKILLS, Slice, pizzaLegend } from "../charts/pizza";
import {
  Catalog,
  COMPONENT_LABEL,
  Component,
  ModelId,
  LocalPlayerSkills,
  LocalSkillIndex,
  Player,
  PlayerIndex,
  RapmLabPayload,
  Role,
  RoleSide,
  rating,
  resolveModel,
} from "../lib/data";
import {
  COMPONENT_COLOR,
  SERIES,
  fmtRating,
} from "../lib/viz";
import { ModelField } from "./controls";
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
  const seasons = player.annual.map((row) => row.Season);
  const current =
    player.annual.find((row) => row.Season === season) ?? player.annual.at(-1);
  const currentRows = current ? [current] : player.annual;
  const active = resolveModel(currentRows, model);
  const profile = player.profiles.find((row) => row.Season === season);
  const roles = player.roles.find((row) => row.Season === season);
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
      const value = rating(row, active.prefix, key);
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
  const currentSeason = current?.Season ?? season;
  const currentRatings = (['net', 'offense', 'defense'] as Component[]).map(
    (key) => ({ key, value: rating(current, active.prefix, key) }),
  );
  const selectedValue = rating(current, active.prefix, "net");
  const researchComparisons = (() => {
    if (!rapmLab) return [];
    const values: { label: string; value: number; note: string }[] = [];
    const wp = rapmLab.leaderboards
      .find((board) => board.id === `wp-spm-aio-${currentSeason}`)
      ?.rows.find((row) => row.player_name === player.PLAYER_NAME && row.candidate === "Zero WP-RAPM");
    if (typeof wp?.net_per_100 === "number") {
      values.push({ label: "WP-RAPM", value: wp.net_per_100, note: "WP change / 100" });
    }
    const seen = new Set<string>();
    for (const board of rapmLab.replication_leaderboards) {
      if (board.season !== currentSeason || seen.has(board.metric)) continue;
      const row = board.rows.find((candidate) => candidate.player === player.PLAYER_NAME);
      const value = row?.raptor ?? row?.net;
      if (typeof value === "number") {
        values.push({ label: board.metric.replace(" table", ""), value, note: "Reference rating" });
        seen.add(board.metric);
      }
    }
    return values;
  })();

  return (
    <section aria-labelledby="player-heading">
      <header className="player-hero">
        <div className="player-identity">
          <div>
            <p className="kicker">Player report · {active.label}</p>
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
            Net · {active.label}
          </div>
        </div>
      </header>

      <div className="filters" aria-label="Player report controls">
        <ModelField rows={currentRows} value={model} onChange={onModel} />
        <label className="field">
          <span>Season</span>
          <select
            value={season}
            onChange={(event) => onSeason(Number(event.target.value))}
          >
            {[...seasons].reverse().map((year) => (
              <option key={year} value={year}>
                {year - 1}–{String(year).slice(2)}
              </option>
            ))}
          </select>
        </label>
      </div>

      <section aria-labelledby="current-impact-heading">
        <div className="section-head" style={{ marginTop: 18 }}>
          <div>
            <p className="kicker">Current impact</p>
            <h2 id="current-impact-heading">
              {currentSeason - 1}–{String(currentSeason).slice(2)} ratings
            </h2>
          </div>
          <span className="meta">{active.label} · per 100 possessions</span>
        </div>
        <div className="kpi-row">
          {currentRatings.map(({ key, value }) => (
            <article className="tile" key={key}>
              <div className="tile-label">{COMPONENT_LABEL[key]}</div>
              <div
                className="tile-value"
                style={{ color: value !== undefined && value < 0 ? "var(--neg)" : "var(--pos)" }}
              >
                {fmtRating(value)}
              </div>
              <div className="tile-sub">{active.label}</div>
            </article>
          ))}
        </div>
      </section>

      {researchComparisons.length > 0 && (
        <section aria-labelledby="research-comparisons-heading">
          <div className="section-head" style={{ marginTop: 18 }}>
            <div>
              <p className="kicker">Local research</p>
              <h2 id="research-comparisons-heading">Replications and WP-RAPM</h2>
            </div>
          </div>
          <div className="kpi-row">
            {researchComparisons.map((item) => <article className="tile" key={item.label}>
              <div className="tile-label">{item.label}</div>
              <div className="tile-value">{fmtRating(item.value)}</div>
              <div className="tile-sub">{item.note}</div>
            </article>)}
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
          <RoleComparisonPlayer name={player.PLAYER_NAME} roles={roles} catalog={catalog} />
          {comparePlayer ? (
            <RoleComparisonPlayer
              name={comparePlayer.PLAYER_NAME}
              roles={comparePlayer.roles.find((row) => row.Season === season)}
              catalog={catalog}
            />
          ) : (
            <aside className="card">
              <p className="kicker">Role context</p>
              <h2>Usage is not impact</h2>
              <p className="note">Role labels describe a player’s behaviour in this season. They do not change the rating above.</p>
            </aside>
          )}
        </section>
      </div>
    </section>
  );
}

function ComparisonCards({ left, right, model, season }: { left: Player; right: Player; model: ModelId; season: number }) {
  const leftActive = resolveModel(left.annual, model);
  const rightActive = resolveModel(right.annual, model);
  const players = [
    { player: left, active: leftActive, row: left.annual.find((row) => row.Season === season) },
    { player: right, active: rightActive, row: right.annual.find((row) => row.Season === season) },
  ];
  return <div className="comparison-cards" style={{ marginTop: 16 }}>
    {players.map(({ player, active, row }) => <article key={player.PLAYER_ID} className="comparison-player">
      <h3>{player.PLAYER_NAME}</h3>
      <div className="comparison-values" aria-label={`${player.PLAYER_NAME} ${season} offense defense and net`}>
        {(["offense", "defense", "net"] as Component[]).map((key) => <span key={key}><small>{COMPONENT_LABEL[key]}</small><b>{fmtRating(rating(row, active.prefix, key))}</b></span>)}
      </div>
    </article>)}
  </div>;
}

function RoleComparisonPlayer({ name, roles, catalog }: { name: string; roles?: Player["roles"][number]; catalog: Catalog }) {
  return <section className="role-comparison-player"><h2>{name} · role mix</h2><RoleMix title="Offense" role={roles?.offense} side="offense" catalog={catalog} /><RoleMix title="Defense" role={roles?.defense} side="defense" catalog={catalog} /></section>;
}

/**
 * Role affinities as a small bar set. Colour is keyed to the role's position in
 * the published label list, so a player with a different mix keeps the same hue
 * per role — colour follows the entity, never the row order.
 */
function RoleMix({
  title,
  role,
  side,
  catalog,
}: {
  title: string;
  role?: Role;
  side: RoleSide;
  catalog: Catalog;
}) {
  const order = Object.keys(catalog.catalog.role_labels?.[side] ?? {});
  if (!role) {
    return (
      <div>
        <h3>{title}</h3>
        <p className="note">No role data for this season.</p>
      </div>
    );
  }
  const colorFor = (roleId: string) => {
    const slot = order.indexOf(roleId);
    return SERIES[(slot < 0 ? 0 : slot) % SERIES.length];
  };
  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: 10,
        }}
      >
        <h3>{title}</h3>
        <span className="meta" style={{ fontSize: 11 }}>
          {role.primary_role}
        </span>
      </div>
      <div className="mix" style={{ marginTop: 10 }}>
        {role.memberships.map((membership) => (
          <div className="mix-row" key={membership.role_id}>
            <span className="label">
              <i style={{ background: colorFor(membership.role_id) }} />
              {membership.label}
            </span>
            <b>{Math.round(membership.affinity * 100)}%</b>
            <span className="track">
              <i
                style={{
                  width: `${Math.max(1, membership.affinity * 100)}%`,
                  background: colorFor(membership.role_id),
                }}
              />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
