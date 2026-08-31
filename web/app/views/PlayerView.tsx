"use client";

import { useMemo, useState } from "react";
import { Figure, Legend } from "../charts/frame";
import { MultiLine } from "../charts/lines";
import { Pizza, RadarComparison, SKILL_DEFINITIONS, SKILLS, Slice, pizzaLegend } from "../charts/pizza";
import {
  Catalog,
  COMPONENT_LABEL,
  Component,
  ModelId,
  LocalPlayerSkills,
  LocalSkillIndex,
  Player,
  PlayerIndex,
  Role,
  RoleSide,
  rating,
  resolveModel,
} from "../lib/data";
import {
  COMPONENT_COLOR,
  SERIES,
  fmtRating,
  ordinalSuffix,
} from "../lib/viz";
import { ComponentToggle, ModelField } from "./controls";
import { PlayerSkills } from "./PlayerSkills";

type PlayerViewProps = {
  catalog: Catalog;
  season: number;
  onSeason: (season: number) => void;
  model: ModelId;
  onModel: (model: ModelId) => void;
  component: Component;
  onComponent: (component: Component) => void;
  index: PlayerIndex[];
  comparePlayer: Player | null;
  localSkillIndex: LocalSkillIndex | null;
  localSkills: LocalPlayerSkills | null;
  compareLocalSkills: LocalPlayerSkills | null;
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
  component,
  onComponent,
  index,
  comparePlayer,
  localSkillIndex,
  localSkills,
  compareLocalSkills,
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
  const selectedValue = rating(current, active.prefix, component);

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
            {COMPONENT_LABEL[component]} · {active.label}
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
        <ComponentToggle value={component} onChange={onComponent} />
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
          table={
            <table className="mini">
              <thead>
                <tr>
                  <th scope="col">Season</th>
                  <th scope="col">Team</th>
                  <th scope="col">Off</th>
                  <th scope="col">Def</th>
                  <th scope="col">Net</th>
                </tr>
              </thead>
              <tbody>
                {player.annual.map((row) => (
                  <tr
                    key={row.Season}
                    className={row.Season === season ? "flag" : undefined}
                  >
                    <td>
                      <button
                        type="button"
                        className="text-button"
                        aria-current={row.Season === season ? "true" : undefined}
                        aria-label={`Show ${row.Season - 1}–${String(row.Season).slice(2)} ratings`}
                        onClick={() => onSeason(row.Season)}
                      >
                        {row.Season - 1}–{String(row.Season).slice(2)}
                      </button>
                    </td>
                    <td>{row.TEAM_ABBREVIATION ?? "—"}</td>
                    <td>{fmtRating(rating(row, active.prefix, "offense"))}</td>
                    <td>{fmtRating(rating(row, active.prefix, "defense"))}</td>
                    <td>{fmtRating(rating(row, active.prefix, "net"))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
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
            note="Percentiles compare this player to the rest of the season. Open the table for the inputs behind each skill."
            table={
              <table className="mini">
                <thead>
                  <tr>
                    <th scope="col">Skill</th>
                    <th scope="col">Group</th>
                    <th scope="col">Inputs</th>
                    <th scope="col">Percentile</th>
                  </tr>
                </thead>
                <tbody>
                  {[...slices]
                    .sort((a, b) => b.value - a.value)
                    .map((slice) => (
                      <tr key={slice.key}>
                        <td>{slice.label}</td>
                        <td>
                          {slice.group === "shared" ? "Both ends" : slice.group}
                        </td>
                        <td>{SKILL_DEFINITIONS[slice.key]}</td>
                        <td>{ordinalSuffix(slice.value)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            }
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
              <ComparisonTable left={player} right={comparePlayer} model={model} />
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
            table={<SkillComparisonTable left={slices} right={comparisonSlices} leftName={player.PLAYER_NAME} rightName={comparePlayer.PLAYER_NAME} />}
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

function ComparisonTable({ left, right, model }: { left: Player; right: Player; model: ModelId }) {
  const leftActive = resolveModel(left.annual, model);
  const rightActive = resolveModel(right.annual, model);
  const rightBySeason = new Map(right.annual.map((row) => [row.Season, row]));
  const seasons = left.annual
    .filter((row) => rightBySeason.has(row.Season))
    .map((row) => row.Season)
    .sort((a, b) => b - a);
  return <div className="table-wrap" style={{ marginTop: 16 }}>
    <table className="mini comparison-table">
      <caption className="visually-hidden">{left.PLAYER_NAME} and {right.PLAYER_NAME} comparison</caption>
      <thead><tr><th scope="col">Season</th><th scope="col">{left.PLAYER_NAME}<br />Off / Def / Net</th><th scope="col">{right.PLAYER_NAME}<br />Off / Def / Net</th></tr></thead>
      <tbody>{seasons.map((season) => { const leftRow = left.annual.find((row) => row.Season === season); const rightRow = rightBySeason.get(season); return <tr key={season}><th scope="row">{season - 1}–{String(season).slice(2)}</th><td>{["offense", "defense", "net"].map((key) => fmtRating(rating(leftRow, leftActive.prefix, key as Component))).join(" / ")}</td><td>{["offense", "defense", "net"].map((key) => fmtRating(rating(rightRow, rightActive.prefix, key as Component))).join(" / ")}</td></tr>; })}</tbody>
    </table>
  </div>;
}

function SkillComparisonTable({ left, right, leftName, rightName }: { left: Slice[]; right: Slice[]; leftName: string; rightName: string }) {
  const rightByKey = new Map(right.map((slice) => [slice.key, slice]));
  return <table className="mini comparison-table"><thead><tr><th scope="col">Skill</th><th scope="col">{leftName}</th><th scope="col">{rightName}</th></tr></thead><tbody>{left.map((slice) => <tr key={slice.key}><td>{slice.label}</td><td>{ordinalSuffix(slice.value)}</td><td>{rightByKey.get(slice.key) ? ordinalSuffix(rightByKey.get(slice.key)!.value) : "—"}</td></tr>)}</tbody></table>;
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
