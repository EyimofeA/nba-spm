"use client";

import { useMemo } from "react";
import { Distribution } from "../charts/bars";
import { Figure, Legend } from "../charts/frame";
import { MultiLine } from "../charts/lines";
import { Pizza, SKILLS, Slice, pizzaLegend } from "../charts/pizza";
import {
  Catalog,
  COMPONENT_LABEL,
  Component,
  LeaderboardRow,
  ModelId,
  Player,
  Role,
  RoleSide,
  possessions,
  rating,
  resolveModel,
  missingModelNote,
} from "../lib/data";
import {
  COMPONENT_COLOR,
  SERIES,
  fmtInt,
  fmtRating,
  ordinalSuffix,
} from "../lib/viz";
import { ComponentToggle, ModelField } from "./controls";

type PlayerViewProps = {
  catalog: Catalog;
  season: number;
  onSeason: (season: number) => void;
  model: ModelId;
  onModel: (model: ModelId) => void;
  component: Component;
  onComponent: (component: Component) => void;
  peers: LeaderboardRow[];
  peerSeason: number;
};

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
  peers,
  peerSeason,
}: PlayerViewProps & { player: Player }) {
  const active = resolveModel(player.annual, model);
  const seasons = player.annual.map((row) => row.Season);
  const current =
    player.annual.find((row) => row.Season === season) ?? player.annual.at(-1);
  const profile = player.profiles.find((row) => row.Season === season);
  const roles = player.roles.find((row) => row.Season === season);

  const net = rating(current, active.prefix, "net");
  const spmCenter = rating(current, "spm_", component);
  const aioValue = rating(current, "aio_", component);

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

  // Left unmemoized on purpose: the compiler handles it, and a hand-written dep
  // list on `active.prefix` is something it cannot verify.
  const swarm = peers
    .filter((row) => possessions(row) >= 500)
    .flatMap((row) => {
      const value = rating(row, active.prefix, component);
      return value === undefined
        ? []
        : [
            {
              id: row.PLAYER_ID,
              name: row.PLAYER_NAME,
              team: row.TEAM_ABBREVIATION,
              value,
            },
          ];
    });
  const mine = swarm.find((row) => row.id === player.PLAYER_ID);
  const rank = mine
    ? (() => {
        const better = swarm.filter((row) => row.value > mine.value).length;
        return {
          place: better + 1,
          of: swarm.length,
          percentile: (1 - better / swarm.length) * 100,
        };
      })()
    : null;

  return (
    <>
      <div className="player-hero">
        <div>
          <p className="kicker">{active.label} · points per 100 possessions</p>
          <h1>{player.PLAYER_NAME}</h1>
          <div className="id-line">
            <span className="chip">{current?.TEAM_ABBREVIATION ?? "—"}</span>
            <span className="chip">
              {season - 1}–{String(season).slice(2)}
            </span>
            <span className="chip">
              {current ? `${fmtInt(possessions(current as never))} poss` : "—"}
            </span>
            {roles?.offense && (
              <span className="chip">{roles.offense.primary_role}</span>
            )}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div
            className={`hero-figure ${net !== undefined && net < 0 ? "neg" : "pos"}`}
          >
            {fmtRating(net)}
          </div>
          <div className="hero-caption">Net impact · {season}</div>
        </div>
      </div>

      <div className="filters">
        <ModelField rows={player.annual} value={model} onChange={onModel} />
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

      <div className="grid">
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
              {missingModelNote(player.annual)
                ? ` ${missingModelNote(player.annual)}`
                : ""}
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
                  <th scope="col">Poss</th>
                </tr>
              </thead>
              <tbody>
                {player.annual.map((row) => (
                  <tr
                    key={row.Season}
                    className={row.Season === season ? "flag" : undefined}
                  >
                    <td>{row.Season}</td>
                    <td>{row.TEAM_ABBREVIATION ?? "—"}</td>
                    <td>{fmtRating(rating(row, active.prefix, "offense"))}</td>
                    <td>{fmtRating(rating(row, active.prefix, "defense"))}</td>
                    <td>{fmtRating(rating(row, active.prefix, "net"))}</td>
                    <td>{fmtInt(possessions(row as never))}</td>
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

        {active.id === "aio" &&
          spmCenter !== undefined &&
          aioValue !== undefined && (
            <section className="card">
              <div className="card-head">
                <div>
                  <p className="kicker">Decomposition</p>
                  <h2>
                    Where the {COMPONENT_LABEL[component].toLowerCase()} number
                    comes from
                  </h2>
                </div>
              </div>
              <div className="equation" style={{ marginTop: 14 }}>
                <div>
                  <span>SPM center</span>
                  <b>{fmtRating(spmCenter)}</b>
                </div>
                <i>+</i>
                <div>
                  <span>RAPM update</span>
                  <b>{fmtRating(aioValue - spmCenter)}</b>
                </div>
                <i>=</i>
                <div>
                  <span>AIO {COMPONENT_LABEL[component].toLowerCase()}</span>
                  <b>{fmtRating(aioValue)}</b>
                </div>
              </div>
              <p className="note">{catalog.methods.rapm_update_note}</p>
            </section>
          )}

        <div className="grid two">
          <Figure
            kicker={`Skill profile · ${season}`}
            title="Season-relative percentile"
            legend={<Legend items={pizzaLegend} />}
            note="Rings mark the 25th, 50th, and 75th percentiles. Percentiles compare this player to the rest of the season, not to history. Skills are descriptive: they are not rating inputs."
            table={
              <table className="mini">
                <thead>
                  <tr>
                    <th scope="col">Skill</th>
                    <th scope="col">Group</th>
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
                        <td>{ordinalSuffix(slice.value)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            }
          >
            <Pizza slices={slices} />
          </Figure>

          <section className="card">
            <div className="card-head">
              <div>
                <p className="kicker">Role mix · {season}</p>
                <h2>How he was used</h2>
              </div>
            </div>
            <div style={{ display: "grid", gap: 18, marginTop: 16 }}>
              <RoleMix
                title="Offense"
                role={roles?.offense}
                side="offense"
                catalog={catalog}
              />
              <RoleMix
                title="Defense"
                role={roles?.defense}
                side="defense"
                catalog={catalog}
              />
            </div>
            <p className="note">
              Clusters come from behaviour only, one season at a time. Height
              and listed position are excluded. Roles describe usage, not
              impact.
            </p>
          </section>
        </div>

        {rank && (
          <Figure
            kicker={`Against the ${peerSeason} field`}
            title={`${COMPONENT_LABEL[component]} among rated players`}
            note={`Every player-season with 500 or more possessions on the smaller side. ${player.PLAYER_NAME} ranks ${rank.place} of ${rank.of} — ${ordinalSuffix(rank.percentile)} percentile.`}
            table={
              <table className="mini">
                <thead>
                  <tr>
                    <th scope="col">#</th>
                    <th scope="col">Player</th>
                    <th scope="col">{COMPONENT_LABEL[component]}</th>
                  </tr>
                </thead>
                <tbody>
                  {[...swarm]
                    .sort((a, b) => b.value - a.value)
                    .slice(0, 40)
                    .map((row, position) => (
                      <tr
                        key={row.id}
                        className={
                          row.id === player.PLAYER_ID ? "flag" : undefined
                        }
                      >
                        <td>{position + 1}</td>
                        <td>{row.name}</td>
                        <td>{fmtRating(row.value)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            }
          >
            <Distribution
              rows={swarm}
              highlight={player.PLAYER_ID}
              label={COMPONENT_LABEL[component]}
            />
          </Figure>
        )}
      </div>
    </>
  );
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
