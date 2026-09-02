"use client";

import { useEffect, useMemo, useState } from "react";
import { Figure } from "../charts/frame";
import { RoleDatum, RoleMap } from "../charts/scatter";
import { Catalog, Player, RolePoint, RoleSide, loadRoleMap } from "../lib/data";

export function RolesView({
  catalog,
  side,
  onSide,
  player,
  onPlayer,
}: {
  catalog: Catalog;
  side: RoleSide;
  onSide: (side: RoleSide) => void;
  player: Player | null;
  onPlayer: (id: number) => void;
}) {
  const available = useMemo(
    () => (catalog.catalog.role_seasons[side] ?? []).filter((year) => year >= 2017),
    [catalog, side],
  );
  const [chosenSeason, setChosenSeason] = useState<number | null>(null);
  const [points, setPoints] = useState<RolePoint[]>([]);
  const [emphasis, setEmphasis] = useState<string>("All");
  const [selectedId, setSelectedId] = useState<number | null>(player?.PLAYER_ID ?? null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  // Each side publishes its own season list, so the chosen season can fall out
  // of range when the side changes. Derive the effective season rather than
  // correcting it in an effect.
  const season =
    chosenSeason !== null && available.includes(chosenSeason)
      ? chosenSeason
      : available.length
        ? Math.max(...available)
        : 0;

  useEffect(() => {
    if (!season) return;
    let live = true;
    const run = async () => {
      setState("loading");
      try {
        const rows = await loadRoleMap(side, season);
        if (!live) return;
        setPoints(rows);
        // Open on the biggest cluster: with every role in one colour the map
        // shows no structure, so "All roles" is an option rather than a default.
        const tally = new Map<string, number>();
        for (const row of rows)
          tally.set(row.raw_role, (tally.get(row.raw_role) ?? 0) + 1);
        const biggest = [...tally.entries()].sort(
          (a, b) => b[1] - a[1],
        )[0]?.[0];
        setEmphasis(biggest ?? "All");
        setState("ready");
      } catch {
        if (live) setState("error");
      }
    };
    const start = window.setTimeout(() => void run(), 0);
    return () => {
      live = false;
      window.clearTimeout(start);
    };
  }, [side, season]);

  const shaped = useMemo<RoleDatum[]>(
    () =>
      points.map((point) => ({
        id: point.PLAYER_ID,
        name: point.PLAYER_NAME,
        team: point.TEAM_ABBREVIATION,
        x: point.x,
        y: point.y,
        role: point.raw_role,
      })),
    [points],
  );

  const roleNames = useMemo(
    () => [...new Set(shaped.map((row) => row.role))].sort(),
    [shaped],
  );
  const counts = useMemo(() => {
    const tally = new Map<string, number>();
    for (const row of shaped)
      tally.set(row.role, (tally.get(row.role) ?? 0) + 1);
    return tally;
  }, [shaped]);

  const chosen = shaped.find((row) => row.id === selectedId) ?? shaped[0];
  const similar = useMemo(() => {
    if (!chosen) return [];
    return shaped
      .filter((row) => row.id !== chosen.id)
      .map((row) => ({
        ...row,
        distance: Math.hypot(row.x - chosen.x, row.y - chosen.y),
      }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 8);
  }, [shaped, chosen]);

  return (
    <>
      <div className="page-head">
        <div>
          <p className="kicker">Roles</p>
          <h1>The behavioural map</h1>
        </div>
      </div>

      <div className="filters">
        <div className="field">
          <span>Side</span>
          <div className="segmented" role="group" aria-label="Side">
            {(["offense", "defense"] as RoleSide[]).map((item) => (
              <button
                key={item}
                type="button"
                aria-pressed={side === item}
                onClick={() => onSide(item)}
              >
                {item === "offense" ? "Offense" : "Defense"}
              </button>
            ))}
          </div>
        </div>
        <label className="field">
          <span>Season</span>
          <select
            value={season}
            onChange={(event) => setChosenSeason(Number(event.target.value))}
          >
            {[...available].reverse().map((year) => (
              <option key={year} value={year}>
                {year - 1}–{String(year).slice(2)}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Highlight role</span>
          <select
            value={emphasis}
            onChange={(event) => setEmphasis(event.target.value)}
          >
            <option value="All">All roles</option>
            {roleNames.map((role) => (
              <option key={role} value={role}>
                {role} ({counts.get(role) ?? 0})
              </option>
            ))}
          </select>
        </label>
      </div>

      {state === "error" ? (
        <div className="empty">
          <b>Role map unavailable.</b>
          <span>
            This snapshot has no{" "}
            <code>
              roles-{side}-{season}.json
            </code>{" "}
            shard.
          </span>
        </div>
      ) : (
        <div className="grid">
          <Figure
            kicker={
              side === "offense" ? "Offense clusters" : "Defense clusters"
            }
            title={emphasis === "All" ? "All roles" : emphasis}
          >
            <RoleMap
              rows={shaped}
              emphasis={emphasis === "All" ? undefined : emphasis}
              highlight={chosen?.id}
              onSelect={(row) => setSelectedId(row.id)}
            />
          </Figure>

          {chosen && <section className="role-focus card" aria-label="Selected player role">
            <PlayerFace id={chosen.id} name={chosen.name} />
            <div>
              <p className="kicker">Selected player</p>
              <h2>{chosen.name}</h2>
              <p className="role-focus-meta">{chosen.team ?? "—"} · {chosen.role}</p>
            </div>
            <button type="button" className="button-secondary" onClick={() => onPlayer(chosen.id)}>Open player</button>
          </section>}

          <div>
            <section className="card role-neighbour-panel">
              <div className="card-head">
                <div>
                  <p className="kicker">Nearest neighbours</p>
                  <h2>{chosen ? chosen.name : "No player selected"}</h2>
                </div>
              </div>
              {chosen ? (
                <>
                  <div className="role-neighbors" style={{ marginTop: 14 }}>
                    {similar.map((row) => (
                      <button
                        key={row.id}
                        type="button"
                        className="role-neighbor"
                        onClick={() => setSelectedId(row.id)}
                      >
                        <PlayerFace id={row.id} name={row.name} />
                        <span>{row.name}</span>
                        <small>{row.role}</small>
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <p className="note" style={{ marginTop: 12 }}>
                  Pick a player from the map or the search box to see who
                  occupies the same space.
                </p>
              )}
            </section>
          </div>
        </div>
      )}
    </>
  );
}

function PlayerFace({ id, name }: { id: number; name: string }) {
  return <span className="role-face" aria-hidden="true">
    <img src={`https://cdn.nba.com/headshots/nba/latest/260x190/${id}.png`} alt="" onError={(event) => { event.currentTarget.hidden = true; }} />
    <span>{name.split(/\s+/).map((part) => part[0]).slice(0, 2).join("")}</span>
  </span>;
}
