"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Figure } from "../charts/frame";
import { RoleDatum, RoleMap } from "../charts/scatter";
import { Catalog, Player, RolePoint, RoleSide, loadRoleMap } from "../lib/data";

const SVG_NS = "http://www.w3.org/2000/svg";

function exportRoleMapPng(
  host: HTMLDivElement | null,
  player: RoleDatum,
  similar: RoleDatum[],
  side: RoleSide,
  season: number,
) {
  const chart = host?.querySelector("svg");
  if (!chart) return;

  const root = getComputedStyle(document.documentElement);
  const color = (name: string, fallback: string) =>
    root.getPropertyValue(name).trim() || fallback;
  const output = document.createElementNS(SVG_NS, "svg");
  output.setAttribute("xmlns", SVG_NS);
  output.setAttribute("viewBox", "0 0 880 725");
  output.setAttribute("width", "1760");
  output.setAttribute("height", "1450");
  output.style.setProperty("--text-muted", color("--text-muted", "#8793a5"));
  output.style.setProperty("--series-1", color("--series-1", "#63a5ff"));
  output.style.setProperty("--series-2", color("--series-2", "#f0a45d"));

  const background = document.createElementNS(SVG_NS, "rect");
  background.setAttribute("width", "880");
  background.setAttribute("height", "725");
  background.setAttribute("fill", color("--surface-1", "#11161c"));
  output.append(background);

  const title = document.createElementNS(SVG_NS, "text");
  title.setAttribute("x", "32");
  title.setAttribute("y", "34");
  title.setAttribute("fill", color("--text-primary", "#edf1f7"));
  title.setAttribute("font-family", "system-ui, sans-serif");
  title.setAttribute("font-size", "20");
  title.setAttribute("font-weight", "700");
  title.textContent = `${player.name} · ${side === "offense" ? "Offensive" : "Defensive"} role map`;
  output.append(title);

  const subtitle = document.createElementNS(SVG_NS, "text");
  subtitle.setAttribute("x", "32");
  subtitle.setAttribute("y", "56");
  subtitle.setAttribute("fill", color("--text-secondary", "#aeb7c5"));
  subtitle.setAttribute("font-family", "ui-monospace, monospace");
  subtitle.setAttribute("font-size", "11");
  subtitle.textContent = `${season - 1}–${String(season).slice(2)} · ${player.role} · CourtSignal`;
  output.append(subtitle);

  const copy = chart.cloneNode(true) as SVGSVGElement;
  copy.setAttribute("x", "0");
  copy.setAttribute("y", "70");
  copy.setAttribute("width", "880");
  copy.setAttribute("height", "500");
  copy.querySelectorAll(".hit").forEach((node) => node.remove());
  const style = document.createElementNS(SVG_NS, "style");
  style.textContent = `.grid-line{stroke:${color("--grid", "#2b333d")};stroke-width:1}.axis-title{fill:${color("--text-muted", "#8793a5")};font:11px ui-monospace,monospace}.mark-label{fill:${color("--text-primary", "#edf1f7")};font:600 12px system-ui,sans-serif}`;
  copy.prepend(style);
  output.append(copy);

  const divider = document.createElementNS(SVG_NS, "line");
  divider.setAttribute("x1", "32");
  divider.setAttribute("x2", "848");
  divider.setAttribute("y1", "590");
  divider.setAttribute("y2", "590");
  divider.setAttribute("stroke", color("--grid", "#2b333d"));
  output.append(divider);

  const neighbourLabel = document.createElementNS(SVG_NS, "text");
  neighbourLabel.setAttribute("x", "32");
  neighbourLabel.setAttribute("y", "616");
  neighbourLabel.setAttribute("fill", color("--text-muted", "#8793a5"));
  neighbourLabel.setAttribute("font-family", "ui-monospace, monospace");
  neighbourLabel.setAttribute("font-size", "10");
  neighbourLabel.setAttribute("font-weight", "700");
  neighbourLabel.setAttribute("letter-spacing", "1.4");
  neighbourLabel.textContent = "NEAREST ROLE NEIGHBOURS";
  output.append(neighbourLabel);

  similar.slice(0, 8).forEach((neighbour, index) => {
    const x = 32 + (index % 4) * 210;
    const y = 646 + Math.floor(index / 4) * 48;
    const name = document.createElementNS(SVG_NS, "text");
    name.setAttribute("x", String(x));
    name.setAttribute("y", String(y));
    name.setAttribute("fill", color("--text-primary", "#edf1f7"));
    name.setAttribute("font-family", "system-ui, sans-serif");
    name.setAttribute("font-size", "11");
    name.setAttribute("font-weight", "650");
    name.textContent = neighbour.name.length > 24
      ? `${neighbour.name.slice(0, 23)}…`
      : neighbour.name;
    output.append(name);
    const role = document.createElementNS(SVG_NS, "text");
    role.setAttribute("x", String(x));
    role.setAttribute("y", String(y + 16));
    role.setAttribute("fill", color("--text-secondary", "#aeb7c5"));
    role.setAttribute("font-family", "ui-monospace, monospace");
    role.setAttribute("font-size", "9");
    role.textContent = neighbour.role.length > 28
      ? `${neighbour.role.slice(0, 27)}…`
      : neighbour.role;
    output.append(role);
  });

  const blob = new Blob([new XMLSerializer().serializeToString(output)], {
    type: "image/svg+xml;charset=utf-8",
  });
  const sourceUrl = URL.createObjectURL(blob);
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 1760;
    canvas.height = 1450;
    canvas.getContext("2d")?.drawImage(image, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(sourceUrl);
    canvas.toBlob((png) => {
      if (!png) return;
      const downloadUrl = URL.createObjectURL(png);
      const link = document.createElement("a");
      const slug = player.name.normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
        .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
      link.href = downloadUrl;
      link.download = `${slug}-${side}-role-map-${season}.png`;
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0);
    }, "image/png");
  };
  image.onerror = () => URL.revokeObjectURL(sourceUrl);
  image.src = sourceUrl;
}

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
  const mapRef = useRef<HTMLDivElement>(null);

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
          <div ref={mapRef}>
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
          </div>

          {chosen && <section className="role-focus card" aria-label="Selected player role">
            <PlayerFace id={chosen.id} name={chosen.name} />
            <div>
              <p className="kicker">Selected player</p>
              <h2>{chosen.name}</h2>
              <p className="role-focus-meta">{chosen.team ?? "—"} · {chosen.role}</p>
            </div>
            <div className="role-focus-actions">
              <button type="button" className="btn ghost small" onClick={() => exportRoleMapPng(mapRef.current, chosen, similar, side, season)}>Export PNG</button>
              <button type="button" className="btn ghost small" onClick={() => onPlayer(chosen.id)}>Open player</button>
            </div>
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
