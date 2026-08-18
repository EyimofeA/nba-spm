"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Catalog,
  Component,
  LeaderboardRow,
  ModelId,
  Player,
  PlayerIndex,
  RoleSide,
  loadCatalog,
  loadIndex,
  loadPlayer,
  loadSeason,
} from "./lib/data";
import { HomeView } from "./views/HomeView";
import { LandscapeView } from "./views/LandscapeView";
import { PlayerView } from "./views/PlayerView";
import { ProjectionsView } from "./views/ProjectionsView";
import { RatingsView } from "./views/RatingsView";
import { ResearchView } from "./views/ResearchView";
import { RolesView } from "./views/RolesView";

const TABS = [
  { id: "home", label: "Overview" },
  { id: "ratings", label: "Ratings" },
  { id: "landscape", label: "Landscape" },
  { id: "player", label: "Player" },
  { id: "roles", label: "Roles" },
  { id: "projections", label: "Projections" },
  { id: "research", label: "Research" },
] as const;

type Tab = (typeof TABS)[number]["id"];
const isTab = (value: string): value is Tab =>
  TABS.some((tab) => tab.id === value);

/** The location hash is the single source of truth for the current view. */
type Route = { tab: Tab; playerId?: number };

const readHash = (): Route => {
  const [head, id] = window.location.hash.slice(1).split("/");
  const parsed = Number(id);
  return {
    tab: isTab(head) ? head : "home",
    playerId: Number.isFinite(parsed) && parsed > 0 ? parsed : undefined,
  };
};

export function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [index, setIndex] = useState<PlayerIndex[]>([]);
  const [route, setRoute] = useState<Route>({ tab: "home" });
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");

  // Filters live at the shell so every view below reads the same slice.
  const [season, setSeason] = useState(2024);
  const [model, setModel] = useState<ModelId>("aio");
  const [component, setComponent] = useState<Component>("net");
  const [minPoss, setMinPoss] = useState(0);

  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [player, setPlayer] = useState<Player | null>(null);
  const [comparePlayer, setComparePlayer] = useState<Player | null>(null);
  const [playerSeason, setPlayerSeason] = useState(2024);
  const [roleSide, setRoleSide] = useState<RoleSide>("offense");

  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  /* -------------------------------------------------------------- boot --- */

  useEffect(() => {
    let live = true;
    Promise.all([loadCatalog(), loadIndex()])
      .then(([nextCatalog, nextIndex]) => {
        if (!live) return;
        const latest = Math.max(...nextCatalog.catalog.seasons);
        setCatalog(nextCatalog);
        setIndex(nextIndex);
        setSeason(latest);
        setPlayerSeason(latest);
        setBusy(false);
      })
      .catch(() => {
        if (!live) return;
        setError("Snapshot unavailable.");
        setBusy(false);
      });
    return () => {
      live = false;
    };
  }, []);

  /* ------------------------------------------------------------ routing -- */

  useEffect(() => {
    const apply = () => setRoute(readHash());
    // Deferred so the first read lands after mount rather than during it.
    const initial = window.setTimeout(apply, 0);
    window.addEventListener("hashchange", apply);
    return () => {
      window.clearTimeout(initial);
      window.removeEventListener("hashchange", apply);
    };
  }, []);

  const navigate = (tab: Tab, playerId?: number) => {
    const next = playerId ? `${tab}/${playerId}` : tab;
    if (window.location.hash.slice(1) === next) setRoute(readHash());
    else window.location.hash = next;
  };

  const openPlayer = (id: number) => {
    setQuery("");
    setSearchOpen(false);
    navigate("player", id);
  };

  const openComparison = async (id: number) => {
    if (!catalog) return;
    try {
      setComparePlayer(await loadPlayer(id, index, catalog.shards));
    } catch {
      setError("Comparison player unavailable.");
    }
  };

  /* -------------------------------------------------------------- data --- */

  // The ratings board and the landscape read the same season slice.
  useEffect(() => {
    if (!catalog) return;
    let live = true;
    const run = async () => {
      setBusy(true);
      try {
        const next = await loadSeason(season);
        if (live) setRows(next);
        // A blank visit opens on a real rated player. Explicit hashes keep
        // their requested view, and the player itself comes from the loaded
        // snapshot rather than a fabricated current-season value.
        if (live && !window.location.hash.slice(1)) {
          const defaultRow = [...next]
            .filter((row) => Number.isFinite(row.PLAYER_ID))
            .sort((a, b) => {
              const value = (row: LeaderboardRow) =>
                typeof row.aio_net === "number"
                  ? row.aio_net
                  : typeof row.normal_rapm_net === "number"
                    ? row.normal_rapm_net
                    : typeof row.spm_net === "number"
                      ? row.spm_net
                      : -Infinity;
              return value(b) - value(a);
            })[0];
          if (defaultRow) navigate("player", defaultRow.PLAYER_ID);
        }
      } catch {
        if (live) setError("Season unavailable.");
      } finally {
        if (live) setBusy(false);
      }
    };
    const start = window.setTimeout(() => void run(), 0);
    return () => {
      live = false;
      window.clearTimeout(start);
    };
  }, [catalog, season]);

  // The route asked for a player; fetch its shard.
  const wanted = route.playerId;
  useEffect(() => {
    if (!wanted || !catalog) return;
    let live = true;
    const run = async () => {
      setBusy(true);
      try {
        const next = await loadPlayer(wanted, index, catalog.shards);
        if (!live) return;
        setPlayer(next);
        const seasons = next.annual.map((row) => row.Season);
        setPlayerSeason(
          seasons.includes(season) ? season : (seasons.at(-1) ?? season),
        );
      } catch {
        if (live) setError("Player unavailable.");
      } finally {
        if (live) setBusy(false);
      }
    };
    const start = window.setTimeout(() => void run(), 0);
    return () => {
      live = false;
      window.clearTimeout(start);
    };
    // `season` seeds the initial player season only; re-running on it would
    // fight the reader's own selection on the player page.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wanted, catalog, index]);

  /* ------------------------------------------------------------- theme --- */

  // The current theme is read from the DOM, which a blocking script in the
  // document head has already stamped, so there is no flash and no effect.
  const toggleTheme = () => {
    const root = document.documentElement;
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      window.localStorage.setItem("impact-theme", next);
    } catch {
      /* private mode: the choice simply does not persist */
    }
  };

  /* ------------------------------------------------------------ search --- */

  const matches = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (needle.length < 2) return [];
    return index
      .filter((item) => item.name.toLocaleLowerCase().includes(needle))
      .sort((a, b) => {
        const aStarts = a.name.toLocaleLowerCase().startsWith(needle) ? 0 : 1;
        const bStarts = b.name.toLocaleLowerCase().startsWith(needle) ? 0 : 1;
        return aStarts - bStarts || a.name.localeCompare(b.name);
      })
      .slice(0, 8);
  }, [index, query]);

  useEffect(() => {
    const away = (event: MouseEvent) => {
      if (!searchRef.current?.contains(event.target as Node))
        setSearchOpen(false);
    };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, []);

  useEffect(() => {
    if (!error) return;
    const clear = window.setTimeout(() => setError(""), 4500);
    return () => window.clearTimeout(clear);
  }, [error]);

  function submitSearch(event: FormEvent) {
    event.preventDefault();
    if (matches[0]) openPlayer(matches[0].id);
  }

  const seasons = catalog?.catalog.seasons ?? [];
  const { tab } = route;

  /* -------------------------------------------------------------- view --- */

  return (
    <>
      <header className="masthead">
        <a
          className="wordmark"
          href="#home"
          aria-label="NBA Impact Lab, overview"
        >
          <b>NBA Impact</b>
          <span>Lab</span>
        </a>
        <div className="masthead-spacer" />
        <div className="search" ref={searchRef}>
          <form onSubmit={submitSearch} role="search">
            <svg
              viewBox="0 0 16 16"
              aria-hidden="true"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
            >
              <circle cx="7" cy="7" r="4.5" />
              <path d="M10.5 10.5 14 14" strokeLinecap="round" />
            </svg>
            <input
              aria-label="Find a player"
              placeholder="Find a player"
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setSearchOpen(true);
              }}
              onFocus={() => setSearchOpen(true)}
            />
          </form>
          {searchOpen && matches.length > 0 && (
            <div className="search-results">
              {matches.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  onClick={() => openPlayer(item.id)}
                >
                  {item.name}
                  <em>#{item.id}</em>
                </button>
              ))}
            </div>
          )}
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label="Toggle light and dark theme"
          title="Toggle theme"
        >
          <svg
            width="15"
            height="15"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            aria-hidden="true"
          >
            <circle cx="8" cy="8" r="3.4" />
            <path
              d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3.1 3.1l1.1 1.1M11.8 11.8l1.1 1.1M12.9 3.1l-1.1 1.1M4.2 11.8l-1.1 1.1"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </header>

      <nav className="tabs" aria-label="Sections">
        {TABS.map((item) => (
          <a
            key={item.id}
            href={`#${item.id}`}
            className={tab === item.id ? "active" : ""}
            aria-current={tab === item.id ? "page" : undefined}
          >
            {item.label}
          </a>
        ))}
      </nav>

      <main className={busy ? "content stale" : "content"}>
        {!catalog ? (
          <div className="empty">{error || "Loading snapshot…"}</div>
        ) : (
          <>
            {tab === "home" && (
              <HomeView
                catalog={catalog}
                rows={rows}
                onGo={(next) => navigate(next)}
                onPlayer={openPlayer}
              />
            )}
            {tab === "ratings" && (
              <RatingsView
                catalog={catalog}
                rows={rows}
                season={season}
                onSeason={setSeason}
                model={model}
                onModel={setModel}
                minPoss={minPoss}
                onMinPoss={setMinPoss}
                onPlayer={openPlayer}
                selected={player?.PLAYER_ID}
              />
            )}
            {tab === "landscape" && (
              <LandscapeView
                rows={rows}
                seasons={seasons}
                season={season}
                onSeason={setSeason}
                model={model}
                onModel={setModel}
                minPoss={minPoss}
                onMinPoss={setMinPoss}
                onPlayer={openPlayer}
                selected={player?.PLAYER_ID}
              />
            )}
            {tab === "player" && (
              <PlayerView
                catalog={catalog}
                player={player}
                season={playerSeason}
                onSeason={setPlayerSeason}
                model={model}
                onModel={setModel}
                component={component}
                onComponent={setComponent}
                peers={rows}
                peerSeason={season}
                index={index}
                comparePlayer={comparePlayer}
                onCompare={openComparison}
              />
            )}
            {tab === "roles" && (
              <RolesView
                catalog={catalog}
                side={roleSide}
                onSide={setRoleSide}
                player={player}
                onPlayer={openPlayer}
              />
            )}
            {tab === "projections" && <ProjectionsView onPlayer={openPlayer} />}
            {tab === "research" && <ResearchView catalog={catalog} />}
          </>
        )}
      </main>

      <footer className="site-footer">
        <p>Retrospective NBA impact research · points per 100 possessions.</p>
      </footer>

      {error && (
        <p className="toast" role="status">
          {error}
        </p>
      )}
    </>
  );
}
