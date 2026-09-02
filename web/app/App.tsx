"use client";

import {
  FormEvent,
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Catalog,
  LeaderboardRow,
  LocalPlayerSkills,
  LocalSkillIndex,
  RapmLabPayload,
  SpmLabPayload,
  ModelId,
  Player,
  PlayerIndex,
  RoleSide,
  loadCatalog,
  loadIndex,
  loadLocalPlayerSkills,
  loadLocalSkillIndex,
  loadRapmLab,
  loadSpmLab,
  loadPlayer,
  loadSeason,
} from "./lib/data";
const PlayerView = lazy(() =>
  import("./views/PlayerView").then((module) => ({ default: module.PlayerView })),
);
const RatingsView = lazy(() =>
  import("./views/RatingsView").then((module) => ({ default: module.RatingsView })),
);
const ResearchView = lazy(() =>
  import("./views/ResearchView").then((module) => ({ default: module.ResearchView })),
);
const RolesView = lazy(() =>
  import("./views/RolesView").then((module) => ({ default: module.RolesView })),
);
const PublicRapmLabView = lazy(() =>
  import("./views/PublicRapmLabView").then((module) => ({ default: module.PublicRapmLabView })),
);
const SpmLabView = lazy(() =>
  import("./views/SpmLabView").then((module) => ({ default: module.SpmLabView })),
);
const ReplicationsView = lazy(() =>
  import("./views/ReplicationsView").then((module) => ({ default: module.ReplicationsView })),
);
const ALL_TABS = [
  { id: "ratings", label: "Ratings" },
  { id: "player", label: "Player" },
  { id: "roles", label: "Roles" },
  { id: "rapm-lab", label: "RAPM Lab" },
  { id: "replications", label: "Reconstructions" },
  { id: "spm-lab", label: "SPM Lab" },
  { id: "research", label: "Research" },
] as const;

const PUBLIC_NAV_TABS = [
  { id: "ratings", label: "Ratings" },
  { id: "roles", label: "Role map" },
  { id: "rapm-lab", label: "RAPM Lab" },
  { id: "research", label: "Research" },
] as const;

const LOCAL_NAV_TABS = [
  { id: "ratings", label: "Ratings" },
  { id: "roles", label: "Role map" },
  { id: "rapm-lab", label: "RAPM Lab" },
  { id: "replications", label: "Reconstructions" },
  { id: "research", label: "Research" },
] as const;

type Tab = (typeof ALL_TABS)[number]["id"];
const isLocalHost = () => {
  if (typeof window === "undefined") return false;
  const host = window.location.hostname.toLowerCase();
  return (
    host === "localhost" ||
    host === "127.0.0.1" ||
    host === "::1" ||
    host.endsWith(".local")
  );
};
const isTab = (value: string, showResearchLab: boolean): value is Tab => {
  const tabs = showResearchLab
    ? ALL_TABS
    : ALL_TABS.filter((tab) => !["spm-lab", "replications"].includes(tab.id));
  return tabs.some((tab) => tab.id === value);
};

const foldName = (value: string) =>
  value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase();

/** The location hash is the single source of truth for the current view. */
type Route = { tab: Tab; playerId?: number };

const readHash = (showResearchLab: boolean): Route => {
  const [head, id] = window.location.hash.slice(1).split("/");
  const parsed = Number(id);
  const routed = head;
  return {
    tab:
      routed === "landscape"
        ? "ratings"
        : isTab(routed, showResearchLab)
          ? routed
          : "ratings",
    playerId: Number.isFinite(parsed) && parsed > 0 ? parsed : undefined,
  };
};

export function App() {
  const [hasLocalResearch, setHasLocalResearch] = useState(false);

  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [index, setIndex] = useState<PlayerIndex[]>([]);
  const [route, setRoute] = useState<Route>({ tab: "ratings" });
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");

  // Filters live at the shell so every view below reads the same slice.
  const [season, setSeason] = useState(2026);
  const [model, setModel] = useState<ModelId>("pulse");
  const [minPoss, setMinPoss] = useState(100);

  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [player, setPlayer] = useState<Player | null>(null);
  const [comparePlayer, setComparePlayer] = useState<Player | null>(null);
  const [playerSeason, setPlayerSeason] = useState(2024);
  const [roleSide, setRoleSide] = useState<RoleSide>("offense");
  const [rapmLab, setRapmLab] = useState<RapmLabPayload | null>(null);
  const [spmLab, setSpmLab] = useState<SpmLabPayload | null>(null);
  const [localSkillIndex, setLocalSkillIndex] = useState<LocalSkillIndex | null>(null);
  const [localPlayerSkills, setLocalPlayerSkills] = useState<LocalPlayerSkills | null>(null);
  const [compareLocalSkills, setCompareLocalSkills] = useState<LocalPlayerSkills | null>(null);

  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const comparisonRequest = useRef(0);

  /* -------------------------------------------------------------- boot --- */

  useEffect(() => {
    const detect = window.setTimeout(
      () => setHasLocalResearch(isLocalHost()),
      0,
    );
    return () => window.clearTimeout(detect);
  }, []);

  useEffect(() => {
    let live = true;
    loadCatalog()
      .then((nextCatalog) => {
        if (!live) return;
        // Open on the most recent complete PULSE season.
        const latest = Math.max(...nextCatalog.catalog.seasons);
        setCatalog(nextCatalog);
        setSeason(latest);
        setPlayerSeason(latest);
      })
      .catch(() => {
        if (!live) return;
        setError("Snapshot unavailable.");
        setBusy(false);
      });
    loadIndex()
      .then((nextIndex) => {
        if (live) setIndex(nextIndex);
      })
      .catch(() => {
        if (live) setError("Player search unavailable.");
      });
    return () => {
      live = false;
    };
  }, []);

  /* ------------------------------------------------------------ routing -- */

  useEffect(() => {
    const apply = () => setRoute(readHash(hasLocalResearch));
    // Deferred so the first read lands after mount rather than during it.
    const initial = window.setTimeout(apply, 0);
    window.addEventListener("hashchange", apply);
    return () => {
      window.clearTimeout(initial);
      window.removeEventListener("hashchange", apply);
    };
  }, [hasLocalResearch]);

  const navigate = (tab: Tab, playerId?: number) => {
    const next = playerId ? `${tab}/${playerId}` : tab;
    if (window.location.hash.slice(1) === next) setRoute(readHash(hasLocalResearch));
    else window.location.hash = next;
  };

  const openPlayer = (id: number) => {
    setQuery("");
    setSearchOpen(false);
    navigate("player", id);
  };

  const openComparison = async (id: number) => {
    if (!catalog) return;
    const request = ++comparisonRequest.current;
    try {
      const [nextPlayer, nextSkills] = await Promise.all([
        loadPlayer(id, index, catalog.shards),
        localSkillIndex ? loadLocalPlayerSkills(id).catch(() => null) : Promise.resolve(null),
      ]);
      if (request !== comparisonRequest.current) return;
      setComparePlayer(nextPlayer);
      setCompareLocalSkills(nextSkills);
    } catch {
      if (request === comparisonRequest.current) {
        setError("Comparison player unavailable.");
      }
    }
  };

  /* -------------------------------------------------------------- data --- */

  // The ratings table and chart read the same season slice.
  useEffect(() => {
    if (!catalog) return;
    let live = true;
    const run = async () => {
      setBusy(true);
      try {
        const next = await loadSeason(season);
        if (live) setRows(next);
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

  useEffect(() => {
    if (!catalog || !hasLocalResearch || !["player", "rapm-lab", "replications"].includes(route.tab)) return;
    let live = true;
    const run = async () => {
      try {
        const nextLab = await loadRapmLab().catch(() => null);
        if (!live) return;
        setRapmLab(nextLab);
      } catch {
        if (live) {
          setRapmLab(null);
        }
      }
    };
    const start = window.setTimeout(() => void run(), 0);
    return () => {
      live = false;
      window.clearTimeout(start);
    };
  }, [catalog, hasLocalResearch, route.tab]);

  useEffect(() => {
    if (!catalog || !hasLocalResearch || route.tab !== "spm-lab") return;
    let live = true;
    loadSpmLab()
      .then((next) => {
        if (live) setSpmLab(next);
      })
      .catch(() => {
        if (live) setSpmLab(null);
      });
    return () => {
      live = false;
    };
  }, [catalog, hasLocalResearch, route.tab]);

  useEffect(() => {
    if (!hasLocalResearch || route.tab !== "player") return;
    let live = true;
    loadLocalSkillIndex()
      .then((next) => {
        if (!live) return;
        setLocalSkillIndex(next);
        if (!route.playerId) window.location.hash = `player/${next.defaultPlayerId}`;
      })
      .catch(() => {
        if (live) setLocalSkillIndex(null);
      });
    return () => {
      live = false;
    };
  }, [hasLocalResearch, route.playerId, route.tab]);

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

  useEffect(() => {
    if (!wanted || !localSkillIndex || route.tab !== "player") return;
    let live = true;
    loadLocalPlayerSkills(wanted)
      .then((next) => {
        if (live) setLocalPlayerSkills(next);
      })
      .catch(() => {
        if (live) setLocalPlayerSkills(null);
      });
    return () => {
      live = false;
    };
  }, [localSkillIndex, route.tab, wanted]);

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
    const needle = foldName(query.trim());
    if (needle.length < 2) return [];
    return index
      .filter((item) => foldName(item.name).includes(needle))
      .sort((a, b) => {
        const aStarts = foldName(a.name).startsWith(needle) ? 0 : 1;
        const bStarts = foldName(b.name).startsWith(needle) ? 0 : 1;
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

  const { tab } = route;
  const navTabs = hasLocalResearch ? LOCAL_NAV_TABS : PUBLIC_NAV_TABS;
  // Player pages are drilled into from the ratings board, so retain that
  // context in the primary navigation instead of making a dead-end tab.
  const activeNavTab = tab === "player" ? "ratings" : tab;
  const viewLabel =
    tab === "player"
      ? "Player report"
      : (navTabs.find((item) => item.id === activeNavTab)?.label ?? "Ratings");

  /* -------------------------------------------------------------- view --- */

  return (
    <div className="app-shell">
      <aside className="side-rail">
        <a className="rail-brand" href="#ratings" aria-label="CourtSignal ratings">
          <span className="brand-glyph" aria-hidden="true">CS</span>
          <span className="brand-copy">
            <b>CourtSignal</b>
            <small>NBA impact</small>
          </span>
        </a>

        <nav className="rail-nav" aria-label="Primary analysis">
          {navTabs.map((item, index) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className={activeNavTab === item.id ? "active" : ""}
              aria-current={activeNavTab === item.id ? "page" : undefined}
            >
              <span aria-hidden="true">0{index + 1}</span>
              {item.label}
            </a>
          ))}
        </nav>

        <div className="rail-spacer" />
        <div className="rail-dataset" aria-label={`NBA season ${season - 1} to ${season}`}>
          <span>Dataset</span>
          <strong>{season - 1}–{String(season).slice(2)}</strong>
          <small>{catalog ? "Loaded" : "Connecting"}</small>
        </div>
        <button
          type="button"
          className="theme-toggle rail-theme"
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
              d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4V8H15M3.1 3.1l1.1 1.1M11.8 11.8l1.1 1.1M12.9 3.1l-1.1 1.1M4.2 11.8l-1.1 1.1"
              strokeLinecap="round"
            />
          </svg>
          <span>Theme</span>
        </button>
      </aside>

      <div className="workspace">
        <header className="commandbar">
          <div className="command-context">
            <span>NBA / PLAYER IMPACT</span>
            <strong>{viewLabel}</strong>
          </div>
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
                placeholder="Search players"
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
                  </button>
                ))}
              </div>
            )}
          </div>
        </header>

        <main className={busy ? "content stale" : "content"}>
        {!catalog ? (
          <div className="empty">{error || "Loading snapshot…"}</div>
        ) : (
          <Suspense fallback={<div className="empty">Loading view…</div>}>
            {tab === "ratings" && rows.length > 0 && (
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
                index={index}
                comparePlayer={comparePlayer}
                localSkillIndex={localSkillIndex}
                localSkills={localPlayerSkills}
                compareLocalSkills={compareLocalSkills}
                rapmLab={rapmLab}
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
            {tab === "rapm-lab" && (
              <PublicRapmLabView />
            )}
            {tab === "replications" && hasLocalResearch && (
              <ReplicationsView lab={rapmLab} />
            )}
            {tab === "spm-lab" && <SpmLabView lab={spmLab} />}
            {tab === "research" && <ResearchView catalog={catalog} />}
          </Suspense>
        )}
        </main>

      </div>

      {error && (
        <p className="toast" role="status">
          {error}
        </p>
      )}
    </div>
  );
}
