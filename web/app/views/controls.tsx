"use client";

import {
  COMPONENTS,
  COMPONENT_LABEL,
  Component,
  ModelId,
  RatingRow,
  availableModels,
  resolveModel,
} from "../lib/data";

/** Model choice, with any model the snapshot cannot show disabled rather than hidden. */
export function ModelField({
  rows,
  value,
  onChange,
}: {
  rows: RatingRow[];
  value: ModelId;
  onChange: (id: ModelId) => void;
}) {
  const models = availableModels(rows);
  const active = resolveModel(rows, value);
  return (
    <label className="field">
      <span>Model</span>
      <select
        value={active.id}
        onChange={(event) => onChange(event.target.value as ModelId)}
      >
        {models.map((item) => (
          <option key={item.id} value={item.id} disabled={!item.available}>
            {item.available ? item.label : `${item.label} · not in snapshot`}
          </option>
        ))}
      </select>
    </label>
  );
}

export function SeasonField({
  seasons,
  value,
  onChange,
}: {
  seasons: number[];
  value: number;
  onChange: (season: number) => void;
}) {
  return (
    <label className="field">
      <span>Season</span>
      <select
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {[...seasons].reverse().map((season) => (
          <option key={season} value={season}>
            {season - 1}–{String(season).slice(2)}
          </option>
        ))}
      </select>
    </label>
  );
}

export function MinPossField({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span>Min poss</span>
      <select
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        <option value={0}>Any</option>
        <option value={100}>100+</option>
        <option value={500}>500+</option>
        <option value={1000}>1,000+</option>
        <option value={2000}>2,000+</option>
        <option value={3000}>3,000+</option>
      </select>
    </label>
  );
}

export function ComponentToggle({
  value,
  onChange,
  label = "Side",
}: {
  value: Component;
  onChange: (component: Component) => void;
  label?: string;
}) {
  return (
    <div className="field">
      <span>{label}</span>
      <div className="segmented" role="group" aria-label={label}>
        {COMPONENTS.map((item) => (
          <button
            key={item}
            type="button"
            aria-pressed={value === item}
            onClick={() => onChange(item)}
          >
            {COMPONENT_LABEL[item]}
          </button>
        ))}
      </div>
    </div>
  );
}

export function TeamField({
  teams,
  value,
  onChange,
}: {
  teams: string[];
  value: string;
  onChange: (team: string) => void;
}) {
  return (
    <label className="field">
      <span>Team</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="All">All teams</option>
        {teams.map((team) => (
          <option key={team} value={team}>
            {team}
          </option>
        ))}
      </select>
    </label>
  );
}
