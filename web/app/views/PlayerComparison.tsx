import {
  Catalog,
  COMPONENT_LABEL,
  Component,
  ModelId,
  Player,
  Role,
  RoleSide,
  rating,
  resolveModel,
} from "../lib/data";
import { SERIES, fmtRating } from "../lib/viz";

export function ComparisonCards({ left, right, model, season }: {
  left: Player; right: Player; model: ModelId; season: number;
}) {
  const players = [
    { player: left, active: resolveModel(left.annual, model), row: left.annual.find((row) => row.Season === season) },
    { player: right, active: resolveModel(right.annual, model), row: right.annual.find((row) => row.Season === season) },
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

export function RoleComparisonPlayer({ name, roles, roleSeason, catalog }: {
  name: string;
  roles?: Player["roles"][number];
  roleSeason?: number;
  catalog: Catalog;
}) {
  if (!roles) return null;
  return <section className="role-comparison-player">
    <h2>{name} · {roleSeason ? `${roleSeason - 1}–${String(roleSeason).slice(2)} ` : ""}role mix</h2>
    <RoleMix title="Offense" role={roles.offense} side="offense" catalog={catalog} />
    <RoleMix title="Defense" role={roles.defense} side="defense" catalog={catalog} />
  </section>;
}

function RoleMix({ title, role, side, catalog }: {
  title: string;
  role?: Role;
  side: RoleSide;
  catalog: Catalog;
}) {
  const order = Object.keys(catalog.catalog.role_labels?.[side] ?? {});
  if (!role) return <div><h3>{title}</h3><p className="note">No role data for this season.</p></div>;
  const colorFor = (roleId: string) => {
    const slot = order.indexOf(roleId);
    return SERIES[(slot < 0 ? 0 : slot) % SERIES.length];
  };
  return <div>
    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10 }}>
      <h3>{title}</h3>
      <span className="meta" style={{ fontSize: 11 }}>{role.primary_role}</span>
    </div>
    <div className="mix" style={{ marginTop: 10 }}>
      {role.memberships.map((membership) => (
        <div className="mix-row" key={membership.role_id}>
          <span className="label"><i style={{ background: colorFor(membership.role_id) }} />{membership.label}</span>
          <b>{Math.round(membership.affinity * 100)}%</b>
          <span className="track"><i style={{
            width: `${Math.max(1, membership.affinity * 100)}%`,
            background: colorFor(membership.role_id),
          }} /></span>
        </div>
      ))}
    </div>
  </div>;
}
