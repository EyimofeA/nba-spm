"use client";

import { useState } from "react";
import { Figure, Legend } from "../charts/frame";
import { MultiLine } from "../charts/lines";
import { COMPONENT_LABEL, Catalog, Component } from "../lib/data";
import { COMPONENT_COLOR, fmtInt, fmtRating } from "../lib/viz";

const IMPORTANCE = [
  ["Offense", "Shooting / scoring / spacing", 90, 0.2719],
  ["Offense", "Public composites", 8, 0.1081],
  ["Offense", "Rebounding / screening", 6, 0.0106],
  ["Defense", "Disruption", 3, 0.1167],
  ["Defense", "Creation / passing", 24, 0.0816],
  ["Defense", "Rebounding / screening", 6, 0.0793],
] as const;

export function ResearchView({ catalog }: { catalog: Catalog }) {
  const [source, setSource] = useState<"rapm" | "aio">("rapm");
  const [view, setView] = useState<"change" | "level">("change");
  const aging = catalog.aging[source];
  const external = catalog.validation.external_benchmark;
  const externalRows = external.rows ?? [];
  const forwardRows = catalog.validation.walk_forward ?? [];
  const backwardRows = catalog.validation.walk_backward ?? [];
  const projection = catalog.validation.aging_projection;
  const projectionRows = projection.selection ?? [];
  const selectedProjection = projection.selected_method;

  // Only the RAPM curve publishes levels; AIO publishes year-over-year change.
  const effectiveView = source === "aio" ? "change" : view;
  const keyFor = (item: Component) =>
    effectiveView === "change"
      ? `change_${item}`
      : item === "net"
        ? "f_total"
        : item === "offense"
          ? "f_off"
          : "f_def";

  const agingSeries = (["net", "offense", "defense"] as Component[]).map(
    (item) => ({
      label: COMPONENT_LABEL[item],
      color: COMPONENT_COLOR[item],
      points: aging.rows.flatMap((row) => {
        const value = row[keyFor(item)];
        return typeof value === "number" ? [{ x: row.Age, y: value }] : [];
      }),
    }),
  );
  const hasSeries = agingSeries.some((series) => series.points.length > 0);

  return (
    <>
      <div className="page-head">
        <div>
          <p className="kicker">Research</p>
          <h1>What held up</h1>
        </div>
        {forwardRows.length > 0 && (
          <span className="meta">Frozen runs only · {forwardRows[0].seasons}</span>
        )}
      </div>

      <div className="grid">
        <section className="card prose-grid">
          <div>
            <p className="kicker">Build order</p>
            <h2>From SPM to AIO</h2>
            <p className="note">
              SPM predicts one season of impact from box, tracking, playtype,
              and matchup features. One ridge RAPM fit is then centered on that
              prediction. RAPM is the separate zero-prior fit — the
              reference model, not a part of AIO.
            </p>
          </div>
          <div>
            <p className="formula">{catalog.methods.aio_equation}</p>
            <div className="tag-grid">
              <span>Rated season excluded from its own SPM labels</span>
              <span>Ridge penalties fixed at 3000 / 3000 / 300</span>
              <span>Offense and defense estimated jointly; net is their sum</span>
              <span>Identity holds to floating-point precision</span>
            </div>
            <p className="note">{catalog.methods.rapm_update_note}</p>
          </div>
        </section>

        <section className="card prose-grid">
          <div>
            <p className="kicker">Next projects</p>
            <h2>What we are testing next</h2>
            <p className="note">
              Shot-quality matchup ratings will separate rim and non-rim
              opportunity from shotmaking, but stay exploratory until a
              shot-level defender assignment is available. Combination RAPM
              will test whether recurring 2–5-player groups add predictive
              value beyond additive player RAPM.
            </p>
          </div>
          <div className="tag-grid">
            <span>Shot quality: event location, no defender-at-shot claim</span>
            <span>Matchups: rim and non-rim models</span>
            <span>Groups: pair RAPM before trios, fours, or fives</span>
            <span>Promotion requires whole-game held-out improvement</span>
          </div>
        </section>

        <section className="card prose-grid">
          <div>
            <p className="kicker">Inputs</p>
            <h2>What each model reads</h2>
            <p className="note">
              RAPM reads possession points, home offense, and the ten players on
              court—no box-score features. SPM uses 127 offense and 68 defense
              columns from box, tracking, playtype, and matchup data. zTS is in
              offense. Roles were tested but were not selected. AIO uses the
              same possession design as RAPM, centered on the held-out SPM.
            </p>
          </div>
          <table className="mini">
            <thead>
              <tr>
                <th scope="col">Side</th>
                <th scope="col">Feature family</th>
                <th scope="col">Columns</th>
                <th scope="col">RMSE Δ</th>
              </tr>
            </thead>
            <tbody>
              {IMPORTANCE.map(([side, family, features, delta]) => (
                <tr key={`${side}-${family}`}>
                  <td>{side}</td>
                  <td>{family}</td>
                  <td>{features}</td>
                  <td>{delta.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        {forwardRows.length > 0 && <section className="card prose-grid">
          <div>
            <p className="kicker">Forward test</p>
            <h2>Earlier seasons only</h2>
            <p className="note">
              Each season’s SPM is trained on earlier seasons only, then scored
              against that season’s RAPM. Errors are weighted by the
              square root of the smaller possession count. The published ratings
              use leave-one-season-out centers, which may also see later
              seasons, so this stricter table is the honest accuracy check.
            </p>
          </div>
          <table className="mini">
            <thead>
              <tr>
                <th scope="col">Side</th>
                <th scope="col">Years</th>
                <th scope="col">Folds</th>
                <th scope="col">RMSE</th>
                <th scope="col">r</th>
                <th scope="col">R²</th>
              </tr>
            </thead>
            <tbody>
              {forwardRows.map((row) => (
                <tr key={row.component}>
                  <td>{COMPONENT_LABEL[row.component as Component]}</td>
                  <td>{row.seasons}</td>
                  <td>{row.folds}</td>
                  <td>{row.rmse.toFixed(3)}</td>
                  <td>{row.correlation.toFixed(3)}</td>
                  <td>{row.r2.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>}

        {backwardRows.length > 0 && <section className="card prose-grid">
          <div>
            <p className="kicker">Direction check</p>
            <h2>Next year and last year</h2>
            <p className="note">
              The frozen held-out SPM at one season is compared with RAPM
              in the next season and in the previous season, over the same
              matched transitions. Net correlation is almost equal in both
              directions, so the agreement is not explained by player
              development alone. This is a diagnostic, not forecast evidence.
            </p>
          </div>
          <table className="mini">
            <thead>
              <tr>
                <th scope="col">Direction</th>
                <th scope="col">Side</th>
                <th scope="col">Rows</th>
                <th scope="col">RMSE</th>
                <th scope="col">r</th>
              </tr>
            </thead>
            <tbody>
              {backwardRows.map((row) => (
                <tr key={`${row.direction}-${row.component}`}>
                  <td>
                    {row.direction === "forward"
                      ? "Next season"
                      : "Previous season"}
                  </td>
                  <td>{COMPONENT_LABEL[row.component as Component]}</td>
                  <td>{fmtInt(row.rows)}</td>
                  <td>{row.rmse.toFixed(3)}</td>
                  <td>{row.correlation.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>}

        {externalRows.length > 0 && <section className="card prose-grid">
          <div>
            <p className="kicker">External agreement</p>
            <h2>Against public metrics</h2>
            <p className="note">{external.note}</p>
            <p className="note">{external.pinned_model_note}</p>
          </div>
          <table className="mini">
            <thead>
              <tr>
                <th scope="col">Comparison</th>
                <th scope="col">Side</th>
                <th scope="col">Players</th>
                <th scope="col">r · BPM</th>
                <th scope="col">r · xRAPM</th>
              </tr>
            </thead>
            <tbody>
              {externalRows.map((row, index) => (
                <tr key={`${row.scope}-${row.component}-${index}`}>
                  <td>
                    {row.scope}
                    <span className="sub">{row.exposure}</span>
                  </td>
                  <td>{COMPONENT_LABEL[row.component as Component]}</td>
                  <td>{fmtInt(row.players)}</td>
                  <td>{row.bpm?.toFixed(3) ?? "—"}</td>
                  <td>{row.xrapm?.toFixed(3) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>}

        <Figure
          kicker="Aging"
          title={
            effectiveView === "change"
              ? "Observed year-over-year change by age"
              : "Observed level by age"
          }
          legend={
            <Legend
              items={agingSeries.map((s) => ({
                label: s.label,
                color: s.color,
              }))}
              shape="key"
            />
          }
          controls={
            <>
              <div className="segmented" role="group" aria-label="Aging metric">
                {(["rapm", "aio"] as const).map((item) => (
                  <button
                    key={item}
                    type="button"
                    aria-pressed={source === item}
                    onClick={() => setSource(item)}
                  >
                    {item.toUpperCase()}
                  </button>
                ))}
              </div>
              {source === "rapm" && (
                <div className="segmented" role="group" aria-label="Aging view">
                  {(["change", "level"] as const).map((item) => (
                    <button
                      key={item}
                      type="button"
                      aria-pressed={view === item}
                      onClick={() => setView(item)}
                    >
                      {item === "change" ? "Change" : "Level"}
                    </button>
                  ))}
                </div>
              )}
            </>
          }
          note={
            <>
              Coverage: {aging.coverage}. The curve is measured change for
              players who kept playing, so survivors stay in the sample — it is
              descriptive, not a causal aging effect. Sample size per age is in
              the table.
            </>
          }
          table={
            <table className="mini">
              <thead>
                <tr>
                  <th scope="col">Age</th>
                  <th scope="col">n</th>
                  <th scope="col">Net</th>
                  <th scope="col">Off</th>
                  <th scope="col">Def</th>
                </tr>
              </thead>
              <tbody>
                {aging.rows.map((row) => (
                  <tr key={row.Age}>
                    <td>{row.Age}</td>
                    <td>{fmtInt(row.n)}</td>
                    <td>{fmtRating(row[keyFor("net")] as number)}</td>
                    <td>{fmtRating(row[keyFor("offense")] as number)}</td>
                    <td>{fmtRating(row[keyFor("defense")] as number)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        >
          {hasSeries ? (
            <MultiLine
              series={agingSeries}
              xTitle="Age"
              xFormat={String}
              markEvery={2}
              height={280}
            />
          ) : (
            <div className="empty">
              This metric publishes no curve for that view.
            </div>
          )}
        </Figure>

        {selectedProjection && projectionRows.length > 0 && <section className="card prose-grid">
          <div>
            <p className="kicker">Projection method</p>
            <h2>Age helps a little</h2>
            <p className="note">
              Candidate one-year projection methods were selected on four
              earlier origin seasons and then scored on two later ones.{" "}
              {selectedProjection.replaceAll("_", " ")}{" "}
              was selected. Its gain over the plain AR(1) baseline is small.
            </p>
          </div>
          <table className="mini">
            <thead>
              <tr>
                <th scope="col">Method</th>
                <th scope="col">Selection RMSE</th>
                <th scope="col">Diagnostic RMSE</th>
              </tr>
            </thead>
            <tbody>
              {projectionRows.map((row) => (
                <tr
                  key={row.method}
                  className={
                    row.method === selectedProjection
                      ? "flag"
                      : undefined
                  }
                >
                  <td>{row.method.replaceAll("_", " ")}</td>
                  <td>{row.mean_rmse.toFixed(3)}</td>
                  <td>
                    {(projection.diagnostic ?? [])
                      .find((item) => item.method === row.method)
                      ?.mean_rmse.toFixed(3) ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>}

        <section className="card">
          <div className="card-head">
            <div>
              <p className="kicker">Limits</p>
              <h2>What is not proven</h2>
            </div>
            <span className="meta">Read before quoting a rating</span>
          </div>
          <ul className="limit-list" style={{ marginTop: 16 }}>
            <li>
              <b>Defense is the weak side.</b>
              <span>
                Held-out defensive accuracy stays well below offense in every
                test above.
              </span>
            </li>
            <li>
              <b>The frozen model missed its untouched confirmation season.</b>
              <span>
                Offense, defense, and net RMSE were 1.102, 1.154, and 1.610, and
                defense correlation fell to 0.331. The public ratings therefore
                stay pinned to 2017–24.
              </span>
            </li>
            <li>
              <b>Production RAPM keeps a zero prior.</b>
              <span>
                The matched-game test of a prior-centered fit had a paired
                interval that crossed zero. A later annual prior challenger
                looks better, but it stays research only.
              </span>
            </li>
            <li>
              <b>No published intervals.</b>
              <span>
                SPM and AIO uncertainty is not calibrated yet, so no rating here
                shows a range.
              </span>
            </li>
            <li>
              <b>Roles and skill profiles are descriptive.</b>
              <span>They are not impact estimates and not model inputs.</span>
            </li>
          </ul>
        </section>
      </div>
    </>
  );
}
