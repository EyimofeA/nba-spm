# CourtSignal Codex control plane

Open this `New SPM` directory as the project root so Codex loads these agents
and hooks.

## Long work

Start one bounded milestone with `/goal`. Define:

- outcome: the artifact or behavior that must exist;
- constraints: frozen data, model, source, and publication boundaries;
- verification: commands and observable evidence that prove completion.

Use a side chat for explanations. Edit or pause the goal when the research
question changes.

## Parallel work

The project provides `data_auditor`, `stat_reviewer`, `model_worker`, and
`ui_qa`. Give every agent that writes files its own worktree. Read-only agents
may share a checkout. Use Handoff before integrating a finished worktree.

## Review

Run `/review` against the intended branch or uncommitted diff. Use detached
review when implementation work should continue in the main chat. The root
`AGENTS.md` defines the statistical and release invariants for review.

## Hooks

The session-start hook reports branch and dirty-tree state. The stop hook runs
only after a completion claim and checks whitespace, repository boundaries,
and the research control plane. Review and trust the project hooks with
`/hooks` after they change.

## Disposable analysis

Use `@Visualize` for temporary distributions, residuals, sensitivity curves,
and model comparisons. Add UI code only when a result belongs in the product.

## Cursor

Use **Settings > Import** to import selected Cursor instructions, recent chats,
skills, or agents. Review Cursor hooks and permissions before enabling them.
Do not import duplicate CourtSignal agents or hooks from Cursor.
