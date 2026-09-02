# Code quality

The repository carries research history and several large legacy files. A hard
whole-repository size gate would fail today and reward arbitrary file splits.
New and edited code must improve the baseline instead.

## Automated gate

Every pull request checks changed source files.

- New source files may not exceed 500 lines.
- A legacy file above 500 lines may not grow.
- New or worsened Python functions may not exceed cyclomatic complexity 22.
- New explicit `Any` types fail. Use a concrete type. Use `unknown` only at an
  external boundary and narrow it before use.
- The web client must pass lint, build, and every test that does not require an
  ignored localhost research payload.

The gate applies to production, tests, research runners, and the web client. It
does not scan generated data, artifacts, archives, or local research output.

## Review policy

Review the diff before the metric. Delete duplicated branches, helpers, flags,
and options when one existing path already handles the case. Do not split a
cohesive file only to lower a line count. Require focused tests for changed
behavior.

Use 100% branch coverage and zero surviving mutants for small critical kernels,
such as score conservation, lineup identity, sign conventions, and release
contracts. Applying those thresholds to ingestion adapters and research runners
would add brittle tests without proving correctness.

Cognitive complexity, Halstead difficulty, and CRAP are review signals, not
release gates here. They overlap with size, branch complexity, and coverage,
and they are easy to lower without making the code easier to change.
