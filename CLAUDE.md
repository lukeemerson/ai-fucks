# CLAUDE.md — Working agreement for this repo

Read this first. It captures how we build here: the engineering values, the workflow, the discipline. If anything here conflicts with a freshly-typed instruction, follow the freshly-typed instruction and update this file.

---

## What we're building

A publication-grade chest X-ray ML pipeline. The end goal is a peer-reviewed paper at a free venue (arXiv → MIDL workshop → ML4H → TMLR). The work is staged through `PAPER_CHECKLIST.md`. Architectural source of truth is `harness/docs/ARCHITECTURE.md` (§13 documents the v1 implemented surface, intentional spec drift and all).

```
analyzer/             original rule-based + GBT pipeline (legacy, ships as-is)
harness/              publication-grade harness (ports & adapters)
  domain/             pure types, stdlib + numpy only
  ports/              @runtime_checkable Protocols
  adapters/
    fakes/            in-memory fakes for fast tests
    sklearn/          real sklearn-backed adapters
    fs/               filesystem adapters (artifact store, NIH dataset)
    torch/            scaffold for the v1.1 backbone adapter
  composition/        run_experiment + factories (the only layer that wires both ports + adapters)
  docs/               ARCHITECTURE.md, NIH_DATASET_SPEC.md, future spec docs
tests/
  harness/
    unit/             per-adapter algorithmic correctness, fast
    contract/         one Protocol contract suite per port, run against every adapter
    integration/      ExperimentRunner end-to-end + smoke against real data
    fixtures/         small synthetic datasets checked into git
  test_*.py           legacy analyzer tests
PAPER_CHECKLIST.md    tracking doc, $0 budget, free venues only
```

---

## Core engineering values (non-negotiable)

1. **Hexagonal discipline.** Domain → Ports → Adapters → Composition. No reverse arrows, no cross-adapter imports, no leaks.
2. **TDD red-green-refactor.** Failing test first, watch it fail, then implement. No tests written after the fact.
3. **Strict types and lint.** `mypy --strict` clean. `ruff` with the project's full ruleset clean. Zero `Any`. Every `# type: ignore` carries a justification comment on the same line.
4. **Determinism.** Every randomness flow has an explicit seed. No global `np.random.seed()`. No wall-clock dependence in production code (use injected clocks; default to epoch).
5. **No silent failures.** Don't widen confidence intervals to satisfy invariants. Don't catch-and-swallow. Errors raise `HarnessError` subclasses; everything else is a bug.
6. **YAGNI.** Don't build the v1.1 surface until v1 ships. Don't add fields "we might need." Document deferred work in ARCHITECTURE.md §13 instead.
7. **Honest reporting.** Mark VERIFIED / REFUTED / UNVERIFIABLE. Surface tradeoffs in PR descriptions. If thresholds aren't the bottleneck, say so even when threshold work was asked for.
8. **$0 budget.** No paid services, no APC journals, no commercial IRB letters, no GPU rentals. Use Apple Silicon MPS, Kaggle/Colab/Lightning free tiers, arXiv + workshop venues + TMLR. Documented in `PAPER_CHECKLIST.md`.

---

## How work gets delegated

The user manages. Claude delegates to subagents. The user's literal phrasing: **"subagent, subagent, subagent."**

```
USER (manager) ──► Claude (lead) ──► subagents (do the work)
```

- **Don't do the work yourself** when a subagent can do it. Reading status counts as managerial oversight, not work.
- **Pick the right agent type.** `backend-architect` for design + implementation, `code-reviewer` for adversarial passes, `cleanup-auditor` for lint/type/test sweeps, `test-writer-fixer` for test changes, `general-purpose` for research / fixture builds.
- **Run independent agents in parallel** in a single message. Sequential only when there's a real dependency.
- **Brief agents like a smart colleague who just walked into the room.** They've never seen this conversation. List exact files to read, exact files they own, exact files they CANNOT touch. Specify TDD red-green ordering. Cite ARCHITECTURE.md sections.
- **Recovery from agent failures.** Stream timeouts happen. Don't restart from scratch — read the partial filesystem state and brief a recovery agent on what's already on disk.

---

## Waves

Multi-step work runs in explicit waves. Format:

```
Wave 1   spec / scaffold              → 1-2 agents
Wave 2   parallel TDD implementation  → 2-4 agents (one per concern)
Wave 3   composition / wiring         → 1 agent
Wave 4   adversarial review + audit   → 2 agents in parallel
Wave 5   apply fixes                  → 1-3 agents
Wave 6   final adversarial verify     → 1 agent (with fix authority)
```

**"If you think you are done, you are not. Use more sub-agents."** — User, this session. The final verify wave is mandatory. It re-runs every gate, spot-checks that prior fixes actually shipped (don't trust subagent reports — verify the code), and either signs off `READY-TO-SHIP` or escalates with concrete next-agent scope.

For one-PR features, 6-10 agents is normal. For the harness build it took 20. Track waves in `TaskCreate` so progress is visible.

---

## TDD red-green discipline

Every TDD-spec'd agent must:

1. Read the relevant Protocol / spec / port file.
2. Write the failing test FIRST.
3. Run pytest, capture the RED (collection error / assertion failure / both).
4. Implement the production code.
5. Run pytest, capture GREEN.
6. Run `ruff check` and `mypy --strict`. Both clean.
7. Report RED→GREEN evidence in the final summary (one line each is fine).

Tests must assert **behavior**, not implementation. Banned patterns:

- `assert isinstance(x, X)` as the only assertion (mypy already enforces this statically).
- Constructing an object and asserting nothing else.
- Asserting on internal `_private` attributes.
- Tests with no assertion.

Wave 5 review explicitly hunts these and deletes them. We've removed nine such tautological tests this session.

---

## Architecture rules

```
domain/          imports: stdlib + numpy ONLY (numpy is the documented exception)
ports/           imports: domain + typing
adapters/fakes/  imports: domain + ports + numpy (no sklearn, no torch, no PIL)
adapters/sklearn imports: domain + ports + numpy + sklearn (no torch, no fakes, no fs)
adapters/fs/     imports: domain + ports + stdlib + Pillow + numpy (no sklearn, no torch)
adapters/torch/  imports: domain + ports + numpy + torch + torchvision (no sklearn)
composition/     imports: ports + adapters by class name (the ONLY layer that bridges)
```

Per-adapter file rules within `adapters/fs/`:
- `nih_csv.py` does not import `PIL` or `numpy`. Pure CSV/strings.
- `nih_images.py` does not import `csv`. Pure image work.
- `nih_dataset.py` composes both; never bypasses them with its own CSV or image logic.

When implementing a new port: write the Protocol in `ports/`, write a fake adapter in `adapters/fakes/`, write the contract test base class in `tests/harness/contract/`, write the real adapter (and its unit test + contract subclass). In that order.

When implementing a new adapter for an existing port: read the port file FIRST, copy the signature verbatim, append a new contract subclass to the existing test file. Don't touch other adapters' contract subclasses.

---

## Type strictness

```
mypy.python_version = "3.12"
mypy.strict = true
mypy.warn_unused_ignores = true
```

- No `Any` in production code. Period.
- Narrowing via `cast()` only when structural typing genuinely can't express the constraint. Prefer real type-narrowing.
- `# type: ignore[error-code]  # reason: <one sentence>` if absolutely necessary.
- Domain dataclasses are `@dataclass(frozen=True, slots=True)`.
- Use `tuple[...]` not `list[...]` for collection fields on frozen dataclasses (immutability).
- Use `numpy.typing.NDArray[np.float32]` etc. Never bare `np.ndarray`.

Lint:

```
ruff.target-version = "py312"
ruff.line-length = 110
ruff.lint.select = ["E", "F", "W", "I", "B", "UP", "SIM", "C4", "PIE", "RET", "ARG", "PTH"]
```

`PTH` rules mean `pathlib.Path` everywhere, never `os.path`. `ARG` flags unused arguments; ports are exempt because Protocol method bodies are stubs.

---

## Determinism

- Every `numpy.random.Generator` is constructed from an explicit seed.
- `RandomnessPort.child_seed(parent, label)` derives sub-seeds deterministically — use it whenever a subsystem needs randomness.
- No `np.random.seed(...)` or `random.seed(...)` global mutation. Anywhere. (We caught and fixed exactly this in Wave 6.)
- `created_at` defaults to epoch (1970-01-01) when no clock is injected. Production callers can pass a real clock; tests don't.
- Same seed + same data → byte-identical output. Reproducibility tests assert this with `np.testing.assert_array_equal` for arrays and exact `==` for floats.

---

## Test taxonomy and markers

```
tests/harness/unit/            per-adapter algorithmic correctness, < 100ms each
tests/harness/contract/        Protocol-level contract base classes; subclassed per adapter
tests/harness/integration/     ExperimentRunner end-to-end with fakes; reproducibility
tests/harness/fixtures/        small synthetic datasets (e.g. nih/ with 16 rows + 8x8 PNGs)
```

Markers (registered in `pyproject.toml` `[tool.pytest.ini_options].markers`):

- `@pytest.mark.smoke` — tests against the real on-disk data subset (NIH CSV + ~5k PNGs). Slower, machine-dependent. Excluded from default run.
- `@pytest.mark.slow` — tests > 5 seconds (e.g. fitting CalibratedClassifierCV across 8 findings). Excluded from default run. Opt in with `pytest -m slow`.

Default `addopts` excludes both: `-m 'not smoke and not slow'`. The default run must finish in < 5 seconds. We dropped it from 125s to 1.5s by tagging one slow test.

When adding a test, ask: does this need real data? → smoke. Does this take > 5s? → slow. Otherwise → default.

---

## Quality gates (must pass before opening a PR)

```
.venv/bin/pytest tests/                          # default suite, fast
.venv/bin/pytest tests/harness/ -m smoke         # against real data on disk
.venv/bin/pytest tests/ -m slow                  # opt-in slow tests
.venv/bin/ruff check harness/ tests/harness/
.venv/bin/mypy harness/
.venv/bin/mypy tests/harness/
```

All clean. No skipped tests other than the markers above. Use `--durations=10` to surface anything new and slow.

---

## PR-per-feature workflow

Feature branches: `pate/<feature-name>`. Direct push to `main` is BLOCKED — use a PR every time.

PR description template (drop sections that don't apply):

```
## Summary
- one-bullet what changed
- one-bullet why

## Quality gates
| pytest tests/ default          | N passed in Xs |
| pytest tests/harness/          | N passed       |
| pytest -m smoke                | N passed       |
| ruff check                     | clean          |
| mypy --strict harness/         | clean          |

## Built by
N subagents across M waves under TDD discipline.
- Wave 1: ...
- Wave N: ...

## Test plan
- [ ] reproducible install
- [ ] full default suite passes
- [ ] smoke passes against local data (if applicable)
- [ ] lint + type clean

## Out of scope
Explicitly note what we didn't do and why.
```

Commit messages:
- `feat(scope): one-line summary`
- `fix(scope): ...`
- `docs(scope): ...`
- Body explains the WHY, not just the what
- Ends with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

Don't push uncommitted unrelated changes. If the working tree has work that isn't this feature, commit it on its own branch or stash it. (We do this — analyzer/dashboard.html and server.py changes were specifically reviewed before being added to PR #4.)

---

## Adversarial review (Wave 4-style)

Every implementation wave is reviewed by a `code-reviewer` subagent (independent of the implementer). The review hunts:

- **Hexagonal leaks.** Wrong-layer imports.
- **Spec drift.** Where does the implementation disagree with `ARCHITECTURE.md` / `*_SPEC.md`? Either fix the code or document the deviation in §13.
- **Tautological tests.** Behavior or nothing.
- **Reviewer-bait for the paper.** Calibrator + Threshold co-fit on the same val fold? Note it. Bootstrap CI silently widened? Reject.
- **Determinism leaks.** Global RNG mutation. Wall-clock dependence. Unseeded randomness.
- **Type/lint debt.** Any `Any`. Any bare `# type: ignore`.

Output format: CRITICAL / MAJOR / MINOR / PASS, with `file:line` for every issue.

The fix wave that follows must:
- Address every CRITICAL.
- Address every MAJOR unless explicitly deferred to a follow-up issue.
- MINOR is judgment-call.

The final-verify agent re-reads the cited files to confirm fixes actually landed. **Don't trust agent self-reports** — verify in the source.

---

## Doc files and ownership

```
README.md                          high-level project README, public-facing
CLAUDE.md                          this file — working agreement for Claude
PAPER_CHECKLIST.md                 paper path tracker, $0 budget, append-only decisions log
harness/docs/ARCHITECTURE.md       harness architecture, source of truth for ports/types
harness/adapters/fs/docs/NIH_*.md  per-adapter spec docs (one per major adapter)
```

When the implementation deviates from a spec, document the deviation in the spec's `§v1 Implemented Surface` section (or equivalent). Don't silently let the spec rot.

When making a non-obvious decision (e.g. lenient vs strict whitespace handling, blank-line handling, missing-image policy), document it inline in the production code's docstring AND in the agent's final report so the next reviewer can audit it.

---

## Reading list for new sessions

Before starting any non-trivial work, read in order:

1. `CLAUDE.md` (this file)
2. `PAPER_CHECKLIST.md` (where we are on the path)
3. `harness/docs/ARCHITECTURE.md` (the harness contract)
4. The relevant adapter or port file you're touching

If a spec exists for the area you're working in (e.g. `NIH_DATASET_SPEC.md`), read that too. Don't redesign what the spec already specifies.

---

## Communication conventions

- **ASCII diagrams over emojis.** The user has explicitly preferred ASCII boxes / arrows / progress bars throughout this session. Use them for status snapshots, architecture overviews, and progress updates.
- **No emojis** unless the user asks for them.
- **Terse over verbose.** Short bullets, scannable tables, no narration of internal deliberation.
- **One sentence updates** during work, not paragraphs.
- **End-of-turn summary**: 1-2 sentences. What changed, what's next.
- **Honest tradeoffs in PR descriptions.** Don't oversell. Note what's deferred and why.
- **`/schedule` offers** when work has a natural follow-up (rare in this project so far).

---

## Out-of-scope discipline

If a task seems to balloon (one bug fix turning into a refactor, one feature pulling in a dependency upgrade), STOP and surface it. Do the originally-scoped work; open an issue or follow-up branch for the rest. The harness build was 20 agents because the scope was 20 agents; no individual wave grew its own scope mid-flight.

When unsure: ask. The user reads the diffs and will redirect — don't synthesize from ambiguity.

---

## Things we explicitly are NOT doing in v1

(These are in `ARCHITECTURE.md` §13 and `PAPER_CHECKLIST.md` "Out of Scope.")

- MIMIC-CXR / CheXpert datasets (commercial restrictions, credentialing friction)
- Real torch backbone adapter (Step 2 of the checklist — separate PR)
- DICOM support
- Multi-process data loading
- Training the backbone (only fine-tune-free embedding-based pipelines for v1)
- LLM-generated radiology reports
- Real-time / clinical deployment
- Paid services of any kind
- IRB exemption letter ($300) — using self-attestation per Common Rule §46.104(d)(4)
- APC journals (JMIR AI / Sci Reports / PLOS One)
- Clinical journals (Radiology AI / Lancet Digit Health / npj Digital Medicine — empirically gated by clinician co-authorship)

---

## When this file is wrong

Update it. Then commit. The same PR-per-feature rules apply — even to this file.
