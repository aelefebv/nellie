---
created: 2026-05-06
modified: 2026-05-06
---

# Wiki navigation guide

This is the **repo wiki** for nellie. It captures the **why and how things hang together** layer of the project — rationale, history, cross-system interactions, gotchas, invariants, in-flight state.

**Code is ground truth for *what***. Read source files for behavior. Read here for *why* it's that way, *what else it touches*, and *what to watch out for*.

## How to navigate

Standard order when you arrive:

1. **CLAUDE.md** (this file) — orientation
2. **now.md** — what's live, in-flight, or recently shifted
3. **index.md** — entry point into the article tree
4. **articles** — follow `[[wiki-links]]` between them

## Wiki-link conventions

- `[[page]]` — link by basename
- `[[path/page|Display Text]]` — path-qualified when two articles share a basename, or when you want custom display text

## Frontmatter

Every article begins with a YAML block fenced by `---`, containing two fields: `created: YYYY-MM-DD` and `modified: YYYY-MM-DD`. Bump `modified` whenever you meaningfully change an article.

## Directories

- `inputs/` — read-only source material (RFCs, design docs, PR notes, transcripts) that gets compiled into articles. Never modify files in here.
- `outputs/` — standalone artifacts (one-shot reports, generated diagrams, exported docs) produced from wiki content.
- `decisions/` — architecture decision records (ADRs), if/when added.

## Drift detection

Drift detection is **on-demand** via the LINT pass. Articles can fall behind code. When you act on a wiki claim that names a specific file, function, or flag, cross-check against the current code before relying on it.
