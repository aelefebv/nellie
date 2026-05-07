---
created: 2026-05-06
modified: 2026-05-06
---

# Build & release

How nellie is versioned, packaged for PyPI, deployed as a napari plugin, and how its docs site is built.

## Versioning

Version is dynamic (`dynamic = ["version"]` in `pyproject.toml`), derived by `setuptools-scm` from the latest `vX.Y.Z` git tag at build time. A `fallback_version = "1.0.3"` in `[tool.setuptools_scm]` covers source checkouts without git history (e.g. a downloaded tarball with no `.git/`).

`verify_version.py` is a developer helper — **not** part of build or CI. It prints the locally-installed `nellie` version (via `importlib.metadata`) alongside the latest version on PyPI, intended for manual sanity-checks before/after a release.

## Multi-platform lock

`[tool.uv].required-environments` enumerates four targets: macOS arm64, macOS x86_64, Linux x86_64, win32 x86_64. By default `uv lock` resolves only for the current platform, so a lockfile generated on (say) macOS arm64 can omit wheels needed on Windows. Pinning `required-environments` forces uv to resolve a single lockfile that satisfies all four platforms simultaneously — fixing the cross-platform reproducibility gap that prompted commit `22cf8d1` ("uv: pin required-environments for cross-platform lockfile").

## Release flow

Trigger: a pushed tag matching `v*` (see `.github/workflows/release.yml`). Single `release` job on `ubuntu-latest`:

1. Checkout with `fetch-depth: 0` so `setuptools-scm` can read tags.
2. Set up Python 3.11.
3. `python -m build` produces sdist + wheel.
4. `pypa/gh-action-pypi-publish` uploads to PyPI via **OIDC Trusted Publishing** (no API token; uses `id-token: write`).
5. `softprops/action-gh-release` creates the GitHub Release with auto-generated notes and attaches the dist artifacts.

## Docs

`.github/workflows/docs.yml` builds and deploys docs on every push to `main` (or manual `workflow_dispatch`). Installs `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`, runs `mkdocs build --strict` (fails on warnings — broken links/refs block deploy), then `peaceiris/actions-gh-pages` publishes `./site` to GitHub Pages. `mkdocs.yml` uses the Material theme and `mkdocstrings` (NumPy docstring style) to auto-generate the API Reference (`api/run.md`, `api/segmentation.md`, etc.) from source. The same docs deps are exposed as `[project.optional-dependencies].docs` for local builds.

**This is a separate site from this wiki.** The published site is API-reference-style (rendered docstrings); the wiki is intent and connective tissue.

## Plugin manifest

`[project.entry-points."napari.manifest"] nellie = "nellie_napari:napari.yaml"` registers nellie with napari's plugin discovery (see [[napari-plugin/index|napari plugin]]). Combined with `[tool.setuptools.package-data] nellie_napari = ["napari.yaml", "logo.png"]`, the manifest and logo are guaranteed to ship inside the wheel — without that, `pip install nellie` would expose the entry point but napari would fail to resolve `napari.yaml` at import time.

## Gotchas

- **`dynamic = ["version"]` means *never* hand-edit a version string** — the only source of truth is the git tag. Re-tagging a commit that's already published won't republish; PyPI Trusted Publishing rejects duplicate versions.
- **`fallback_version` silently lies if you build from a `.git`-less checkout** — the wheel will claim `1.0.3` regardless of actual contents. Bump it when cutting a release if you care about tarball builds.
- **`package-data` is `nellie_napari`-only.** If a future subpackage ships data, it needs its own entry.
- **`release.yml` requires `fetch-depth: 0`.** A shallow clone breaks `setuptools-scm`.
- **`main.py` at the repo root is a tiny launcher** (opens a `napari.Viewer`, calls `add_nellie_plugins_to_menu`, `napari.run()`) — a developer convenience, **not** a packaged entry point. The shipped UX is the napari plugin manifest, not this script.

## Invariants

- Tags must match `v*` (workflow filter) and the `vX.Y.Z` SemVer shape (per CHANGELOG and setuptools-scm expectations) to publish.
- PyPI publish uses Trusted Publishing — the project must be configured on PyPI to trust this repo's `release` workflow on `ubuntu-latest`.
- Docs deploy only from `main`; non-main pushes do not update the site.
- `mkdocs build --strict` — docstring/nav errors fail the docs deploy.
