---
created: 2026-05-06
modified: 2026-05-06
---

# Nellie wiki — index

Entry point into the wiki. Articles document the *why* and *connective tissue* of the codebase; code itself is the source of truth for *what*.

## Start here

- [[CLAUDE]] — navigation guide for agents
- [[now]] — what's live, in-flight, or recently shifted
- [[queue]] — open architectural questions
- [[glossary]] — term lookup

## Pipeline

- [[pipeline]] — orchestration (`nellie/run.py`)
- [[im-info]] — image metadata + on-disk contract
- [[segmentation/index|Segmentation]] — Frangi → labels → skeleton + markers
- [[tracking/index|Tracking]] — radius-adaptive matching → flow → relabel
- [[feature-extraction]] — voxel → node → branch → organelle → image hierarchy

## Runtime

- [[gpu-runtime]] — CPU/GPU dispatch + adaptive memory cascade

## Napari plugin

- [[napari-plugin/index|Napari plugin]] — multi-tab dock widget; entry to the per-widget articles

## Build & ops

- [[build-release]] — versioning, multi-platform lock, release flow

## Decisions

- [[decisions/index|Architecture decision records]] — hard-to-reverse choices and their rationale
