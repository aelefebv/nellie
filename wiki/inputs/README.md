---
created: 2026-05-06
modified: 2026-05-06
---

# inputs/

Read-only source material that gets **compiled** into wiki articles via the COMPILE pass.

Drop in things like:

- RFCs, design docs
- PR descriptions worth preserving
- Meeting notes, transcripts
- External references that informed a decision

**Never modify files in this directory.** They are the raw record. The COMPILE pass reads from here and produces (or updates) articles elsewhere in the wiki, leaving these files untouched. If you need to revise content, edit the resulting article — not the input.
