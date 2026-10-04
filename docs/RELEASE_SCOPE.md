# Release scope

This repository is a curated export of the Idea Arena implementation accompanying
Priced Guidance. It begins with fresh Git history. The original experiment and
manuscript repositories remain unchanged.

Included:

- The runtime's protocol, pricing, judging, model transports, common memory,
  participant isolation, durable recording/replay, and local inspection tools.
- The offline example, maintained one/eight-candidate reference pairs, and the
  guide prompt variant used in the development comparison.
- Frozen participants for the reported scaffold/candidate comparisons and main
  Directional-to-Essence runs.
- The exact 40-paper development and 87-paper test target cohorts, with public
  taxonomies and evaluator gold summaries; the offline smoke case.
- Current test-set compression tables, original numerical inputs, the Essence
  promotion charges, and the offline rebuild/accounting tools.
- Opus 5 summary generation, pack creation/validation, title-recall auditing,
  and tests for retained behavior.

Excluded:

- The old `idea_generation_old` tree and prior repository history.
- Diverse lexical-prefix/evidence-gated experiments, forced-audit/repair probes,
  superseded cohorts, unused intermediate artifacts, and experiment notebooks.
- Machine-specific batch launchers, quota monitors, remote migration scripts,
  operational handoffs, and the tests that only exercise those excluded tools.
- Raw historical private transcripts, native account state, credentials, source
  archives, local environments, rendered backups, and caches.
- Manuscript drafts, author review notes, and old conference templates.

The selection is by reported experimental condition, not outcome: failed targets
within retained cohorts are preserved. The compact release is an implementation
and extension starting point with main-result tables, not a complete archival
bundle of every paper figure or provider transcript.

Release adaptations rename the packaged cohorts to `development40` and `test87`,
update doctor/documentation to those cohorts, make target-pack output refuse
existing destinations, and restrict downloaded archive members and TeX includes
to the source directory. Model prompts and core information-pricing rules are
preserved. Compatibility code for optional native backends and Strict remains
because existing manifests and replay use it; it is not presented as a paper
experiment.
