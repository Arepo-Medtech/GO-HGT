# GO-HGT: Graph-Ontology → Heterogeneous Graph Transformer

Track **T2** of the Graph-Ontology next phases. An HGT trained on a leak-free ML view of the graph, feeding better ranking, better coverage (proposals that must be stamped and hand-checked) and stronger evidence into the harness.

## Start here
This repository is self-contained. Two documents hold everything this track needs:

1. **[PLAYBOOK.md](PLAYBOOK.md)**: the authoritative playbook for this track. It covers mission, rules, architecture,
   stages and gates, engineering guidance, evaluation, scorecard, operations, governance, risks, prompts and
   references, with the hand-check protocol and templates as appendices.
2. **[T0.md](T0.md)**: the shared foundations this track consumes (frozen snapshots, the licence matrix, predicate
   properties, the validation register, shared views, the SSSOM export and the quiz suite). It is identical across
   the four Graph-Ontology track repositories.

## How this track uses T0
- exports its ML view (integer ids only) from the snapshot, trained on `edge_displayable` by default, filtered by the register's tiers;
- merges nodes by `equivalence_group`, excluding conflicting groups;
- takes held-out links from the quiz suite, and writes ablation Δ back into the register;
- needs `process_cloud_private` in the licence matrix for any source in a view sent to a rented GPU.

## Pinning a release
`t0.lock` pins Release 7, `GO-20261119-9c9bddf3` (19 Nov 2026), fingerprint `9c9bddf31d6882cb76e5ee7ede7ac9e8b6b3f4b47058f8adec869c4991897d78`: Release 6 plus the Medicare bridge, kind policy version 3 (person state and service), HPO lay names as typed synonyms, and equivalence groups that follow checked links only after the owner's hand checks; every refusing gate passed, and the reported per-track licence gate L-2 still lists 29 unreviewed rows (RELEASE.md in the release directory). Previous pin: Release 6, `GO-20261113-b73a8a33` (13 Nov 2026); the pins before it are listed in `t0.lock`. Upgrade by bumping `t0.lock` in a pull request that carries the release diff and this track's regression results (T0 §9).

Status: plan only. Nothing here has been built or run.
