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
`t0.lock` pins Release 3, `GO-20261108-0670a01e` (8 Nov 2026), fingerprint `0670a01ee967f973e71c6aafb7e99ffc52de790b0f8fd8dfec348da426157708`: MeSH 2026 loaded as a vocabulary (every descriptor a node, the tree and pharmacological actions as native edges, entry terms in the synonym view); every refusing gate passed, and the reported per-track licence gate L-2 still lists 28 unreviewed rows (RELEASE.md in the release directory). Previous pin: Release 2, `GO-20261103-4e9f3709` (3 Nov 2026). Upgrade by bumping `t0.lock` in a pull request that carries the release diff and this track's regression results (T0 §9).

Status: plan only. Nothing here has been built or run.
