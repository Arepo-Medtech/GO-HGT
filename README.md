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
`t0.lock` pins Release 8, `GO-20261120-092e71ea` (20 Nov 2026), fingerprint `092e71ea67d0f5e6f595ab51006b8fee3d7f33824b067bc7bb1c6b75ab84bde4`: the nodes and edges of Release 7; what changed is which links form an equivalence group (no ungraded link forms one, DrugCentral to SNOMED CT is split into a checked same-substance route, an Orphanet exact link to OMIM forms none where the entry is the whole disease and the OMIM entry a subtype, and 71 single links the owner read as not the same are rejected); every refusing gate passed, and the reported per-track licence gate L-2 still lists 29 unreviewed rows (RELEASE.md in the release directory). Previous pin: Release 7, `GO-20261119-9c9bddf3` (19 Nov 2026); the pins before it are listed in `t0.lock`. Upgrade by bumping `t0.lock` in a pull request that carries the release diff and this track's regression results (T0 §9).

Status: plan only. Nothing here has been built or run.
