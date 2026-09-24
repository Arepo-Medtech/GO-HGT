# GO-HGT: Graph-Ontology → Heterogeneous Graph Transformer

Track **T2** of the Graph-Ontology next phases. An HGT trained on a leak-free ML view of the graph, feeding better ranking, better coverage (proposals that must be stamped and hand-checked) and stronger evidence into the harness.

## Start here
1. **[T0.md](T0.md)**: the shared foundations, identical in all four track repos (GO-Harness, GO-HGT, GO-PJI, GO-TS):
   frozen snapshots, the licence matrix, algebraic properties on predicates, the validation register, shared views,
   the SSSOM export and the quiz suite. This track consumes a T0 release; it never reads the live graph.
2. The full recipe for this track is in `COMPENDIUM.md` (track T2), in
   [Arepo-Medtech/graph-ontology-compendium](https://github.com/Arepo-Medtech/graph-ontology-compendium).
3. Method (rules R1–R17, validation layers L0–L6) is governed by `PLAYBOOK.md` in
   [Arepo-Medtech/one-shot](https://github.com/Arepo-Medtech/one-shot).

## How this track uses T0
- exports its ML view (integer ids only) from the snapshot, trained on `edge_displayable` by default, filtered by the register's tiers;
- merges nodes by `equivalence_group`, excluding conflicting groups;
- takes held-out links from the quiz suite, and writes ablation Δ back into the register;
- needs `process_cloud_private` in the licence matrix for any source in a view sent to a rented GPU.

## Pinning a release
Copy `t0.lock.example` to `t0.lock` once the first T0 release exists, and fill in its id and fingerprint (T0 §9).

Status: plan only. Nothing here has been built or run.
