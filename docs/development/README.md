# How mhr_vector was built

mhr_vector was built by an agent_swarm run in October 2026. A host first wrote the
groundwork: the contract ([../requirements.md](../requirements.md)), the exact oracle, the
benchmark harness, the crate skeleton with `todo!` bodies, and an acceptance test for each
of 19 units. Agents then implemented the units in dependency order. A gate merged a unit
once its acceptance test passed and everything that passed on main still passed: the
other units' tests, the conformance checks against the oracle, and the oracle's and the
harness's own tests. Tidy and optimization phases followed, with recall guards and a speed
score. The commits by `vector host` come from the run. The rest are the host's: the
groundwork and f7ad141 (parallel graph builds and SIMD kernels).

The files here are the plan the run worked from:

- [units.md](units.md) and [units.json](units.json): the 19 units, their files,
  dependencies and acceptance tests.
- [swarm-pack.md](swarm-pack.md): how the run was set up, with its gates, buckets, scores
  and build machine.
- [host-owned.md](host-owned.md): the paths agents could not change.
- [signatures.txt](signatures.txt): the skeleton's public interface, frozen for the run.
  `python3 scripts/signatures.py --check docs/development/signatures.txt` checks that the
  crate still has every item.

During the run these files sat at the repository root as `REQUIREMENTS.md`, `UNITS.md`,
`units.json`, `SWARM-PACK.md`, `HOST_OWNED.md` and `contract/signatures.txt`.
