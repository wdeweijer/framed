# Polyvoxel Enumeration Performance Notes

## Canonical representative regression

The following runs all enumerated the same 423,313 representatives and
928,476 factorizations with graph canonicalisation, at most 55 cells,
directions 0 through 3, directional lengths below 4, and 24 worker threads.

| Run | Total wall | Paste wall | Paste merge | Boundary caching |
| --- | ---: | ---: | ---: | ---: |
| `output6.txt` | 103.6 s | 92.071 s | 28.005 s | 5.970 s |
| `output7.txt` | 109.7 s | 100.017 s | 34.780 s | 5.001 s |
| `output8.txt` | 109.5 s | 98.955 s | 34.325 s | 5.343 s |

`output6.txt` predates the canonical-order metadata and the change to retain
the selected canonical OFP as each catalogue representative. The metadata
itself should be cheap, but these runs do not isolate it from the
representative-retention change.

Between `output7.txt` and `output8.txt`, the checks in
`Polyvoxel::from_isomorphism` were disabled in release mode. This recovered
only about 0.2 seconds overall and 0.5 seconds of paste merge time. Those
checks have since been restored; enumeration bypasses them explicitly through
`Polyvoxel::dangerously_from_parts_unchecked`.

The remaining regression against `output6.txt` is about 5.9 seconds overall.
It is concentrated in paste merging, which is 6.3 seconds slower. This is
large enough to investigate, although repeated runs are needed for precise
measurements.

## Canonical-only retained shapes

Enumeration now retains only canonical OFPs. A constructed candidate may have
an incidental cell order briefly, but `prepare_candidate` replaces it with its
selected canonical representative before the parallel result batch or the
catalogue stores it.

For a boundary `B`, release builds of `BoundaryNormalForm` store a 128-bit
BLAKE3 digest of its canonical form and the forward relabelling table of the
isomorphism

```text
eta_B: N(B) -> B.
```

The digest is the boundary-class index, so release builds retain neither the
canonical boundary nor the embedding, which would keep the non-canonical `B`
alive as its codomain. Debug builds retain the canonical boundary only to
assert that a matching digest is not a collision. For two matching boundaries,
the concrete map is computed directly from the two relabelling tables as

```text
B_left --eta_left^-1--> N(B) --eta_right--> B_right.
```

If a direction `i` is outside the total frame, then both directional
boundaries are the whole shape:

```text
delta_i^alpha X = X.
```

Previously the catalogue retained the first construction-ordered `X` and the
canonicaliser supplied the generally nontrivial map `N(X) -> X`. Now the
catalogue retains `N(X)` itself, so the corresponding map is the identity
`N(X) -> N(X)`. The old cache plumbing still stores this map explicitly.

For an inactive direction the canonical boundary is the canonical parent OFP,
and its relabelling table is represented by an explicitly generated identity
map. No second OFP or inverse table is retained.

## Deferred approaches

### Represent the identity implicitly

An implicit identity variant inside `BoundaryNormalForm` could avoid the
forward-map allocation for inactive directions, at the cost of another cache
representation branch.

### Construct traversal-canonical boundaries directly

Add a boundary operation for polyvoxels whose returned domain is already in
traversal-canonical order and whose embedding maps it into the original
polyvoxel. For an inactive direction on a traversal-canonical shape, it can
return the shared shape and construct the identity only on demand.

For active directions, this may avoid the current two-step process:

1. Construct the closed boundary in incidental cell order.
2. Traverse and relabel that boundary into canonical order.

A direct implementation needs either a proof or strong tests that its cell
order agrees with independent traversal normalisation. In particular, it
should not assume without verification that restricting the parent's traversal
order always gives the boundary's intrinsic traversal order. Graph
canonicalisation would continue to use the existing boundary followed by graph
normalisation.

## Next measurement

Before changing the representation, time identity construction separately
inside `CatalogBuilder::record`, or make it lazy and repeat the graph run. Keep
the bounds, thread count, and canonicalisation method identical to the runs
above.
