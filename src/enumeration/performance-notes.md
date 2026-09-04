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

## Why an identity embedding is allocated

For a boundary `B`, `BoundaryNormalForm` stores both its canonical form and an
isomorphism

```text
eta_B: N(B) -> B.
```

For two matching boundaries, their concrete isomorphism is computed as

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

At present, `CatalogBuilder::record` constructs a complete identity embedding
for every new representative. This allocates both `map` and `inv` tables. The
embedding is only used by boundary caching when direction 0 is enabled for
cylinder matching but is absent from that representative's total frame.

## Deferred approaches

### Allocate the identity lazily

Remove `canonical_into_shape` from `WorkingEntry`. In the inactive-direction
branch of `populate_boundary_caches`, construct the identity there instead.
This moves the work into the parallel boundary-caching phase and avoids it for
representatives whose cached directions are all active.

An implicit identity variant inside `BoundaryNormalForm` could avoid even that
allocation, but would complicate composition slightly.

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
