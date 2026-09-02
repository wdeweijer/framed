# OFP pasting viewer

Interactive Three.js visualization of the geometric 0-, 1-, and 2-skeleton of
an oriented framed poset.

The viewer currently assumes the loaded shape is a polyvoxel:

- every 1-cell has exactly one input and one output point;
- the immediate 1-faces of every 2-cell form one simple boundary cycle.

It accepts the version 1 `FramedPoset` JSON representation, either directly or
inside a dataset record's `ofp` field.

```bash
npm install
npm run dev
```

The development server prints its local URL. The default scene is the standard
3-cube; standard cubes through dimension 5 can also be generated in the UI.
Directions assigned to Mouse X, Y, or Z occupy the rotatable spatial frame.
Projected directions have editable azimuth, elevation, and scale. Arrow badges
show the direction number, and arrows sharing a target in one direction use the
separate triangular-angle force control.

```bash
npm test
npm run test:browser
npm run build
```
