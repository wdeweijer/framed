import assert from "node:assert/strict";
import test from "node:test";

import { buildViewModel, standardCube, validateSerializedOFP } from "../src/ofp.js";
import { DirectionProjection } from "../src/projection.js";
import { DirectionalSpringSimulation } from "../src/simulation.js";

test("a tight arrow becomes one directed geometric edge", () => {
  const arrow = {
    version: 1,
    basis: [[[], []], [[0]]],
    faces_in: [[[], []], [[0]]],
    faces_out: [[[], []], [[1]]],
  };

  const model = buildViewModel(arrow);
  assert.equal(model.vertices.length, 2);
  assert.deepEqual(model.edges, [{
    dimension: 1,
    position: 0,
    basis: [0],
    direction: 0,
    source: 0,
    target: 1,
    triangular: false,
  }]);
});

test("arrows sharing a target and direction are triangular", () => {
  const fan = {
    version: 1,
    basis: [[[], [], [], []], [[0], [0], [1]]],
    faces_in: [[[], [], [], []], [[0], [1], [2]]],
    faces_out: [[[], [], [], []], [[3], [3], [3]]],
  };

  const model = buildViewModel(fan);
  assert.deepEqual(
    model.edges.map((edge) => edge.triangular),
    [true, true, false],
  );
});

test("the standard 3-cube has the expected visible skeleton and surfaces", () => {
  const cube = standardCube(3);
  const model = buildViewModel(cube);

  assert.deepEqual(cube.basis.map((level) => level.length), [8, 12, 6, 1]);
  assert.equal(model.vertices.length, 8);
  assert.equal(model.edges.length, 12);
  assert.equal(model.surfaces.length, 6);
  assert.ok(model.surfaces.every((surface) => surface.cycle.length === 4));
});

test("wrapped dataset records are accepted", () => {
  const cube = standardCube(2);
  assert.equal(validateSerializedOFP({ hash: "example", ofp: cube }), cube);
});

test("a non-tight 1-cell is rejected by the viewer contract", () => {
  const halfOpenArrow = {
    version: 1,
    basis: [[[]], [[0]]],
    faces_in: [[[]], [[0]]],
    faces_out: [[[]], [[]]],
  };

  assert.throws(
    () => buildViewModel(halfOpenArrow),
    /is not tight: expected one input and one output point/,
  );
});

test("the spring force orients a tight arrow from input to output", () => {
  const model = buildViewModel(standardCube(1));
  const simulation = new DirectionalSpringSimulation(model, {}, 1);
  simulation.step(1500);

  const source = simulation.positions[model.edges[0].source];
  const target = simulation.positions[model.edges[0].target];
  assert.ok(target[0] > source[0]);
  assert.ok(Math.abs(target[1] - source[1]) < 0.01);
  assert.ok(Math.abs(target[2] - source[2]) < 0.01);
});

test("projection directions can move between mouse axes and manual projection", () => {
  const projection = new DirectionProjection([0, 2, 7, 9], 4, 123);
  assert.deepEqual(
    projection.configuration().map(({ mode }) => mode),
    ["y", "x", "z", "projected"],
  );

  projection.setMode(9, "y");
  assert.equal(projection.setting(0).mode, "projected");
  assert.equal(projection.setting(9).mode, "y");

  projection.setProjected(0, { azimuth: 90, elevation: 0, scale: 2 });
  assert.ok(Math.abs(projection.axes[0].x) < 1e-12);
  assert.ok(Math.abs(projection.axes[0].y) < 1e-12);
  assert.ok(Math.abs(projection.axes[0].z - 2) < 1e-12);
});
