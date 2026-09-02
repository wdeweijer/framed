import * as THREE from "three";

export const PROJECTION_MODES = Object.freeze([
  Object.freeze({ value: "x", label: "Mouse X" }),
  Object.freeze({ value: "y", label: "Mouse Y" }),
  Object.freeze({ value: "z", label: "Mouse Z" }),
  Object.freeze({ value: "projected", label: "Projected" }),
]);

const FRAME_AXES = Object.freeze({
  x: Object.freeze([1, 0, 0]),
  y: Object.freeze([0, 1, 0]),
  z: Object.freeze([0, 0, 1]),
});

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function randomUnitVector(random) {
  let x;
  let y;
  let z;
  let squared;
  do {
    x = random() * 2 - 1;
    y = random() * 2 - 1;
    z = random() * 2 - 1;
    squared = x * x + y * y + z * z;
  } while (squared < 1e-6 || squared > 1);
  const scale = 1 / Math.sqrt(squared);
  return new THREE.Vector3(x * scale, y * scale, z * scale);
}

function vectorToAngles(vector) {
  return {
    azimuth: THREE.MathUtils.radToDeg(Math.atan2(vector.z, vector.x)),
    elevation: THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(vector.y, -1, 1))),
  };
}

function projectedAxis({ azimuth, elevation, scale }) {
  const azimuthRadians = THREE.MathUtils.degToRad(azimuth);
  const elevationRadians = THREE.MathUtils.degToRad(elevation);
  const horizontal = Math.cos(elevationRadians) * scale;
  return new THREE.Vector3(
    Math.cos(azimuthRadians) * horizontal,
    Math.sin(elevationRadians) * scale,
    Math.sin(azimuthRadians) * horizontal,
  );
}

function frameAxis(mode) {
  return new THREE.Vector3(...FRAME_AXES[mode]);
}

export class DirectionProjection {
  constructor(directions, dimension, seed = 0x70726f6a) {
    this.directions = [...directions];
    this.dimension = dimension;
    this.settings = [];
    this.axes = [];
    this.initialize(seed);
  }

  initialize(seed) {
    const random = mulberry32(seed);
    const frameModes = ["y", "x", "z"];
    this.settings = this.directions.map((direction, coordinate) => {
      const vector = coordinate < frameModes.length
        ? frameAxis(frameModes[coordinate])
        : randomUnitVector(random);
      return {
        direction,
        coordinate,
        mode: frameModes[coordinate] ?? "projected",
        ...vectorToAngles(vector),
        scale: 1,
      };
    });
    this.rebuildAxes();
  }

  configuration() {
    return this.settings.map((setting) => ({ ...setting }));
  }

  setting(direction) {
    const result = this.settings.find((entry) => entry.direction === direction);
    if (!result) throw new Error(`unknown direction ${direction}`);
    return result;
  }

  setMode(direction, mode) {
    if (!PROJECTION_MODES.some((candidate) => candidate.value === mode)) {
      throw new Error(`unknown projection mode ${mode}`);
    }

    const selected = this.setting(direction);
    if (mode !== "projected") {
      const previous = this.settings.find(
        (entry) => entry !== selected && entry.mode === mode,
      );
      if (previous) previous.mode = "projected";
    }
    selected.mode = mode;
    this.rebuildAxes();
  }

  setProjected(direction, values) {
    const selected = this.setting(direction);
    for (const property of ["azimuth", "elevation", "scale"]) {
      if (values[property] === undefined) continue;
      if (!Number.isFinite(values[property])) {
        throw new Error(`${property} must be finite`);
      }
      selected[property] = values[property];
    }
    this.rebuildAxes();
  }

  randomizeProjected(seed) {
    const random = mulberry32(seed);
    for (const setting of this.settings) {
      if (setting.mode !== "projected") continue;
      Object.assign(setting, vectorToAngles(randomUnitVector(random)));
    }
    this.rebuildAxes();
  }

  hasProjectedDirections() {
    return this.settings.some((setting) => setting.mode === "projected");
  }

  rebuildAxes() {
    const fallbackAxes = [
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 1),
    ];
    this.axes = Array.from(
      { length: this.dimension },
      (_, coordinate) => fallbackAxes[coordinate % fallbackAxes.length].clone(),
    );

    for (const setting of this.settings) {
      this.axes[setting.coordinate] = setting.mode === "projected"
        ? projectedAxis(setting)
        : frameAxis(setting.mode);
    }
  }

  project(position, target = new THREE.Vector3()) {
    target.set(0, 0, 0);
    for (let axis = 0; axis < this.dimension; axis += 1) {
      target.addScaledVector(this.axes[axis], position[axis]);
    }
    return target;
  }
}
