export const DEFAULT_SIMULATION_PARAMETERS = Object.freeze({
  edgeLength: 2,
  compression: 0.06,
  extension: 0.012,
  angle: 0.06,
  triangularAngle: 0.01,
  directionForce: 0.002,
  repulsion: 0.03,
  damping: 0.7,
  centering: 0.001,
  maxSpeed: 0.35,
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

function length(vector) {
  let squared = 0;
  for (const coordinate of vector) squared += coordinate * coordinate;
  return Math.sqrt(squared);
}

export class DirectionalSpringSimulation {
  constructor(model, parameters = {}, seed = 0x6f667000) {
    this.model = model;
    this.parameters = { ...DEFAULT_SIMULATION_PARAMETERS, ...parameters };
    this.dimension = Math.max(3, model.directions.length);
    this.directionCoordinates = new Map(
      model.directions.map((direction, coordinate) => [direction, coordinate]),
    );
    this.positions = [];
    this.velocities = [];
    this.frame = 0;
    this.restart(seed);
  }

  restart(seed) {
    const random = mulberry32(seed);
    const radius = this.parameters.edgeLength;
    this.positions = this.model.vertices.map(
      () => Float64Array.from(
        { length: this.dimension },
        () => (random() - 0.5) * radius * 2,
      ),
    );
    this.velocities = this.model.vertices.map(() => new Float64Array(this.dimension));
    this.frame = 0;
    this.centerPositions();
  }

  centerPositions() {
    if (this.positions.length === 0) return;
    const center = new Float64Array(this.dimension);
    for (const position of this.positions) {
      for (let axis = 0; axis < this.dimension; axis += 1) {
        center[axis] += position[axis];
      }
    }
    for (let axis = 0; axis < this.dimension; axis += 1) {
      center[axis] /= this.positions.length;
    }
    for (const position of this.positions) {
      for (let axis = 0; axis < this.dimension; axis += 1) {
        position[axis] -= center[axis];
      }
    }
  }

  step(iterations = 1) {
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      this.stepOnce();
    }
  }

  stepOnce() {
    const forces = this.positions.map(() => new Float64Array(this.dimension));
    this.addRepulsion(forces);
    this.addEdgeForces(forces);
    this.integrate(forces);
    this.frame += 1;
  }

  addRepulsion(forces) {
    const delta = new Float64Array(this.dimension);
    for (let left = 0; left < this.positions.length; left += 1) {
      for (let right = left + 1; right < this.positions.length; right += 1) {
        let distanceSquared = 0.0001;
        for (let axis = 0; axis < this.dimension; axis += 1) {
          delta[axis] = this.positions[right][axis] - this.positions[left][axis];
          distanceSquared += delta[axis] * delta[axis];
        }
        const scale = this.parameters.repulsion / (distanceSquared * Math.sqrt(distanceSquared));
        for (let axis = 0; axis < this.dimension; axis += 1) {
          const repulsion = delta[axis] * scale;
          forces[left][axis] -= repulsion;
          forces[right][axis] += repulsion;
        }
      }
    }
  }

  addEdgeForces(forces) {
    const delta = new Float64Array(this.dimension);
    const directionVector = new Float64Array(this.dimension);
    const tangentError = new Float64Array(this.dimension);

    for (const edge of this.model.edges) {
      const source = this.positions[edge.source];
      const target = this.positions[edge.target];
      const directionCoordinate = this.directionCoordinates.get(edge.direction);
      let distanceSquared = 0;
      for (let axis = 0; axis < this.dimension; axis += 1) {
        delta[axis] = target[axis] - source[axis];
        distanceSquared += delta[axis] * delta[axis];
      }

      const distance = Math.sqrt(distanceSquared);
      directionVector.fill(0);
      if (distance > 1e-8) {
        for (let axis = 0; axis < this.dimension; axis += 1) {
          directionVector[axis] = delta[axis] / distance;
        }
      } else {
        directionVector[directionCoordinate] = 1;
      }

      const stretch = distance - this.parameters.edgeLength;
      const lengthStiffness = stretch < 0
        ? this.parameters.compression
        : this.parameters.extension;
      for (let axis = 0; axis < this.dimension; axis += 1) {
        const spring = directionVector[axis] * lengthStiffness * stretch;
        forces[edge.source][axis] += spring;
        forces[edge.target][axis] -= spring;
      }

      const alignment = directionVector[directionCoordinate];
      let tangentMagnitudeSquared = 0;
      for (let axis = 0; axis < this.dimension; axis += 1) {
        tangentError[axis] = directionVector[axis] * alignment
          - (axis === directionCoordinate ? 1 : 0);
        tangentMagnitudeSquared += tangentError[axis] * tangentError[axis];
      }

      const tangentMagnitude = Math.sqrt(tangentMagnitudeSquared);
      if (tangentMagnitude > 1e-8) {
        const angleCoefficient = edge.triangular
          ? this.parameters.triangularAngle
          : this.parameters.angle;
        const scale = distance * angleCoefficient
          + this.parameters.directionForce / tangentMagnitude;
        for (let axis = 0; axis < this.dimension; axis += 1) {
          const spring = tangentError[axis] * scale;
          forces[edge.source][axis] += spring;
          forces[edge.target][axis] -= spring;
        }
      }
    }
  }

  integrate(forces) {
    const { centering, damping, maxSpeed } = this.parameters;
    for (let vertex = 0; vertex < this.positions.length; vertex += 1) {
      for (let axis = 0; axis < this.dimension; axis += 1) {
        forces[vertex][axis] -= this.positions[vertex][axis] * centering;
        this.velocities[vertex][axis] = (
          this.velocities[vertex][axis] + forces[vertex][axis]
        ) * damping;
      }

      const speed = length(this.velocities[vertex]);
      if (speed > maxSpeed) {
        const scale = maxSpeed / speed;
        for (let axis = 0; axis < this.dimension; axis += 1) {
          this.velocities[vertex][axis] *= scale;
        }
      }

      for (let axis = 0; axis < this.dimension; axis += 1) {
        this.positions[vertex][axis] += this.velocities[vertex][axis];
      }
    }
  }

  kineticEnergy() {
    let energy = 0;
    for (const velocity of this.velocities) {
      for (const coordinate of velocity) energy += coordinate * coordinate;
    }
    return energy / 2;
  }
}
