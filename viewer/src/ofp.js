function fail(message) {
  throw new Error(message);
}

function validateIntSet(value, label, upperBound = null) {
  if (!Array.isArray(value)) fail(`${label} must be an array`);

  let previous = -1;
  for (const entry of value) {
    if (!Number.isSafeInteger(entry) || entry < 0) {
      fail(`${label} must contain non-negative integers`);
    }
    if (entry <= previous) fail(`${label} must be sorted without duplicates`);
    if (upperBound !== null && entry >= upperBound) {
      fail(`${label} contains the out-of-bounds index ${entry}`);
    }
    previous = entry;
  }
}

function isSubset(subset, superset) {
  let left = 0;
  let right = 0;
  while (left < subset.length && right < superset.length) {
    if (subset[left] === superset[right]) {
      left += 1;
      right += 1;
    } else if (subset[left] > superset[right]) {
      right += 1;
    } else {
      return false;
    }
  }
  return left === subset.length;
}

function unwrapOFP(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail("the loaded value must be a serialized OFP object");
  }
  return value.ofp && typeof value.ofp === "object" ? value.ofp : value;
}

export function validateSerializedOFP(value) {
  const ofp = unwrapOFP(value);
  if (ofp.version !== 1) fail(`unsupported OFP version ${String(ofp.version)}`);

  const tables = [ofp.basis, ofp.faces_in, ofp.faces_out];
  if (!tables.every(Array.isArray)) {
    fail("basis, faces_in, and faces_out must be arrays");
  }
  if (ofp.faces_in.length !== ofp.basis.length
      || ofp.faces_out.length !== ofp.basis.length) {
    fail("basis and signed face tables must have the same number of levels");
  }

  const levelSizes = ofp.basis.map((level, dimension) => {
    if (!Array.isArray(level)) fail(`basis level ${dimension} must be an array`);
    return level.length;
  });

  for (let dimension = 0; dimension < ofp.basis.length; dimension += 1) {
    const basisLevel = ofp.basis[dimension];
    const facesInLevel = ofp.faces_in[dimension];
    const facesOutLevel = ofp.faces_out[dimension];
    if (!Array.isArray(facesInLevel) || !Array.isArray(facesOutLevel)
        || facesInLevel.length !== basisLevel.length
        || facesOutLevel.length !== basisLevel.length) {
      fail(`signed face rows at level ${dimension} must match the cells`);
    }

    for (let position = 0; position < basisLevel.length; position += 1) {
      const cellName = `cell (${dimension}, ${position})`;
      const basis = basisLevel[position];
      validateIntSet(basis, `${cellName} basis`);
      if (basis.length !== dimension) {
        fail(`${cellName} has basis cardinality ${basis.length}, expected ${dimension}`);
      }

      const faceBound = dimension === 0 ? 0 : levelSizes[dimension - 1];
      const facesIn = facesInLevel[position];
      const facesOut = facesOutLevel[position];
      validateIntSet(facesIn, `${cellName} input faces`, faceBound);
      validateIntSet(facesOut, `${cellName} output faces`, faceBound);

      if (dimension === 0 && (facesIn.length !== 0 || facesOut.length !== 0)) {
        fail(`${cellName} is zero-dimensional but has faces`);
      }

      const outputSet = new Set(facesOut);
      if (facesIn.some((face) => outputSet.has(face))) {
        fail(`${cellName} has a face with both orientations`);
      }

      if (dimension > 0) {
        const allFaces = [...facesIn, ...facesOut];
        if (allFaces.length === 0) fail(`${cellName} has no faces`);
        for (const face of allFaces) {
          if (!isSubset(ofp.basis[dimension - 1][face], basis)) {
            fail(`${cellName} has a face whose basis is not a subset of its basis`);
          }
        }
        for (const direction of basis) {
          const realized = allFaces.some(
            (face) => !ofp.basis[dimension - 1][face].includes(direction),
          );
          if (!realized) {
            fail(`${cellName} does not realize its face opposite direction ${direction}`);
          }
        }
      }
    }
  }

  if (ofp.basis.length > 1) {
    for (let position = 0; position < ofp.basis[1].length; position += 1) {
      const inputs = ofp.faces_in[1][position];
      const outputs = ofp.faces_out[1][position];
      if (inputs.length !== 1 || outputs.length !== 1) {
        fail(
          `1-cell (1, ${position}) is not a voxel: expected one input and one output point`,
        );
      }
      if (inputs[0] === outputs[0]) {
        fail(`1-cell (1, ${position}) has the same input and output point`);
      }
    }
  }

  return ofp;
}

function boundaryCycle(surfacePosition, edgePositions, edgesByPosition) {
  if (edgePositions.length < 3) {
    fail(`2-cell (2, ${surfacePosition}) has fewer than three boundary edges`);
  }

  const adjacency = new Map();
  const addIncident = (vertex, edgePosition) => {
    const incident = adjacency.get(vertex) ?? [];
    incident.push(edgePosition);
    adjacency.set(vertex, incident);
  };

  for (const edgePosition of edgePositions) {
    const edge = edgesByPosition[edgePosition];
    if (!edge) fail(`2-cell (2, ${surfacePosition}) refers to a missing 1-cell`);
    addIncident(edge.source, edgePosition);
    addIncident(edge.target, edgePosition);
  }

  for (const [vertex, incident] of adjacency) {
    incident.sort((left, right) => left - right);
    if (incident.length !== 2) {
      fail(
        `2-cell (2, ${surfacePosition}) is not a surface cycle: point ${vertex} has degree ${incident.length}`,
      );
    }
  }

  const startVertex = Math.min(...adjacency.keys());
  const cycle = [startVertex];
  const usedEdges = new Set();
  let currentVertex = startVertex;
  let previousEdge = null;

  for (let step = 0; step < edgePositions.length; step += 1) {
    const candidates = adjacency.get(currentVertex);
    const nextEdgePosition = candidates.find((edge) => edge !== previousEdge);
    if (nextEdgePosition === undefined || usedEdges.has(nextEdgePosition)) {
      fail(`2-cell (2, ${surfacePosition}) has a disconnected or repeated boundary`);
    }

    usedEdges.add(nextEdgePosition);
    const edge = edgesByPosition[nextEdgePosition];
    const nextVertex = edge.source === currentVertex ? edge.target : edge.source;
    previousEdge = nextEdgePosition;
    currentVertex = nextVertex;

    if (step + 1 < edgePositions.length) {
      if (currentVertex === startVertex) {
        fail(`2-cell (2, ${surfacePosition}) has more than one boundary component`);
      }
      cycle.push(currentVertex);
    }
  }

  if (currentVertex !== startVertex || usedEdges.size !== edgePositions.length) {
    fail(`2-cell (2, ${surfacePosition}) does not have one closed boundary cycle`);
  }
  return cycle;
}

export function buildViewModel(value) {
  const ofp = validateSerializedOFP(value);
  const directions = [...new Set(ofp.basis.flat(2))].sort((left, right) => left - right);
  const vertices = (ofp.basis[0] ?? []).map((basis, position) => ({
    dimension: 0,
    position,
    basis,
  }));

  const rawEdges = (ofp.basis[1] ?? []).map((basis, position) => ({
    dimension: 1,
    position,
    basis,
    direction: basis[0],
    source: ofp.faces_in[1][position][0],
    target: ofp.faces_out[1][position][0],
  }));

  const incomingByDirection = new Map();
  for (const edge of rawEdges) {
    const key = `${edge.target}:${edge.direction}`;
    incomingByDirection.set(key, (incomingByDirection.get(key) ?? 0) + 1);
  }
  const edges = rawEdges.map((edge) => ({
    ...edge,
    triangular: incomingByDirection.get(`${edge.target}:${edge.direction}`) > 1,
  }));

  const surfaces = (ofp.basis[2] ?? []).map((basis, position) => {
    const edgePositions = [
      ...ofp.faces_in[2][position],
      ...ofp.faces_out[2][position],
    ].sort((left, right) => left - right);
    return {
      dimension: 2,
      position,
      basis,
      edgePositions,
      cycle: boundaryCycle(position, edgePositions, edges),
    };
  });

  return { ofp, directions, vertices, edges, surfaces };
}

function ternaryState(code, dimension) {
  const state = new Array(dimension);
  let remaining = code;
  for (let direction = 0; direction < dimension; direction += 1) {
    state[direction] = remaining % 3;
    remaining = Math.floor(remaining / 3);
  }
  return state;
}

function stateKey(state) {
  return state.join(",");
}

export function standardCube(dimension) {
  if (!Number.isSafeInteger(dimension) || dimension < 0) {
    fail("cube dimension must be a non-negative integer");
  }

  const levels = Array.from({ length: dimension + 1 }, () => []);
  const cellCount = 3 ** dimension;
  for (let code = 0; code < cellCount; code += 1) {
    const state = ternaryState(code, dimension);
    const basis = state
      .map((coordinate, direction) => (coordinate === 1 ? direction : null))
      .filter((direction) => direction !== null);
    levels[basis.length].push({ state, basis });
  }

  const indices = new Map();
  levels.forEach((level, cellDimension) => {
    level.forEach((cell, position) => {
      indices.set(stateKey(cell.state), { dimension: cellDimension, position });
    });
  });

  const basis = levels.map((level) => level.map((cell) => cell.basis));
  const facesIn = levels.map((level) => level.map(() => []));
  const facesOut = levels.map((level) => level.map(() => []));

  for (let cellDimension = 1; cellDimension <= dimension; cellDimension += 1) {
    levels[cellDimension].forEach((cell, position) => {
      for (const direction of cell.basis) {
        const inputState = [...cell.state];
        const outputState = [...cell.state];
        inputState[direction] = 0;
        outputState[direction] = 2;
        facesIn[cellDimension][position].push(indices.get(stateKey(inputState)).position);
        facesOut[cellDimension][position].push(indices.get(stateKey(outputState)).position);
      }
      facesIn[cellDimension][position].sort((left, right) => left - right);
      facesOut[cellDimension][position].sort((left, right) => left - right);
    });
  }

  return {
    version: 1,
    basis,
    faces_in: facesIn,
    faces_out: facesOut,
  };
}
