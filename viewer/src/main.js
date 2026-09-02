import { buildViewModel, standardCube } from "./ofp.js";
import { DirectionProjection, PROJECTION_MODES } from "./projection.js";
import { directionColor, OFPScene } from "./scene.js";
import {
  DEFAULT_SIMULATION_PARAMETERS,
  DirectionalSpringSimulation,
} from "./simulation.js";

const elements = {
  viewport: document.getElementById("viewport"),
  tooltip: document.getElementById("tooltip"),
  sourceName: document.getElementById("source-name"),
  status: document.getElementById("status"),
  playPause: document.getElementById("play-pause"),
  step: document.getElementById("step"),
  restart: document.getElementById("restart"),
  resetView: document.getElementById("reset-view"),
  randomProjection: document.getElementById("random-projection"),
  frameStat: document.getElementById("frame-stat"),
  energyStat: document.getElementById("energy-stat"),
  dimensionStat: document.getElementById("dimension-stat"),
  pointStat: document.getElementById("point-stat"),
  edgeStat: document.getElementById("edge-stat"),
  surfaceStat: document.getElementById("surface-stat"),
  directionLegend: document.getElementById("direction-legend"),
  projectionControls: document.getElementById("projection-controls"),
  cubeDimension: document.getElementById("cube-dimension"),
  loadCube: document.getElementById("load-cube"),
  dataFile: document.getElementById("data-file"),
  jsonText: document.getElementById("json-text"),
  loadJson: document.getElementById("load-json"),
};

const parameterControls = [
  { id: "edge-length", key: "edgeLength", digits: 2 },
  { id: "compression", key: "compression", digits: 3 },
  { id: "extension", key: "extension", digits: 3 },
  { id: "angle-force", key: "angle", digits: 3 },
  { id: "triangular-angle-force", key: "triangularAngle", digits: 3 },
  { id: "repulsion", key: "repulsion", digits: 3 },
  { id: "damping", key: "damping", digits: 2 },
].map((control) => ({
  ...control,
  input: document.getElementById(control.id),
  output: document.getElementById(`${control.id}-out`),
}));

const stepsPerFrameInput = document.getElementById("steps-per-frame");
const stepsPerFrameOutput = document.getElementById("steps-per-frame-out");
const surfaceOpacityInput = document.getElementById("surface-opacity");
const surfaceOpacityOutput = document.getElementById("surface-opacity-out");

let model;
let simulation;
let projection;
let running = true;
let stepsPerFrame = Number(stepsPerFrameInput.value);
let projectionSeed = 0x70726f6a;
let simulationSeed = 0x6f667000;

const scene = new OFPScene(elements.viewport, elements.tooltip);

function freshSeed() {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  return values[0];
}

function setStatus(message, error = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", error);
}

function setRunning(nextRunning) {
  running = nextRunning;
  elements.playPause.textContent = running ? "Pause" : "Play";
}

function updateSummary() {
  elements.dimensionStat.textContent = String(model.ofp.basis.length - 1);
  elements.pointStat.textContent = String(model.vertices.length);
  elements.edgeStat.textContent = String(model.edges.length);
  elements.surfaceStat.textContent = String(model.surfaces.length);
  elements.directionLegend.replaceChildren();
  model.directions.forEach((direction, index) => {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = `#${directionColor(direction, index).getHexString()}`;
    item.append(swatch, document.createTextNode(`{${direction}}`));
    elements.directionLegend.append(item);
  });
}

function updateFrameStats() {
  elements.frameStat.textContent = `Frame ${simulation.frame.toLocaleString()}`;
  elements.energyStat.textContent = `Energy ${simulation.kineticEnergy().toExponential(2)}`;
}

function updateProjectionButton() {
  elements.randomProjection.disabled = !projection?.hasProjectedDirections();
}

function addProjectionRange(container, setting, property, label, minimum, maximum, step) {
  const control = document.createElement("div");
  control.className = "range-control projection-range";

  const inputId = `projection-${setting.direction}-${property}`;
  const inputLabel = document.createElement("label");
  inputLabel.htmlFor = inputId;
  inputLabel.textContent = label;

  const output = document.createElement("output");
  output.setAttribute("for", inputId);
  const digits = property === "scale" ? 2 : 0;
  output.value = setting[property].toFixed(digits);

  const input = document.createElement("input");
  input.id = inputId;
  input.type = "range";
  input.min = String(minimum);
  input.max = String(maximum);
  input.step = String(step);
  input.value = String(setting[property]);
  input.addEventListener("input", () => {
    const value = Number(input.value);
    output.value = value.toFixed(digits);
    projection.setProjected(setting.direction, { [property]: value });
    scene.update();
  });
  input.addEventListener("change", () => scene.fitView());

  control.append(inputLabel, output, input);
  container.append(control);
}

function renderProjectionControls() {
  elements.projectionControls.replaceChildren();
  if (!projection) return;

  for (const setting of projection.configuration()) {
    const row = document.createElement("div");
    row.className = "projection-direction";
    row.dataset.direction = String(setting.direction);

    const heading = document.createElement("div");
    heading.className = "projection-heading";
    const directionName = document.createElement("span");
    directionName.className = "projection-name";
    const swatch = document.createElement("i");
    const directionIndex = model.directions.indexOf(setting.direction);
    swatch.style.backgroundColor = `#${directionColor(
      setting.direction,
      directionIndex,
    ).getHexString()}`;
    directionName.append(swatch, document.createTextNode(`{${setting.direction}}`));

    const mode = document.createElement("select");
    mode.setAttribute("aria-label", `Direction ${setting.direction} projection mode`);
    for (const candidate of PROJECTION_MODES) {
      const option = document.createElement("option");
      option.value = candidate.value;
      option.textContent = candidate.label;
      option.selected = candidate.value === setting.mode;
      mode.append(option);
    }
    mode.addEventListener("change", () => {
      projection.setMode(setting.direction, mode.value);
      renderProjectionControls();
      scene.update();
      scene.fitView();
    });
    heading.append(directionName, mode);
    row.append(heading);

    const projectedSettings = document.createElement("div");
    projectedSettings.className = "projection-settings";
    projectedSettings.hidden = setting.mode !== "projected";
    addProjectionRange(projectedSettings, setting, "azimuth", "Azimuth", -180, 180, 1);
    addProjectionRange(projectedSettings, setting, "elevation", "Elevation", -90, 90, 1);
    addProjectionRange(projectedSettings, setting, "scale", "Scale", 0.1, 3, 0.05);
    row.append(projectedSettings);
    elements.projectionControls.append(row);
  }
  updateProjectionButton();
}

function currentParameters() {
  return Object.fromEntries(
    parameterControls.map((control) => [control.key, Number(control.input.value)]),
  );
}

function loadShape(rawOFP, name, updateText = true) {
  try {
    const nextModel = buildViewModel(rawOFP);
    const nextSimulation = new DirectionalSpringSimulation(
      nextModel,
      { ...DEFAULT_SIMULATION_PARAMETERS, ...currentParameters() },
      simulationSeed,
    );
    const nextProjection = new DirectionProjection(
      nextModel.directions,
      nextSimulation.dimension,
      projectionSeed,
    );

    model = nextModel;
    simulation = nextSimulation;
    projection = nextProjection;
    scene.load(model, simulation, projection, Number(surfaceOpacityInput.value));
    elements.sourceName.textContent = name;
    if (updateText) elements.jsonText.value = JSON.stringify(model.ofp, null, 2);
    updateSummary();
    renderProjectionControls();
    updateFrameStats();
    setRunning(true);
    setStatus("Ready");
    document.documentElement.dataset.ready = "true";
  } catch (error) {
    setStatus(error.message, true);
  }
}

for (const control of parameterControls) {
  control.input.value = String(DEFAULT_SIMULATION_PARAMETERS[control.key]);
  const update = () => {
    const value = Number(control.input.value);
    control.output.value = value.toFixed(control.digits);
    if (simulation) simulation.parameters[control.key] = value;
  };
  control.input.addEventListener("input", update);
  update();
}

stepsPerFrameInput.addEventListener("input", () => {
  stepsPerFrame = Number(stepsPerFrameInput.value);
  stepsPerFrameOutput.value = String(stepsPerFrame);
});

surfaceOpacityInput.addEventListener("input", () => {
  const opacity = Number(surfaceOpacityInput.value);
  surfaceOpacityOutput.value = opacity.toFixed(2);
  scene.setSurfaceOpacity(opacity);
});

elements.playPause.addEventListener("click", () => setRunning(!running));
elements.step.addEventListener("click", () => {
  simulation.step();
  scene.update();
  updateFrameStats();
});
elements.restart.addEventListener("click", () => {
  simulationSeed = freshSeed();
  simulation.restart(simulationSeed);
  scene.update();
  scene.fitView();
  updateFrameStats();
});
elements.resetView.addEventListener("click", () => scene.fitView());
elements.randomProjection.addEventListener("click", () => {
  projectionSeed = freshSeed();
  projection.randomizeProjected(projectionSeed);
  renderProjectionControls();
  scene.update();
  scene.fitView();
});

elements.loadCube.addEventListener("click", () => {
  const dimension = Number(elements.cubeDimension.value);
  loadShape(standardCube(dimension), `Standard ${dimension}-cube`);
});

elements.dataFile.addEventListener("change", async () => {
  const [file] = elements.dataFile.files;
  if (!file) return;
  try {
    const text = await file.text();
    elements.jsonText.value = text;
    loadShape(JSON.parse(text), file.name, false);
  } catch (error) {
    setStatus(error.message, true);
  }
});

elements.loadJson.addEventListener("click", () => {
  try {
    loadShape(JSON.parse(elements.jsonText.value), "Entered JSON", false);
  } catch (error) {
    setStatus(error.message, true);
  }
});

scene.setAnimationLoop(() => {
  if (running && simulation) {
    simulation.step(stepsPerFrame);
    scene.update();
    updateFrameStats();
  }
});

loadShape(standardCube(3), "Standard 3-cube");
