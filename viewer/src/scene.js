import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DIRECTION_PALETTE = [
  0xc23b3b,
  0x2767b2,
  0x25845b,
  0xd18a22,
  0x7851a9,
  0x00838f,
  0xbd4f8a,
  0x657326,
];

const Y_AXIS = new THREE.Vector3(0, 1, 0);

export function directionColor(direction, index = direction) {
  if (index < DIRECTION_PALETTE.length) return new THREE.Color(DIRECTION_PALETTE[index]);
  const hue = (index * 0.618033988749895) % 1;
  return new THREE.Color().setHSL(hue, 0.58, 0.46);
}

function disposeObject(object) {
  const geometries = new Set();
  const materials = new Set();
  object.traverse((child) => {
    if (child.geometry) geometries.add(child.geometry);
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => materials.add(material));
    } else if (child.material) {
      materials.add(child.material);
    }
  });
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((material) => {
    for (const value of Object.values(material)) {
      if (value?.isTexture) value.dispose();
    }
    material.dispose();
  });
}

function matrixAlongSegment(start, end, radius, target) {
  const delta = new THREE.Vector3().subVectors(end, start);
  const segmentLength = Math.max(delta.length(), 0.0001);
  delta.multiplyScalar(1 / segmentLength);
  const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
  const rotation = new THREE.Quaternion().setFromUnitVectors(Y_AXIS, delta);
  const scale = new THREE.Vector3(radius, segmentLength, radius);
  return target.compose(midpoint, rotation, scale);
}

function surfaceColor(basis, directionIndices) {
  const color = new THREE.Color(0, 0, 0);
  for (const direction of basis) {
    color.add(directionColor(direction, directionIndices.get(direction)));
  }
  return color.multiplyScalar(1 / basis.length);
}

function directionLabelMaterial(direction, color) {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  const colorStyle = `#${color.getHexString()}`;

  context.beginPath();
  context.arc(size / 2, size / 2, 45, 0, Math.PI * 2);
  context.fillStyle = "rgba(255, 255, 255, 0.94)";
  context.fill();
  context.lineWidth = 8;
  context.strokeStyle = colorStyle;
  context.stroke();

  const label = String(direction);
  const fontSize = label.length <= 2 ? 58 : Math.max(32, 76 - label.length * 8);
  context.fillStyle = "#17202a";
  context.font = `700 ${fontSize}px sans-serif`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(label, size / 2, size / 2 + 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
}

export class OFPScene {
  constructor(container, tooltip) {
    this.container = container;
    this.tooltip = tooltip;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf1f3f5);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 500);
    const preserveDrawingBuffer = new URLSearchParams(window.location.search).has("test");
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.append(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x66717d, 2.15));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
    keyLight.position.set(4, 6, 5);
    this.scene.add(keyLight);

    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.model = null;
    this.simulation = null;
    this.projection = null;
    this.projectedPositions = [];
    this.surfaceMeshes = [];
    this.edgeLabels = [];
    this.pickables = [];
    this.surfaceOpacity = 0.2;
    this.hasFittedView = false;

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.renderer.domElement.addEventListener("pointermove", (event) => this.pick(event));
    this.renderer.domElement.addEventListener("pointerleave", () => {
      this.tooltip.hidden = true;
    });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
  }

  load(model, simulation, projection, surfaceOpacity = this.surfaceOpacity) {
    disposeObject(this.content);
    this.scene.remove(this.content);
    this.content = new THREE.Group();
    this.scene.add(this.content);

    this.model = model;
    this.simulation = simulation;
    this.projection = projection;
    this.surfaceOpacity = surfaceOpacity;
    this.projectedPositions = model.vertices.map(() => new THREE.Vector3());
    this.surfaceMeshes = [];
    this.edgeLabels = [];
    this.pickables = [];
    this.hasFittedView = false;

    this.makePoints();
    this.makeArrows();
    this.makeArrowLabels();
    this.makeSurfaces();
    this.update();
    this.fitView();
  }

  makePoints() {
    if (this.model.vertices.length === 0) {
      this.points = null;
      return;
    }
    this.points = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.11, 20, 14),
      new THREE.MeshStandardMaterial({ color: 0x20272e, roughness: 0.68 }),
      this.model.vertices.length,
    );
    this.points.userData.kind = "vertex";
    this.points.userData.items = this.model.vertices;
    this.content.add(this.points);
    this.pickables.push(this.points);
  }

  makeArrows() {
    if (this.model.edges.length === 0) {
      this.arrowShafts = null;
      this.arrowHeads = null;
      return;
    }
    const directionIndices = new Map(
      this.model.directions.map((direction, index) => [direction, index]),
    );
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
    });
    this.arrowShafts = new THREE.InstancedMesh(
      new THREE.CylinderGeometry(1, 1, 1, 12),
      material,
      this.model.edges.length,
    );
    this.arrowHeads = new THREE.InstancedMesh(
      new THREE.ConeGeometry(1, 1, 16),
      material.clone(),
      this.model.edges.length,
    );
    for (let index = 0; index < this.model.edges.length; index += 1) {
      const edge = this.model.edges[index];
      const color = directionColor(edge.direction, directionIndices.get(edge.direction));
      this.arrowShafts.setColorAt(index, color);
      this.arrowHeads.setColorAt(index, color);
    }
    this.arrowShafts.instanceColor.needsUpdate = true;
    this.arrowHeads.instanceColor.needsUpdate = true;
    for (const object of [this.arrowShafts, this.arrowHeads]) {
      object.userData.kind = "edge";
      object.userData.items = this.model.edges;
      this.content.add(object);
      this.pickables.push(object);
    }
  }

  makeArrowLabels() {
    const directionIndices = new Map(
      this.model.directions.map((direction, index) => [direction, index]),
    );
    const materials = new Map();
    for (const edge of this.model.edges) {
      let material = materials.get(edge.direction);
      if (!material) {
        material = directionLabelMaterial(
          edge.direction,
          directionColor(edge.direction, directionIndices.get(edge.direction)),
        );
        materials.set(edge.direction, material);
      }
      const label = new THREE.Sprite(material);
      label.scale.setScalar(0.3);
      label.renderOrder = 20;
      this.edgeLabels.push(label);
      this.content.add(label);
    }
  }

  makeSurfaces() {
    const directionIndices = new Map(
      this.model.directions.map((direction, index) => [direction, index]),
    );
    for (const surface of this.model.surfaces) {
      const positions = new Float32Array((surface.cycle.length + 1) * 3);
      const indices = [];
      for (let index = 0; index < surface.cycle.length; index += 1) {
        indices.push(0, index + 1, ((index + 1) % surface.cycle.length) + 1);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage),
      );
      geometry.setIndex(indices);

      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          color: surfaceColor(surface.basis, directionIndices),
          opacity: this.surfaceOpacity,
          transparent: true,
          depthWrite: false,
          roughness: 0.82,
          metalness: 0,
          side: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1,
        }),
      );
      mesh.userData.kind = "surface";
      mesh.userData.item = surface;
      this.surfaceMeshes.push(mesh);
      this.content.add(mesh);
      this.pickables.push(mesh);
    }
  }

  update() {
    if (!this.model) return;
    for (let index = 0; index < this.simulation.positions.length; index += 1) {
      this.projection.project(this.simulation.positions[index], this.projectedPositions[index]);
    }
    this.updatePoints();
    this.updateArrows();
    this.updateSurfaces();
    this.updateArrowLabels();
  }

  updatePoints() {
    if (!this.points) return;
    const matrix = new THREE.Matrix4();
    for (let index = 0; index < this.projectedPositions.length; index += 1) {
      matrix.makeTranslation(...this.projectedPositions[index].toArray());
      this.points.setMatrixAt(index, matrix);
    }
    this.points.instanceMatrix.needsUpdate = true;
    this.points.computeBoundingSphere();
  }

  updateArrows() {
    if (!this.arrowShafts) return;
    const matrix = new THREE.Matrix4();
    const direction = new THREE.Vector3();
    const shaftStart = new THREE.Vector3();
    const shaftEnd = new THREE.Vector3();
    const headCenter = new THREE.Vector3();
    const nodeRadius = 0.1;
    const headLength = 0.22;
    const headRadius = 0.082;

    for (let index = 0; index < this.model.edges.length; index += 1) {
      const edge = this.model.edges[index];
      const source = this.projectedPositions[edge.source];
      const target = this.projectedPositions[edge.target];
      direction.subVectors(target, source);
      if (direction.lengthSq() < 1e-8) {
        const coordinate = this.simulation.directionCoordinates.get(edge.direction);
        direction.copy(this.projection.axes[coordinate]);
      }
      direction.normalize();
      shaftStart.copy(source).addScaledVector(direction, nodeRadius);
      shaftEnd.copy(target).addScaledVector(direction, -(nodeRadius + headLength));
      if (shaftEnd.clone().sub(shaftStart).dot(direction) < 0.02) {
        shaftEnd.copy(shaftStart).addScaledVector(direction, 0.02);
      }
      matrixAlongSegment(shaftStart, shaftEnd, 0.025, matrix);
      this.arrowShafts.setMatrixAt(index, matrix);

      headCenter.copy(target).addScaledVector(direction, -(nodeRadius + headLength / 2));
      const headTip = headCenter.clone().addScaledVector(direction, headLength / 2);
      const headBase = headCenter.clone().addScaledVector(direction, -headLength / 2);
      matrixAlongSegment(headBase, headTip, headRadius, matrix);
      this.arrowHeads.setMatrixAt(index, matrix);
    }
    this.arrowShafts.instanceMatrix.needsUpdate = true;
    this.arrowHeads.instanceMatrix.needsUpdate = true;
    this.arrowShafts.computeBoundingSphere();
    this.arrowHeads.computeBoundingSphere();
  }

  updateSurfaces() {
    for (let surfaceIndex = 0; surfaceIndex < this.model.surfaces.length; surfaceIndex += 1) {
      const surface = this.model.surfaces[surfaceIndex];
      const mesh = this.surfaceMeshes[surfaceIndex];
      const attribute = mesh.geometry.getAttribute("position");
      const center = new THREE.Vector3();
      for (const vertex of surface.cycle) center.add(this.projectedPositions[vertex]);
      center.multiplyScalar(1 / surface.cycle.length);
      attribute.setXYZ(0, center.x, center.y, center.z);
      surface.cycle.forEach((vertex, index) => {
        const position = this.projectedPositions[vertex];
        attribute.setXYZ(index + 1, position.x, position.y, position.z);
      });
      attribute.needsUpdate = true;
      mesh.geometry.computeVertexNormals();
      mesh.geometry.computeBoundingSphere();
    }
  }

  updateArrowLabels() {
    for (let index = 0; index < this.model.edges.length; index += 1) {
      const edge = this.model.edges[index];
      this.edgeLabels[index].position
        .addVectors(
          this.projectedPositions[edge.source],
          this.projectedPositions[edge.target],
        )
        .multiplyScalar(0.5);
    }
  }

  setSurfaceOpacity(opacity) {
    this.surfaceOpacity = opacity;
    for (const mesh of this.surfaceMeshes) mesh.material.opacity = opacity;
  }

  fitView() {
    if (this.projectedPositions.length === 0) {
      this.camera.position.set(3, 2.4, 4);
      this.controls.target.set(0, 0, 0);
      this.controls.update();
      return;
    }
    const bounds = new THREE.Box3().setFromPoints(this.projectedPositions);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 1);
    const verticalTangent = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const horizontalTangent = verticalTangent * Math.max(this.camera.aspect, 0.01);
    const distance = radius * Math.max(
      1 / verticalTangent,
      1 / horizontalTangent,
    ) * 1.22;
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(1.35, 1.05, 1.55).setLength(distance));
    this.camera.near = Math.max(radius / 100, 0.01);
    this.camera.far = Math.max(radius * 100, 100);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.hasFittedView = true;
  }

  resize() {
    const { width, height } = this.container.getBoundingClientRect();
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  pick(event) {
    if (!this.model) return;
    const bounds = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
    this.pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const intersection = this.raycaster.intersectObjects(this.pickables, false)[0];
    if (!intersection) {
      this.tooltip.hidden = true;
      return;
    }

    const { kind } = intersection.object.userData;
    const item = kind === "surface"
      ? intersection.object.userData.item
      : intersection.object.userData.items[intersection.instanceId];
    const basis = item.basis.length === 0 ? "{}" : `{${item.basis.join(",")}}`;
    const detail = kind === "edge" ? `, direction ${item.direction}` : "";
    this.tooltip.textContent = `${kind} (${item.dimension}, ${item.position}), basis ${basis}${detail}`;
    this.tooltip.hidden = false;
    this.tooltip.style.left = `${event.clientX - bounds.left + 12}px`;
    this.tooltip.style.top = `${event.clientY - bounds.top + 12}px`;
  }

  setAnimationLoop(callback) {
    this.renderer.setAnimationLoop((time) => {
      callback(time);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }
}
