import { expect, test } from "@playwright/test";

async function canvasPixelMetrics(canvas) {
  return canvas.evaluate((element) => {
    const gl = element.getContext("webgl2") ?? element.getContext("webgl");
    if (!gl) return { width: 0, height: 0, nonBackground: 0, range: 0 };

    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const background = [pixels[0], pixels[1], pixels[2]];
    let nonBackground = 0;
    let darkest = 255;
    let lightest = 0;
    for (let pixel = 0; pixel < width * height; pixel += 16) {
      const offset = pixel * 4;
      const difference = Math.abs(pixels[offset] - background[0])
        + Math.abs(pixels[offset + 1] - background[1])
        + Math.abs(pixels[offset + 2] - background[2]);
      if (difference > 18) nonBackground += 1;
      const luminance = pixels[offset] + pixels[offset + 1] + pixels[offset + 2];
      darkest = Math.min(darkest, luminance / 3);
      lightest = Math.max(lightest, luminance / 3);
    }
    return { width, height, nonBackground, range: lightest - darkest };
  });
}

test("renders and interacts with cubical OFPs", async ({ page }, testInfo) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  await page.goto("/?test=1");
  await expect(page.locator("html")).toHaveAttribute("data-ready", "true");
  await expect(page.locator("#status")).toHaveText("Ready");
  await expect(page.locator("#point-stat")).toHaveText("8");
  await expect(page.locator("#edge-stat")).toHaveText("12");
  await expect(page.locator("#surface-stat")).toHaveText("6");
  await expect(page.locator("#triangular-angle-force")).toHaveValue("0.01");
  await expect(page.locator("#triangular-angle-force-out")).toHaveText("0.010");
  const randomProjection = page.getByRole("button", { name: "Randomize projected" });
  await expect(randomProjection).toBeDisabled();
  const directionTwo = page.locator('.projection-direction[data-direction="2"]');
  await expect(directionTwo.getByRole("combobox")).toHaveValue("z");

  const canvas = page.locator("canvas");
  await expect(canvas).toBeVisible();
  await page.waitForTimeout(350);
  const metrics = await canvasPixelMetrics(canvas);
  expect(metrics.width).toBeGreaterThan(200);
  expect(metrics.height).toBeGreaterThan(200);
  expect(metrics.nonBackground).toBeGreaterThan(100);
  expect(metrics.range).toBeGreaterThan(30);
  await canvas.screenshot({ path: testInfo.outputPath("standard-3-cube.png") });
  await page.screenshot({ path: testInfo.outputPath("viewer-page.png"), fullPage: true });

  await page.getByRole("button", { name: "Pause" }).click();
  await page.waitForTimeout(100);
  const pausedFrame = await page.locator("#frame-stat").textContent();
  await page.waitForTimeout(150);
  await expect(page.locator("#frame-stat")).toHaveText(pausedFrame);

  await directionTwo.getByRole("combobox").selectOption("projected");
  await expect(randomProjection).toBeEnabled();
  await expect(directionTwo.locator(".projection-settings")).toBeVisible();
  const beforeProjection = await canvas.screenshot();
  await page.locator("#projection-2-azimuth").evaluate((input) => {
    input.value = "35";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.waitForTimeout(100);
  const afterProjection = await canvas.screenshot();
  expect(Buffer.compare(beforeProjection, afterProjection)).not.toBe(0);
  await page.screenshot({ path: testInfo.outputPath("manual-projection.png"), fullPage: true });

  const beforeOrbit = await canvas.screenshot();
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.38, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(150);
  const afterOrbit = await canvas.screenshot();
  expect(Buffer.compare(beforeOrbit, afterOrbit)).not.toBe(0);

  await page.getByRole("button", { name: "Restart" }).click();
  await expect(page.locator("#frame-stat")).toHaveText("Frame 0");
  await page.waitForTimeout(120);
  await expect(page.locator("#frame-stat")).toHaveText("Frame 0");

  await page.getByRole("button", { name: "Play" }).click();
  await page.waitForTimeout(120);
  await expect(page.locator("#frame-stat")).not.toHaveText("Frame 0");

  await page.locator("#cube-dimension").selectOption("4");
  await page.getByRole("button", { name: "Load cube" }).click();
  await expect(page.locator("#dimension-stat")).toHaveText("4");
  await expect(page.locator("#point-stat")).toHaveText("16");
  await expect(page.locator("#edge-stat")).toHaveText("32");
  await expect(page.locator("#surface-stat")).toHaveText("24");
  await expect(randomProjection).toBeEnabled();
  await page.waitForTimeout(250);
  expect((await canvasPixelMetrics(canvas)).nonBackground).toBeGreaterThan(100);
  await canvas.screenshot({ path: testInfo.outputPath("standard-4-cube.png") });

  expect(errors).toEqual([]);
});
