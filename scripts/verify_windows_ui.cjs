#!/usr/bin/env node
"use strict";

// This is an integration test: screenshots and font evidence come from the
// installed browser on the reported OS, never a Windows user-agent override.
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { parseArgs } = require("node:util");
const { chromium } = require("../tools/windows-ui/node_modules/playwright-core");

const { values } = parseArgs({ options: {
  browser: { type: "string", default: "msedge" },
  package: { type: "string", default: "dist/public-dashboard" },
  output: { type: "string", default: "build/windows-ui" },
  "base-url": { type: "string" },
  "allow-non-windows": { type: "boolean", default: false },
} });
const packageRoot = path.resolve(values.package);
const outputRoot = path.resolve(values.output);
const report = {
  schema: "regime-windows-ui/1", startedAt: new Date().toISOString(),
  actualWindows: process.platform === "win32", platform: process.platform,
  architecture: process.arch, osRelease: os.release(), osVersion: os.version(),
  nodeVersion: process.version, browserChannel: values.browser,
  sourceCommit: process.env.GITHUB_SHA || null,
  runnerOS: process.env.RUNNER_OS || null, runnerArch: process.env.RUNNER_ARCH || null,
  imageOS: process.env.ImageOS || null, imageVersion: process.env.ImageVersion || null,
  checks: [], screenshots: [], errors: [],
};
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
fs.mkdirSync(outputRoot, { recursive: true });

async function check(name, action) {
  try {
    const evidence = await action();
    report.checks.push({ name, status: "passed", evidence });
    console.log(`PASS ${name}`);
    return evidence;
  } catch (error) {
    report.checks.push({ name, status: "failed", error: error.message });
    throw error;
  }
}

async function staticServer() {
  const types = { ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8" };
  const server = http.createServer((request, response) => {
    const relative = decodeURIComponent(new URL(request.url, "http://localhost").pathname).replace(/^\/+/, "") || "index.html";
    const target = path.resolve(packageRoot, relative);
    const inside = path.relative(packageRoot, target);
    if (inside.startsWith("..") || path.isAbsolute(inside) || !fs.existsSync(target) || !fs.statSync(target).isFile()) {
      response.writeHead(404); response.end("Not found"); return;
    }
    response.writeHead(200, { "Content-Type": types[path.extname(target)] || "application/octet-stream", "Cache-Control": "no-store" });
    fs.createReadStream(target).pipe(response);
  });
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  return server;
}

async function settle(page) {
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  });
}

async function capture(page, name, selector) {
  const filename = `${name}.png`;
  if (selector) await page.locator(selector).screenshot({ path: path.join(outputRoot, filename), animations: "disabled" });
  else await page.screenshot({ path: path.join(outputRoot, filename), animations: "disabled" });
  report.screenshots.push(filename);
}

async function navigate(page, view) {
  await page.locator(`#dashboard-view-nav [data-dashboard-view="${view}"]`).click();
  await page.waitForFunction((expected) => document.querySelector("#dashboard").dataset.activeView === expected, view);
  await settle(page);
}

async function layout(page, width, theme) {
  await page.setViewportSize({ width, height: 1000 });
  if (await page.locator("html").getAttribute("data-theme") !== theme) {
    await page.locator("#theme-toggle").click();
  }
  await navigate(page, "transition");
  await page.locator("#duration-context dd").first().waitFor({ state: "visible" });
  await settle(page);
  return check(`duration cards ${width}px ${theme}`, async () => {
    const evidence = await page.evaluate(() => {
      const cards = [...document.querySelectorAll("#duration-context > div, #duration-baselines > div")].map((element) => {
        const style = getComputedStyle(element), rect = element.getBoundingClientRect();
        const framed = parseFloat(style.borderLeftWidth) > 0 && parseFloat(style.borderRightWidth) > 0;
        const content = [...element.children].map((child) => {
          const childRect = child.getBoundingClientRect();
          const range = document.createRange(); range.selectNodeContents(child);
          const textRects = [...range.getClientRects()].filter((item) => item.width > 0 && item.height > 0);
          return { text: child.textContent.trim(), width: childRect.width,
            textWithinCard: textRects.every((item) => item.left >= rect.left + (framed ? 8 : -1) && item.right <= rect.right - (framed ? 8 : -1) && item.top >= rect.top + (framed ? 6 : -1) && item.bottom <= rect.bottom - (framed ? 6 : -1)) };
        });
        return { framed, width: rect.width, height: rect.height,
          padding: [style.paddingTop, style.paddingRight, style.paddingBottom, style.paddingLeft].map(parseFloat),
          overflow: element.scrollWidth - element.clientWidth, content };
      });
      return { viewport: innerWidth, pageWidth: document.documentElement.scrollWidth,
        theme: document.documentElement.dataset.theme, cards };
    });
    assert.equal(evidence.theme, theme);
    assert.ok(evidence.pageWidth <= width + 1, `page overflow: ${evidence.pageWidth} > ${width}`);
    assert.ok(evidence.cards.length >= 7, `expected duration metrics and both baseline cards, found ${evidence.cards.length}`);
    for (const card of evidence.cards) {
      assert.ok(card.padding.every((value) => value >= (card.framed ? 14 : 0)), `insufficient padding: ${card.padding}`);
      assert.ok(card.overflow <= 1, `duration card overflow: ${card.overflow}`);
      assert.ok(card.content.every((child) => child.textWithinCard), `text touches card edge: ${JSON.stringify(card.content)}`);
    }
    await capture(page, `duration-${width}-${theme}`, "#duration-context-card");
    await capture(page, `transition-${width}-${theme}`);
    return evidence;
  });
}

async function platformFonts(page) {
  const session = await page.context().newCDPSession(page);
  await session.send("DOM.enable");
  await session.send("CSS.enable");
  const { root } = await session.send("DOM.getDocument");
  const results = [];
  // Both nodes contain Korean, so actual glyph evidence is stronger than a
  // computed font-family declaration or document.fonts.check alone.
  for (const selector of ["#page-title", "#duration-context dt"]) {
    const { nodeId } = await session.send("DOM.querySelector", { nodeId: root.nodeId, selector });
    assert.ok(nodeId, `font probe is missing: ${selector}`);
    const { fonts } = await session.send("CSS.getPlatformFontsForNode", { nodeId });
    assert.ok(fonts.some((font) => font.isCustomFont && /Pretendard/i.test(font.familyName) && font.glyphCount > 0), `Korean glyphs did not use bundled Pretendard: ${JSON.stringify(fonts)}`);
    assert.ok(fonts.filter((font) => font.glyphCount > 0).every((font) => font.isCustomFont && /Pretendard/i.test(font.familyName)), `unexpected fallback glyphs: ${JSON.stringify(fonts)}`);
    results.push({ selector, text: await page.locator(selector).first().textContent(), fonts });
  }
  await session.detach();
  return results;
}

async function main() {
  let server, browser, page;
  try {
    await check("actual Windows environment", () => {
      assert.ok(report.actualWindows || values["allow-non-windows"], "This check must run on actual Windows; use --allow-non-windows only for a labeled rehearsal.");
      assert.ok(["msedge", "chrome"].includes(values.browser), "Use installed Edge or Chrome.");
      if (report.actualWindows) report.windows = JSON.parse(execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", "Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture | ConvertTo-Json -Compress"], { encoding: "utf8" }));
      return { actualWindows: report.actualWindows, windows: report.windows || null };
    });
    const manifestBytes = fs.readFileSync(path.join(packageRoot, "publication-manifest.json"));
    report.publicationManifestSha256 = sha256(manifestBytes);
    if (values["base-url"]) {
      const requested = new URL(values["base-url"]);
      assert.equal(requested.href, "https://sonchanggi.github.io/regime/", "Remote checks are confined to the project's published dashboard.");
      report.baseUrl = requested.href;
    } else {
      server = await staticServer();
      report.baseUrl = `http://127.0.0.1:${server.address().port}/`;
    }
    await check("package manifest matches served release", async () => {
      const response = await fetch(new URL("publication-manifest.json", report.baseUrl));
      assert.equal(response.status, 200);
      const servedHash = sha256(Buffer.from(await response.arrayBuffer()));
      assert.equal(servedHash, report.publicationManifestSha256);
      return { sha256: servedHash };
    });
    browser = await chromium.launch({ channel: values.browser, headless: true });
    report.browserVersion = browser.version();
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1, locale: "ko-KR", colorScheme: "light", reducedMotion: "reduce" });
    page = await context.newPage();
    const pageErrors = [], failedResponses = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    page.on("response", (response) => {
      if (response.status() >= 400 && new URL(response.url()).origin === new URL(report.baseUrl).origin) failedResponses.push({ path: new URL(response.url()).pathname, status: response.status() });
    });
    await page.goto(report.baseUrl, { waitUntil: "domcontentloaded" });
    await page.locator("#dashboard").waitFor({ state: "visible", timeout: 90000 });
    await page.locator("#enhancement-model").waitFor({ state: "visible", timeout: 90000 });
    await settle(page);
    report.browserEnvironment = await page.evaluate(() => ({ userAgent: navigator.userAgent, platform: navigator.platform,
      devicePixelRatio, fonts: [...document.fonts].map((face) => ({ family: face.family, status: face.status, weight: face.weight })),
      selectedWeek: document.querySelector("#week-select").value }));
    await check("bundled webfont loaded", async () => {
      const loaded = await page.evaluate(() => document.fonts.check('16px "Pretendard Variable"', "현재 국면 지속"));
      assert.ok(loaded);
      assert.ok(report.browserEnvironment.fonts.some((face) => /Pretendard/.test(face.family) && face.status === "loaded"));
      return report.browserEnvironment.fonts;
    });
    await capture(page, "overview-1440-light");
    for (const width of [1440, 1280, 390, 320]) await layout(page, width, "light");
    for (const width of [1440, 390]) await layout(page, width, "dark");
    await check("actual Korean glyph font", () => platformFonts(page));
    await page.setViewportSize({ width: 1440, height: 1000 });
    if (await page.locator("html").getAttribute("data-theme") !== "light") await page.locator("#theme-toggle").click();

    await check("week control changes selected analysis", async () => {
      const before = await page.locator("#week-select").inputValue();
      await page.locator("#previous-week").click();
      await page.waitForFunction((previous) => document.querySelector("#week-select").value !== previous && document.querySelector("#forecast-enhancements").dataset.origin === document.querySelector("#week-select").value, before);
      const after = await page.locator("#week-select").inputValue();
      assert.notEqual(after, before);
      assert.equal(new URL(page.url()).searchParams.get("week"), after);
      await page.locator("#latest-week").click();
      await page.waitForFunction((latest) => document.querySelector("#week-select").value === latest, before);
      return { before, after };
    });
    await check("forecast horizon changes rendered results", async () => {
      const initialModel = await page.locator("#enhancement-model").inputValue();
      const before = await page.locator(".enhancement-primary").innerText();
      await page.locator("#enhancement-horizon").selectOption("4");
      await page.waitForFunction(() => document.querySelector("#forecast-enhancements").dataset.horizon === "4");
      const after = await page.locator(".enhancement-primary").innerText();
      assert.notEqual(before, after);
      assert.equal(new URL(page.url()).searchParams.get("forecast_horizon"), "4");
      await page.locator("#enhancement-horizon").selectOption("1");
      await page.waitForFunction(() => document.querySelector("#forecast-enhancements").dataset.horizon === "1");
      // Horizon changes can select a different compatible model. Restore the
      // initial model so the asset check measures the reviewed operating model.
      await page.locator("#enhancement-model").selectOption(initialModel);
      await page.waitForFunction((expected) => document.querySelector("#forecast-enhancements").dataset.model === expected, initialModel);
      return { initialModel, before, after };
    });
    await navigate(page, "performance");
    await page.locator("#decision-action-card").waitFor({ state: "visible" });
    await check("allocation summary retains usable column widths", async () => {
      const evidence = [];
      for (const width of [1440, 390, 320]) {
        await page.setViewportSize({ width, height: 1000 }); await settle(page);
        const dimensions = await page.locator("#decision-shadow-current-summary").evaluate((element) => ({
          summaryWidth: element.getBoundingClientRect().width, pageWidth: document.documentElement.scrollWidth,
          flowColumns: getComputedStyle(element.querySelector(".decision-allocation-flow")).gridTemplateColumns,
          items: [...element.querySelectorAll(".decision-allocation-step")].map((child) => ({
            width: child.getBoundingClientRect().width, overflow: child.scrollWidth - child.clientWidth, text: child.textContent.trim() })) }));
        assert.ok(dimensions.summaryWidth >= 200, `allocation summary collapsed: ${dimensions.summaryWidth}`);
        assert.ok(dimensions.items.length >= 4);
        assert.ok(dimensions.items.every((item) => item.width >= 100 && item.overflow <= 1), JSON.stringify(dimensions));
        assert.ok(dimensions.pageWidth <= width + 1, `allocation page overflow: ${dimensions.pageWidth} > ${width}`);
        await capture(page, `allocation-${width}-light`, "#decision-action-card");
        evidence.push({ viewport: width, ...dimensions });
      }
      return evidence;
    });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await navigate(page, "assets");
    await check("asset controls change analysis", async () => {
      await page.locator("#conditional-stat-grid").waitFor({ state: "visible" });
      const before = await page.locator("#conditional-stat-grid").textContent();
      report.assetControlBefore = { url: page.url(), model: await page.locator("#forecast-enhancements").getAttribute("data-model"),
        horizon: await page.locator("#conditional-horizon-select").inputValue(),
        caption: await page.locator("#conditional-stats-caption").textContent(), results: before };
      await page.locator("#conditional-horizon-select").selectOption("4");
      await page.waitForFunction((previous) => document.querySelector("#conditional-stat-grid").textContent !== previous && document.querySelector("#conditional-horizon-select").value === "4" && document.querySelector("#conditional-stats-caption").textContent.includes("4주"), before);
      const after = await page.locator("#conditional-stat-grid").textContent();
      assert.notEqual(before, after);
      assert.equal(new URL(page.url()).searchParams.get("horizon"), "4");
      await capture(page, "assets-1440-light");
      return { before, after };
    });
    await navigate(page, "model");
    await check("model view renders", async () => {
      await page.locator("#model-forecast-explorer").waitFor({ state: "visible" });
      const text = await page.locator("#model-forecast-explorer").innerText();
      assert.ok(text.length > 30);
      await capture(page, "model-1440-light");
      return { text };
    });
    await navigate(page, "execution");
    await check("browser has no application or resource errors", () => {
      assert.deepEqual(pageErrors, []);
      assert.deepEqual(failedResponses, []);
      return { pageErrors, failedResponses };
    });
    report.status = "passed";
  } catch (error) {
    report.status = "failed";
    report.errors.push(error.stack || error.message);
    console.error(error.stack || error.message);
    if (page && !page.isClosed()) {
      try {
        report.failureState = await page.evaluate(() => ({ url: location.href,
          activeView: document.querySelector("#dashboard")?.dataset.activeView,
          model: document.querySelector("#forecast-enhancements")?.dataset.model,
          forecastHorizon: document.querySelector("#forecast-enhancements")?.dataset.horizon,
          assetHorizon: document.querySelector("#conditional-horizon-select")?.value,
          assetCaption: document.querySelector("#conditional-stats-caption")?.textContent,
          assetResultsHidden: document.querySelector("#conditional-results")?.hidden,
          assetUnavailable: document.querySelector("#conditional-unavailable")?.textContent,
          assetResults: document.querySelector("#conditional-stat-grid")?.textContent }));
      } catch (stateError) { report.errors.push(`Failure state: ${stateError.message}`); }
      try { await capture(page, "failure"); }
      catch (captureError) { report.errors.push(`Failure screenshot: ${captureError.message}`); }
    }
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    if (server) await new Promise((resolve) => server.close(resolve));
    report.finishedAt = new Date().toISOString();
    fs.writeFileSync(path.join(outputRoot, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
    console.log(`Report: ${path.join(outputRoot, "report.json")}`);
  }
}
main();
