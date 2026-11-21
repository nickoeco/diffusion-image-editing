"""Opt-in local browser verification using a separately installed Playwright."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-demo", action="store_true")
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Inspect the UI without model inference")
    parser.add_argument("--check-sd-switch", action="store_true", help="Also verify cached real SD loading before and after FLUX")
    parser.add_argument("--url", default="http://127.0.0.1:7865")
    parser.add_argument("--tool-path", type=Path, default=ROOT / ".cache/flux-ui-tools")
    parser.add_argument("--browser-executable", type=Path,
                        default=Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe")
    args = parser.parse_args()
    if not args.run_demo or not args.probe and not args.run_gpu:
        parser.error("Requires --run-demo and either --probe or --run-gpu")
    if urlparse(args.url).hostname not in ("localhost", "127.0.0.1", "::1"):
        parser.error("Browser verification is restricted to a local demo")
    sys.path.insert(0, str(args.tool_path.resolve()))
    from playwright.sync_api import sync_playwright

    out = ROOT / "docs/results/flux-ui"
    out.mkdir(parents=True, exist_ok=True)
    observations: dict = {"url": args.url, "probe": args.probe, "progress_stages": [], "errors": []}
    client = None
    if args.check_sd_switch:
        if args.probe or not args.run_gpu:
            parser.error("SD switching requires the real GPU browser check")
        from gradio_client import Client
        client = Client(args.url, verbose=False)
        before = client.predict("a red apple on a wooden table", "", 64, 64, 3, 7.5, 42, api_name="/txt2img")
        observations["sd_before_flux"] = {key: before[4][key] for key in ("model_first_load", "config", "runtime")}
        print("REAL SD CACHE LOADED", flush=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(args.browser_executable), headless=True,
                                              args=["--disable-background-networking", "--no-first-run"])
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        context.route("**/*", lambda route: route.continue_() if urlparse(route.request.url).hostname in
                      (None, "localhost", "127.0.0.1", "::1") else route.abort())
        page = context.new_page()
        page.on("pageerror", lambda error: observations["errors"].append(str(error)))
        try:
            page.goto(args.url, wait_until="domcontentloaded")
            page.get_by_role("tab", name="FLUX", exact=True).wait_for(timeout=90000)
            observations["visible_tabs"] = page.get_by_role("tab").all_text_contents()
            page.locator("#workspace-edit-mask-source").get_by_text("Draw", exact=True).click()
            page.wait_for_timeout(1200)
            canvases = page.locator("#workspace-edit-paint canvas:visible")
            observations["visible_paint_canvases"] = canvases.count()
            print("CANVASES", [item.bounding_box() for item in canvases.all()], flush=True)
            if canvases.count():
                box = canvases.last.bounding_box()
                page.mouse.move(box["x"] + box["width"] * .65, box["y"] + box["height"] * .75)
                page.mouse.down()
                page.mouse.move(box["x"] + box["width"] * .85, box["y"] + box["height"] * .85, steps=12)
                page.mouse.up()
                page.wait_for_timeout(1800)
            page.screenshot(path=str(out / "paint-before-flux.png"), full_page=True)
            page.get_by_role("tab", name="FLUX", exact=True).click()
            page.locator("#flux-prompt textarea").wait_for(state="visible")
            page.screenshot(path=str(out / "initial-flux.png"), full_page=True)
            print("FLUX READY", page.locator("#flux-status textarea").input_value(), flush=True)
            if args.probe:
                return

            def wait_for_result(operation: str) -> None:
                deadline = time.monotonic() + 180
                captured = False
                while time.monotonic() < deadline:
                    status = page.locator("#flux-status textarea").input_value()
                    if status.startswith("Failed:"):
                        raise AssertionError(status)
                    if status.startswith(f"FLUX {operation.lower()} complete"):
                        observations[operation + " status"] = status
                        page.locator("#flux-result img").wait_for(state="visible")
                        page.wait_for_timeout(1000)
                        observations[operation + " render styles"] = page.locator("#flux-result img").evaluate(
                            "image => { let out=[];for(let n=image;n;n=n.parentElement){out.push({tag:n.tagName,classes:n.className,opacity:getComputedStyle(n).opacity});}return out; }")
                        return
                    text = page.locator("body").inner_text()
                    for phrase in ("Starting the isolated FLUX runtime", "Loading the INT8 text encoder",
                                   "Prompt encoded", "Loading the INT8 Transformer", "Sampling step"):
                        if phrase in text and phrase not in observations["progress_stages"]:
                            observations["progress_stages"].append(phrase)
                    if not captured and observations["progress_stages"]:
                        page.screenshot(path=str(out / "progress.png"), full_page=True)
                        captured = True
                    page.wait_for_timeout(200)
                raise TimeoutError("The browser did not observe a completed FLUX run within 180 seconds")

            page.locator("#flux-run").click()
            wait_for_result("Generate")
            page.locator("#flux-result img").wait_for(state="visible")
            page.screenshot(path=str(out / "generated.png"), full_page=True)
            observations["generation_image_size"] = page.locator("#flux-result img").evaluate(
                "image => [image.naturalWidth, image.naturalHeight]")
            old_src = page.locator("#flux-result img").get_attribute("src")
            page.locator("#flux-operation").get_by_text("Edit image", exact=True).click()
            page.locator("#flux-example").click()
            page.locator("#flux-reference img").wait_for(state="visible")
            page.wait_for_function("document.querySelector('#flux-sizing').textContent.includes('608 x 400')")
            observations["reference_fitting"] = page.locator("#flux-sizing").inner_text()
            page.get_by_text("Settings changed — showing previous result", exact=False).wait_for()
            assert page.locator("#flux-result img").get_attribute("src") == old_src
            page.screenshot(path=str(out / "edit-input.png"), full_page=True)
            page.locator("#flux-run").click()
            wait_for_result("Edit image")
            page.screenshot(path=str(out / "edited.png"), full_page=True)
            edit_src = page.locator("#flux-result img").get_attribute("src")
            links = page.locator("a[download]")
            observations["download_links"] = [{"text": item.inner_text(), "href": item.get_attribute("href")}
                                               for item in links.all() if item.is_visible()]
            if len(observations["download_links"]) < 3:
                raise AssertionError("Expected PNG, JSON and ZIP download links")
            for index, link in enumerate([item for item in links.all() if item.is_visible()]):
                with page.expect_download() as download:
                    link.click()
                file = download.value
                file.save_as(out / f"download-{index}-{file.suggested_filename}")
            page.locator("#flux-prompt textarea").fill("")
            page.locator("#flux-run").click()
            page.wait_for_function("document.querySelector('#flux-status textarea').value.startsWith('Failed:')")
            page.wait_for_timeout(1000)
            assert page.locator("#flux-result img").get_attribute("src") == edit_src
            observations["failure_retained_previous_result"] = True
            page.screenshot(path=str(out / "error-retention.png"), full_page=True)
            page.get_by_role("tab", name="Edit image", exact=True).click()
            page.locator("#workspace-edit-paint").wait_for(state="visible")
            page.screenshot(path=str(out / "paint-after-flux.png"), full_page=True)
            page.get_by_role("tab", name="FLUX", exact=True).click()
            assert page.locator("#flux-result img").get_attribute("src") == edit_src
            observations["tab_switch_retained_flux_result"] = True
            if client is not None:
                after = client.predict("a red apple on a wooden table", "", 64, 64, 3, 7.5, 42, api_name="/txt2img")
                assert after[4]["model_first_load"] is True
                observations["sd_after_flux"] = {key: after[4][key] for key in ("model_first_load", "config", "runtime")}
            print(json.dumps(observations, indent=2), flush=True)
        finally:
            page.screenshot(path=str(out / "last-page.png"), full_page=True)
            (out / "browser-observations.json").write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")
            browser.close()


if __name__ == "__main__":
    main()
