#!/usr/bin/env python3
"""Headless-browser smoke test for the running viewer.

This harness browser cannot reach cdn.jsdelivr.net (connection reset from the
Chromium sandbox, while curl works), so the three.js ES-module requests are
fulfilled from the local copies in outputs/vendor_three. The project source is
NOT modified.

Usage: python ui_smoke_test.py [url] [screenshot.png]
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "ui_screenshot.png")
VENDOR = Path(__file__).resolve().parent / "vendor_three"
CDN_PREFIX = "https://cdn.jsdelivr.net/npm/three@0.169.0/"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=[
            "--use-gl=angle", "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader", "--no-sandbox",
        ])
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        served = []

        async def route_cdn(route):
            rel = route.request.url[len(CDN_PREFIX):]
            local = VENDOR / rel
            if local.is_file():
                served.append(rel)
                await route.fulfill(path=str(local), content_type="application/javascript; charset=utf-8")
            else:
                await route.abort()

        await page.route(CDN_PREFIX + "**", route_cdn)
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        await page.goto(URL, wait_until="load", timeout=60000)
        status = ""
        for _ in range(45):
            await page.wait_for_timeout(1000)
            status = await page.eval_on_selector("#status", "el => el.textContent")
            if "已加载" in status or "失败" in status or "未找到" in status:
                break
        await page.wait_for_timeout(3000)
        gl = await page.evaluate("""() => {
            const c = document.createElement('canvas');
            const g = c.getContext('webgl2') || c.getContext('webgl');
            return g ? g.getParameter(g.VERSION) : 'no-webgl';
        }""")
        info = await page.evaluate("""() => ({
            canvas: (() => { const r = document.querySelector('#viewer').getBoundingClientRect();
                             return [Math.round(r.width), Math.round(r.height)]; })(),
            options: [...document.querySelectorAll('#gender option')].map(o => o.textContent),
            title: document.title,
        })""")
        await page.screenshot(path=str(OUT))
        print("WEBGL :", gl)
        print("PAGE  :", info)
        print("STATUS:", status)
        print("CDN->local:", served)
        print("ERRORS:", errors if errors else "none")
        await browser.close()


asyncio.run(main())
