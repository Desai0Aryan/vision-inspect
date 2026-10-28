"""Browser smoke test + screenshots of every dashboard page (needs the API serving the built UI on :8000).

    make ui-build && make serve        # in one terminal
    python scripts/ui_smoke.py         # in another; fails on any console error
"""
import asyncio
import sys
from playwright.async_api import async_playwright
S = "docs/screenshots"  # output folder
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1360, "height": 900})
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error","warning") else None)
        pg.on("pageerror", lambda e: logs.append(f"PAGEERROR {e}"))
        await pg.goto("http://localhost:8000/#inspect"); await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{S}/inspect-idle.png")
        # stride every 3rd for speed
        await pg.get_by_role("button", name="Every 3rd").click()
        await pg.get_by_role("button", name="Run the demo conveyor").click()
        await pg.wait_for_timeout(5000)
        await pg.screenshot(path=f"{S}/inspect-running.png")
        await pg.wait_for_selector("text=Defects found", timeout=120000); await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{S}/inspect-results.png", full_page=True)
        await pg.goto("http://localhost:8000/#stills"); await pg.wait_for_timeout(3500)
        await pg.screenshot(path=f"{S}/single-part.png")
        await pg.goto("http://localhost:8000/#models"); await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{S}/models.png", full_page=True)
        await pg.goto("http://localhost:8000/#live"); await pg.wait_for_timeout(4000)
        await pg.screenshot(path=f"{S}/live.png")
        await b.close()
        print("\n".join(logs) or "no console errors")
        sys.exit(1 if logs else 0)
asyncio.run(main())
