import asyncio
import os
from pathlib import Path

from playwright.async_api import Page, async_playwright

from metrics.metrics import metrics_counter_inc
from neo.utils import logger

_BEDROCK_PROJECT = os.environ.get("BEDROCK_PROJECT", "")


def is_bedrock_env() -> bool:
    return _BEDROCK_PROJECT != ""


async def handle_new_page(page: Page):
    """
    Handle new page events and execute custom logic
    """
    print(f"New page created: {page.url}")


async def launch_chrome_debug(use_chrome_channel: bool = False, headless: bool = False):
    """
    Launch Chrome browser with remote debugging enabled on port 9222
    Returns the browser instance when launched successfully
    """
    try:
        extension_path = Path(os.path.dirname(__file__)).joinpath("browser_extension/error_capture")  # type: ignore
        playwright = await async_playwright().start()
        disable_security_args = [
            "--disable-web-security",
            "--disable-site-isolation-trials",
            "--disable-features=IsolateOrigins,site-per-process",
        ]
        workspace = "/workspace" if is_bedrock_env() else "./workspace"
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=os.path.join(workspace, "browser", "user_data"),
            headless=headless,
            viewport={"width": 1280, "height": 720},
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--disable-background-timer-throttling",
                "--disable-popup-blocking",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-window-activation",
                "--disable-focus-on-load",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-position=0,0",
            ]
            + disable_security_args
            + [
                f"--disable-extensions-except={extension_path}",
                f"--load-extension={extension_path}",
                "--disable-web-security",
                "--disable-site-isolation-trials",
                "--remote-debugging-port=9222",
            ],
            channel="chromium" if not use_chrome_channel else "chrome",
            # proxy={"server": "http://data-capture-online.xaminim.com:3160", "username": "default-user", "password": "default"},
        )
        metrics_counter_inc("agent_browser_launch", {"status": "success"})

        # 监听新页面事件
        context.on("page", handle_new_page)

        # 处理已经打开的页面
        for page in context.pages:
            await handle_new_page(page)

        # Keep browser process alive
        while True:
            await asyncio.sleep(1000)

    except Exception as e:
        logger.exception(f"Failed to launch Chrome browser: {str(e)}")
        metrics_counter_inc("agent_browser_launch", {"status": "failed"})
        raise


if __name__ == "__main__":
    asyncio.run(launch_chrome_debug())
