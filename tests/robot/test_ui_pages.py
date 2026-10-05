"""The two pages and the server must agree: ids the script looks up exist, and the messages match both ways."""
import re

import pytest

from echotwin.robot import config

PAGES = ["dashboard.html", "phone.html"]
SRC = {p: (config.STATIC / p).read_text(encoding="utf-8") for p in PAGES}
SERVER_SIDE = "".join(p.read_text(encoding="utf-8") for p in (config.ROOT / "echotwin" / "robot").rglob("*.py"))


@pytest.mark.parametrize("page", PAGES)
def test_every_id_the_script_looks_up_exists(page):
    html = SRC[page]
    used = set(re.findall(r"\$\('#([\w-]+)'\)", html))
    have = set(re.findall(r'\bid="([\w-]+)"', html))
    assert used - have == set(), f"{page} looks up ids that are not in the page"


@pytest.mark.parametrize("page", PAGES)
def test_the_messages_a_page_sends_are_handled_by_the_server(page):
    sent = set(re.findall(r"send\(\{\s*t:\s*'(\w+)'", SRC[page]))
    handled = set(re.findall(r't == "(\w+)"', SERVER_SIDE))
    assert sent and sent - handled == set(), f"{page} sends messages the server ignores"


@pytest.mark.parametrize("page", PAGES)
def test_the_messages_a_page_handles_are_sent_by_the_server(page):
    handled = set(re.findall(r"case '(\w+)':", SRC[page])) | set(re.findall(r"m\.t === '(\w+)'", SRC[page]))
    emitted = set(re.findall(r'"t":\s*"(\w+)"', SERVER_SIDE))
    assert handled and handled - emitted == set(), f"{page} waits for messages nobody sends"


@pytest.mark.parametrize("page", PAGES)
def test_a_page_uses_the_shared_style_and_stage_script(page):
    assert 'href="/ui/ui.css"' in SRC[page] and 'src="/ui/stage.js"' in SRC[page]
