"""The in-app guide page: a small Markdown renderer and the route that serves docs/TEACHING.md."""
from echotwin.robot import docpage


def test_renders_headings_lists_tables_code_and_inline_styles():
    md = ("# Title\n\nSome **bold**, *italic*, `code` and a [link](/x).\n\n- one\n- two\n\n1. first\n2. second\n\n"
          "| A | B |\n|---|---|\n| 1 | 2 |\n\n```\nprint('hi')\n```\n")
    h = docpage.render(md)
    assert "<h1>Title</h1>" in h and "<strong>bold</strong>" in h and "<em>italic</em>" in h
    assert "<code>code</code>" in h and '<a href="/x">link</a>' in h
    assert "<ul><li>one</li><li>two</li></ul>" in h and "<ol><li>first</li><li>second</li></ol>" in h
    assert "<th>A</th>" in h and "<td>2</td>" in h and "<pre><code>print(&#x27;hi&#x27;)</code></pre>" in h


def test_html_in_the_text_is_escaped():
    h = docpage.render("Hello <script>alert(1)</script> & [x](javascript:alert(1))")
    assert "<script>" not in h and "&lt;script&gt;" in h and "&amp;" in h


def test_the_teaching_guide_renders_completely():
    from echotwin.robot import config
    md = (config.ROOT / "docs" / "TEACHING.md").read_text(encoding="utf-8")
    h = docpage.render(md)
    assert h.count("<table>") == 3 and "How to film" in h and "<h2>3. Let it practise</h2>" in h
    assert "|" not in h.replace("&", "")           # no raw table rows left over


def test_the_server_serves_only_listed_guides():
    from fastapi.testclient import TestClient
    from echotwin.robot import server
    c = TestClient(server.app)
    r = c.get("/docs/teaching")
    assert r.status_code == 200 and "Teaching the robot" in r.text and "text/html" in r.headers["content-type"]
    assert c.get("/docs/secrets").status_code == 404
    assert c.get("/docs/..%2FREADME").status_code in (404, 422)
