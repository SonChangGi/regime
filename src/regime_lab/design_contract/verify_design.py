"""Portable source/artifact guard. Vendor together with the identical contract.json."""

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

CONTRACT = json.loads(Path(__file__).with_name("contract.json").read_text())


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.in_head = False

    def handle_starttag(self, tag, attrs):
        if tag == "head":
            self.in_head = True
        if tag in {"script", "link"}:
            self.tags.append((tag, dict(attrs), self.in_head))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag == "head":
            self.in_head = False


def verify_html(html, project, navigation="static"):
    def require(ok, message):
        if not ok:
            raise ValueError("Design contract: " + message)

    require(project in CONTRACT["projects"], "unknown project")
    require(navigation in {"static", "react"}, "unknown navigation owner")
    parser = Tags()
    parser.feed(html)
    tags = parser.tags
    sheets = [
        (i, a)
        for i, (t, a, _) in enumerate(tags)
        if t == "link" and "data-quant-first-paint-css" in a
    ]
    starts = [
        (i, a)
        for i, (t, a, _) in enumerate(tags)
        if t == "script"
        and ("data-quant-design-project" in a or "/shared/design/startup/" in a.get("src", ""))
    ]
    require(len(sheets) == len(starts) == 1, "one first-paint stylesheet and one startup required")
    css_index, sheet = sheets[0]
    start_index, startup = starts[0]
    base = CONTRACT["startupBase"]
    require(
        sheet.get("href", "").split("?")[0] == base + project + ".css"
        and sheet.get("rel") == "stylesheet",
        "wrong stylesheet pin",
    )
    require(
        startup.get("src", "").split("?")[0] == base + "start.js"
        and startup.get("data-quant-design-project") == project,
        "wrong startup pin or project",
    )
    require(
        not any(a in startup for a in ("async", "defer"))
        and startup.get("type", "") in {"", "text/javascript", "application/javascript"},
        "startup must be blocking",
    )
    require(
        tags[start_index][2] and tags[css_index][2] and start_index == css_index + 1,
        "stylesheet must immediately precede startup in head",
    )
    navs = []
    for index, (tag, attrs, in_head) in enumerate(tags):
        src = attrs.get("src", "") if tag == "script" else attrs.get("href", "")
        plain = src.split("?")[0]
        if tag == "link" and attrs.get("rel") == "stylesheet":
            require(index <= css_index, "native stylesheet follows final common stylesheet")
        if not src:
            continue
        require(
            "/shared/design/releases/" not in plain
            and not re.search(r"(?:^|/)(?:quant-range|number|runtime)\.js$", plain),
            "direct or legacy design loader",
        )
        if tag != "script":
            continue
        if plain == CONTRACT["navigationUrl"]:
            navs.append(index)
            require(
                index > start_index and "defer" in attrs and "async" not in attrs,
                "navigation must defer after startup",
            )
            continue
        if index == start_index:
            continue
        name = urlsplit(plain).path.rsplit("/", 1)[-1]
        theme = name in CONTRACT["earlyThemeBasenames"]
        framework = (
            navigation == "react"
            and attrs.get("type") == "module"
            and re.fullmatch(CONTRACT["earlyFrameworkPattern"], name)
        )
        require(index > start_index or theme or framework, "application script precedes startup")
        if in_head and not theme and not framework:
            require(
                "defer" in attrs
                or attrs.get("type") == "module"
                or (navigation == "react" and "async" in attrs),
                "head application must defer",
            )
        if navs and not framework:
            require(index < navs[0], "application script follows navigation owner")
    require(len(navs) == (1 if navigation == "static" else 0), "navigation owner count mismatch")
    return {
        "contract": CONTRACT["version"],
        "project": project,
        "ui": CONTRACT["uiVersion"],
        "startup": CONTRACT["startupVersion"],
        "navigation": navigation,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--html", type=Path, required=True)
    cli.add_argument("--project", required=True)
    cli.add_argument("--navigation", choices=["static", "react"], default="static")
    args = cli.parse_args()
    print(json.dumps(verify_html(args.html.read_text(), args.project, args.navigation)))
