"""Parse the JPAC offline HTML (WB-DSG) into Postgres: entries, scenarios, sections, links.

Usage: python -m semsearch.load_jpac
Drops and recreates the tables from db/schema.sql on every run.
"""
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import psycopg
from bs4 import BeautifulSoup

WB_DIR = Path("data/jpac/site/preview.jpac.org.uk/guidelines/wb")
BASE_URL = "https://www.jpac.org.uk/guidelines/wb/"
KINDS = {
    "also known as": "also_known_as", "definition/s": "definitions", "includes": "includes",
    "excludes": "excludes", "obligatory": "obligatory", "discretionary": "discretionary",
    "see if relevant": "see_if_relevant", "additional information": "additional_information",
    "regulatory information": "regulatory_information", "post donation illness": "post_donation_illness",
}
BLOCKS = ["p", "li", "tr", "dt", "dd", "div", "details", "summary", "ol", "ul", "table",
          "h1", "h2", "h3", "h4", "h5", "h6"]


def to_text(html):
    """HTML fragment -> plain text that keeps list numbering, bullets and table rows."""
    el = BeautifulSoup(html, "html.parser")
    for s in el.find_all(string=True):
        s.replace_with(re.sub(r"\s+", " ", s))
    for ol in el.find_all("ol"):
        for i, li in enumerate(ol.find_all("li", recursive=False), 1):
            li.insert(0, f"{i}. ")
    for ul in el.find_all("ul"):
        for li in ul.find_all("li", recursive=False):
            li.insert(0, "- ")
    for cell in el.find_all(["td", "th"]):
        cell.append(" | ")
    for br in el.find_all("br"):
        br.replace_with("\n")
    for b in el.find_all(BLOCKS):
        b.append("\n")
    lines = (line.strip().rstrip("|").strip() for line in el.get_text().split("\n"))
    return "\n".join(line for line in lines if line)


def classify(url, entry_slugs):
    """Absolute URL -> (target_kind, entry slug or None, url or None)."""
    url = url.replace("://preview.jpac.org.uk", "://www.jpac.org.uk").removesuffix("index.html")
    if url == BASE_URL + "index/":
        return "a_to_z", None, None
    m = re.fullmatch(re.escape(BASE_URL) + r"index/([^/]+)/", url)
    if m and m.group(1) in entry_slugs:
        return "entry", m.group(1), None
    return "external", None, url


def context_of(a, dd):
    """Nearest preceding all-bold paragraph in the same section, e.g. 'For hair transplants:'."""
    for p in a.find_all_previous("p"):
        if not any(x is dd for x in p.parents):
            return None
        b = p.find("b")
        if b and p.get_text(strip=True) == b.get_text(strip=True):
            return p.get_text(" ", strip=True)
    return None


def parse_page(path, slug):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    content = soup.select_one("main .app-section__content")
    h1 = content.find("h1")
    entry = {"slug": slug, "url": f"{BASE_URL}index/{slug}/", "title": next(h1.strings).strip(),
             "release": None, "version_date": None, "version_details": None,
             "reason_for_change": None, "change_notification_url": None}

    meta = content.select_one("dl.inline")
    if meta:
        for row in meta.find_all("div", recursive=False):
            label = row.find("dt").get_text(strip=True).rstrip(":").lower()
            dd = row.find("dd")
            if label == "reason for change":
                entry["reason_for_change"] = to_text(dd.decode_contents())
            elif label == "version" and dd.find("a"):
                entry["change_notification_url"] = dd.find("a")["href"]
            elif label == "version details":
                entry["version_details"] = dd.get_text(" ", strip=True)
                # e.g. 'Release 77 Issue 02 (18 July 2025)'; one page says '(Date)', so parse separately
                if m := re.search(r"Release (\d+)", entry["version_details"]):
                    entry["release"] = int(m.group(1))
                if m := re.search(r"\((\d{1,2} \w+ \d{4})\)", entry["version_details"]):
                    entry["version_date"] = datetime.strptime(m.group(1), "%d %B %Y").date()
        meta.decompose()

    scenarios, sections = [], []  # sections: (scenario title or None, kind, dd tag)
    for dl in content.select("dl.nhsuk-summary-list"):
        if dl.find_parent("dl"):
            continue
        details = dl.find_parent("details")
        scenario = details.summary.get_text(" ", strip=True) if details else None
        if scenario and scenario not in scenarios:
            scenarios.append(scenario)
        for row in dl.find_all("div", class_="nhsuk-summary-list__row", recursive=False):
            label = row.find("dt").get_text(" ", strip=True).lower()
            if label not in KINDS:
                raise ValueError(f"{slug}: unknown section label {label!r}")
            sections.append((scenario, KINDS[label], row.find("dd")))
    return entry, scenarios, sections


def connect(**kwargs):
    """Connect to the Docker Postgres using the settings in .env."""
    pairs = (line.split("=", 1) for line in Path(".env").read_text().splitlines() if "=" in line)
    env = {k.strip(): v.strip() for k, v in pairs}
    return psycopg.connect(f"host=localhost port={env.get('POSTGRES_PORT', 5432)} dbname={env['POSTGRES_DB']} "
                           f"user={env['POSTGRES_USER']} password={env['POSTGRES_PASSWORD']}", **kwargs)


def main():
    pages = [(p, p.parent.name) for p in sorted((WB_DIR / "index").glob("*/index.html"))]
    parsed = [parse_page(*p) for p in pages]
    entry_slugs = {e["slug"] for e, _, _ in parsed}

    with connect() as conn, conn.cursor() as cur:
        cur.execute(Path("db/schema.sql").read_text())
        ids = {}
        for e, _, _ in parsed:
            cur.execute(
                "INSERT INTO entries (slug, title, url, release, version_date, version_details,"
                " reason_for_change, change_notification_url) VALUES (%(slug)s, %(title)s, %(url)s,"
                " %(release)s, %(version_date)s, %(version_details)s, %(reason_for_change)s,"
                " %(change_notification_url)s) RETURNING entry_id", e)
            ids[e["slug"]] = cur.fetchone()[0]

        for e, scenarios, sections in parsed:
            entry_id = ids[e["slug"]]
            scenario_ids = {}
            for pos, title in enumerate(scenarios, 1):
                cur.execute("INSERT INTO scenarios (entry_id, position, title) VALUES (%s, %s, %s)"
                            " RETURNING scenario_id", (entry_id, pos, title))
                scenario_ids[title] = cur.fetchone()[0]
            for pos, (scenario, kind, dd) in enumerate(sections, 1):
                html = dd.decode_contents().strip()
                cur.execute("INSERT INTO sections (entry_id, scenario_id, kind, position, text, html)"
                            " VALUES (%s, %s, %s, %s, %s, %s) RETURNING section_id",
                            (entry_id, scenario_ids.get(scenario), kind, pos, to_text(html), html))
                section_id = cur.fetchone()[0]
                for a in dd.find_all("a", href=True):
                    target_kind, slug, url = classify(urljoin(e["url"], a["href"]), entry_slugs)
                    cur.execute("INSERT INTO links (from_section_id, target_kind, to_entry_id, to_url,"
                                " anchor_text, context) VALUES (%s, %s, %s, %s, %s, %s)",
                                (section_id, target_kind, ids.get(slug), url,
                                 a.get_text(" ", strip=True), context_of(a, dd)))

        for table in ["entries", "scenarios", "sections", "links"]:
            cur.execute(f"SELECT count(*) FROM {table}")
            print(f"{table:10} {cur.fetchone()[0]}")


if __name__ == "__main__":
    main()
