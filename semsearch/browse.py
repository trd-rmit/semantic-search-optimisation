"""Local website to inspect the JPAC tables: a tree of entries -> scenarios -> sections, read live from
Postgres, shown next to the original JPAC page.

Usage: python -m semsearch.browse   then open http://localhost:8000
"""
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from psycopg.rows import dict_row

from semsearch.load_jpac import connect

PAGE = Path(__file__).with_name("browse.html")
SITE = Path("data/jpac/site").resolve()


def query(sql, *params):
    with connect(row_factory=dict_row) as conn:
        return conn.execute(sql, params).fetchall()


def tree():
    return {
        "counts": query("SELECT (SELECT count(*) FROM entries) AS entries, (SELECT count(*) FROM scenarios) AS scenarios,"
                        " (SELECT count(*) FROM sections) AS sections, (SELECT count(*) FROM links) AS links,"
                        " (SELECT count(*) FROM chunks) AS chunks")[0],
        "entries": query("SELECT entry_id, slug, title FROM entries ORDER BY lower(title)"),
        "scenarios": query("SELECT scenario_id, entry_id, position, title FROM scenarios ORDER BY entry_id, position"),
        "sections": query("SELECT section_id, entry_id, scenario_id, kind, position FROM sections ORDER BY entry_id, position"),
    }


def entry(entry_id):
    rows = query("SELECT * FROM entries WHERE entry_id = %s", entry_id)
    if not rows:
        return None
    return {
        "entry": rows[0],
        "scenarios": query("SELECT * FROM scenarios WHERE entry_id = %s ORDER BY position", entry_id),
        "sections": query("SELECT * FROM sections WHERE entry_id = %s ORDER BY position", entry_id),
        "links": query("SELECT l.*, t.title AS to_title FROM links l JOIN sections s ON s.section_id = l.from_section_id"
                       " LEFT JOIN entries t ON t.entry_id = l.to_entry_id WHERE s.entry_id = %s ORDER BY l.link_id",
                       entry_id),
        "chunks": query("SELECT c.chunk_id, c.scenario_id, c.chars, c.text, array_agg(cs.section_id ORDER BY cs.section_id)"
                        " AS section_ids FROM chunks c JOIN chunk_sections cs USING (chunk_id)"
                        " LEFT JOIN scenarios sc USING (scenario_id) WHERE c.entry_id = %s"
                        " GROUP BY c.chunk_id, sc.position ORDER BY sc.position NULLS FIRST", entry_id),
        "incoming": query("SELECT f.entry_id, f.title, s.kind AS in_section, count(*) AS n FROM links l"
                          " JOIN sections s ON s.section_id = l.from_section_id JOIN entries f ON f.entry_id = s.entry_id"
                          " WHERE l.to_entry_id = %s GROUP BY 1, 2, 3 ORDER BY lower(f.title)", entry_id),
    }


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data):
        if data is None:
            return self.send(404, b"not found", "text/plain")
        self.send(200, json.dumps(data, default=str).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/tree":
            return self.send_json(tree())
        if m := re.fullmatch(r"/api/entry/(\d+)", path):
            return self.send_json(entry(int(m.group(1))))
        if path.startswith("/site/"):  # the original offline JPAC pages, for side-by-side comparison
            file = (SITE / path.removeprefix("/site/")).resolve()
            if file.is_relative_to(SITE) and file.is_file():
                return self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        self.send(404, b"not found", "text/plain")

    def log_message(self, *args):  # keep the console quiet
        pass


if __name__ == "__main__":
    print("http://localhost:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
