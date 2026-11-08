from semsearch.load_jpac import classify, to_text


def test_nested_list_starts_on_its_own_indented_line():
    html = "<ol><li>If:<ol><li>3 months after, and</li><li>test done,</li></ol>accept.</li><li>Otherwise refer.</li></ol>"
    assert to_text(html) == "1. If:\n   1. 3 months after, and\n   2. test done,\naccept.\n2. Otherwise refer."


def test_inline_links_and_bold_stay_on_one_line():
    html = '<p>refer to\n <a href="x">Active or recovered HBV infection</a>\n above.</p><p><b>Must not donate if:</b></p><ul><li>a</li></ul>'
    assert to_text(html) == "refer to Active or recovered HBV infection above.\nMust not donate if:\n- a"


def test_table_rows_become_pipe_separated_lines():
    assert to_text("<table><tr><th>Weight</th><th>Volume</th></tr><tr><td>50</td><td>3.2</td></tr></table>") == \
        "Weight | Volume\n50 | 3.2"


def test_classify_links():
    slugs = {"surgery"}
    assert classify("https://preview.jpac.org.uk/guidelines/wb/index/surgery/index.html", slugs) == ("entry", "surgery", None)
    assert classify("https://www.jpac.org.uk/guidelines/wb/index/index.html", slugs) == ("a_to_z", None, None)
    assert classify("https://www.jpac.org.uk/guidelines/wb/appendix-1/index.html", slugs)[0] == "external"
