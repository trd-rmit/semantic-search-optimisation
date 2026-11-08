-- Structure-based chunks: one chunk = one complete set of rules for one situation.
--   entry without scenarios -> 1 chunk (the whole entry)
--   entry with scenarios    -> 1 chunk per scenario + the entry's sections that sit outside any scenario
-- Run after the tables are loaded; drops and rebuilds the chunks each time.

DROP TABLE IF EXISTS chunk_sections, chunks CASCADE;

CREATE TABLE chunks (
    chunk_id    serial PRIMARY KEY,
    key         text NOT NULL UNIQUE,                                -- stable across reloads: 'alopecia', 'hepatitis-b#3'
    entry_id    int  NOT NULL REFERENCES entries ON DELETE CASCADE,
    scenario_id int  UNIQUE REFERENCES scenarios ON DELETE CASCADE,   -- NULL = whole-entry chunk
    text        text NOT NULL,                                       -- what gets embedded
    chars       int  GENERATED ALWAYS AS (length(text)) STORED
);
CREATE UNIQUE INDEX ON chunks (entry_id) WHERE scenario_id IS NULL;  -- one whole-entry chunk per entry

-- Which sections went into each chunk. A shared section (e.g. Hepatitis B Definitions) is in several chunks.
CREATE TABLE chunk_sections (
    chunk_id   int REFERENCES chunks ON DELETE CASCADE,
    section_id int REFERENCES sections ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, section_id)
);
CREATE INDEX ON chunk_sections (section_id);

-- Section kinds that go into the embedded text. Left out: see_if_relevant (a list of entry names,
-- kept via links), regulatory_information (same legal text in many entries).
CREATE TEMP VIEW chunk_members AS
WITH units AS (
    SELECT e.entry_id, NULL::int AS scenario_id FROM entries e
    WHERE NOT EXISTS (SELECT 1 FROM scenarios sc WHERE sc.entry_id = e.entry_id)
    UNION ALL
    SELECT entry_id, scenario_id FROM scenarios
)
SELECT u.entry_id, u.scenario_id AS unit_scenario_id, s.section_id, s.position, s.kind, s.text
FROM units u
JOIN sections s ON s.entry_id = u.entry_id AND (s.scenario_id IS NULL OR s.scenario_id = u.scenario_id)
WHERE s.kind IN ('also_known_as', 'definitions', 'includes', 'excludes', 'obligatory', 'discretionary',
                 'additional_information', 'post_donation_illness');

INSERT INTO chunks (key, entry_id, scenario_id, text)
SELECT e.slug || COALESCE('#' || sc.position, ''), m.entry_id, m.unit_scenario_id,
       'Entry: ' || e.title || COALESCE(E'\nScenario: ' || sc.title, '') || E'\n\n' ||
       string_agg(
           CASE m.kind
               WHEN 'also_known_as'          THEN 'Also known as'
               WHEN 'definitions'            THEN 'Definitions'
               WHEN 'includes'               THEN 'Includes'
               WHEN 'excludes'               THEN 'Excludes'
               WHEN 'obligatory'             THEN 'Obligatory (must not donate)'
               WHEN 'discretionary'          THEN 'Discretionary (may donate if)'
               WHEN 'additional_information' THEN 'Additional information'
               WHEN 'post_donation_illness'  THEN 'Post donation illness'
           END || E':\n' || m.text,
           E'\n\n' ORDER BY m.position)
FROM chunk_members m
JOIN entries e ON e.entry_id = m.entry_id
LEFT JOIN scenarios sc ON sc.scenario_id = m.unit_scenario_id
GROUP BY m.entry_id, m.unit_scenario_id, e.slug, e.title, sc.title, sc.position
ORDER BY lower(e.title), sc.position NULLS FIRST;

INSERT INTO chunk_sections (chunk_id, section_id)
SELECT c.chunk_id, m.section_id
FROM chunk_members m
JOIN chunks c ON c.entry_id = m.entry_id AND c.scenario_id IS NOT DISTINCT FROM m.unit_scenario_id;
