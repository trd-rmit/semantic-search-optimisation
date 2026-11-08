-- JPAC Whole Blood and Components Donor Selection Guidelines, as relational data.
-- Re-running this file drops and recreates everything (the loader does that on each load).

DROP TABLE IF EXISTS chunk_sections, chunks, links, sections, scenarios, entries CASCADE;

-- One A-Z entry (e.g. Alopecia).
CREATE TABLE entries (
    entry_id                serial PRIMARY KEY,
    slug                    text NOT NULL UNIQUE,          -- URL name, e.g. 'alopecia'
    title                   text NOT NULL,
    url                     text NOT NULL,
    release                 int,                           -- from 'WB-DSG Edition 203 Release 81 (...)'
    version_date            date,
    version_details         text,
    reason_for_change       text,
    change_notification_url text
);

-- Some entries (e.g. Hepatitis B) split their rules into named scenarios.
CREATE TABLE scenarios (
    scenario_id serial PRIMARY KEY,
    entry_id    int  NOT NULL REFERENCES entries ON DELETE CASCADE,
    position    int  NOT NULL,
    title       text NOT NULL,
    UNIQUE (entry_id, position)
);

-- A labelled block of an entry: Obligatory, Discretionary, See if relevant, ...
CREATE TABLE sections (
    section_id  serial PRIMARY KEY,
    entry_id    int  NOT NULL REFERENCES entries ON DELETE CASCADE,
    scenario_id int  REFERENCES scenarios ON DELETE CASCADE,    -- NULL = not inside a scenario
    kind        text NOT NULL CHECK (kind IN (
                    'also_known_as', 'definitions', 'includes', 'excludes',
                    'obligatory', 'discretionary', 'see_if_relevant', 'additional_information',
                    'regulatory_information', 'post_donation_illness')),
    position    int  NOT NULL,                                  -- order within the entry
    text        text NOT NULL,
    html        text NOT NULL,
    UNIQUE (entry_id, position)
);

-- A hyperlink found inside a section. Entry-to-entry links make entries many-to-many with themselves.
CREATE TABLE links (
    link_id         serial PRIMARY KEY,
    from_section_id int  NOT NULL REFERENCES sections ON DELETE CASCADE,
    target_kind     text NOT NULL CHECK (target_kind IN ('entry', 'a_to_z', 'external')),
    to_entry_id     int  REFERENCES entries ON DELETE CASCADE,
    to_url          text,
    anchor_text     text NOT NULL,
    context         text,                                       -- e.g. 'For hair transplants:'
    CHECK ((target_kind = 'entry')    = (to_entry_id IS NOT NULL)),
    CHECK (target_kind <> 'external' OR to_url IS NOT NULL)
);

CREATE INDEX ON scenarios (entry_id);
CREATE INDEX ON sections (entry_id);
CREATE INDEX ON sections (scenario_id);
CREATE INDEX ON links (from_section_id);
CREATE INDEX ON links (to_entry_id);
