-- 1) Create three empty “part” databases with the same schema (absolute paths on Windows)
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part1.db' AS part1;
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part2.db' AS part2;
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part3.db' AS part3;

-- create identical Structures tables in each part DB
CREATE TABLE IF NOT EXISTS part1.Structures AS SELECT * FROM main.Structures WHERE 0;
CREATE TABLE IF NOT EXISTS part2.Structures AS SELECT * FROM main.Structures WHERE 0;
CREATE TABLE IF NOT EXISTS part3.Structures AS SELECT * FROM main.Structures WHERE 0;

-- optional indexes for source filtering
CREATE INDEX IF NOT EXISTS part1.idx_structures_source ON Structures(source);
CREATE INDEX IF NOT EXISTS part2.idx_structures_source ON Structures(source);
CREATE INDEX IF NOT EXISTS part3.idx_structures_source ON Structures(source);

-- 2) Split ICSD rows into thirds
PRAGMA journal_mode = WAL;
PRAGMA synchronous = OFF;
BEGIN;

INSERT INTO part1.Structures SELECT * FROM main.Structures WHERE source='ICSD' AND (rowid % 3) = 0;
INSERT INTO part2.Structures SELECT * FROM main.Structures WHERE source='ICSD' AND (rowid % 3) = 1;
INSERT INTO part3.Structures SELECT * FROM main.Structures WHERE source='ICSD' AND (rowid % 3) = 2;

COMMIT;

-- sanity checks
SELECT 'main_ICSD',  COUNT(*) FROM main.Structures  WHERE source='ICSD';
SELECT 'part1_ICSD', COUNT(*) FROM part1.Structures WHERE source='ICSD';
SELECT 'part2_ICSD', COUNT(*) FROM part2.Structures WHERE source='ICSD';
SELECT 'part3_ICSD', COUNT(*) FROM part3.Structures WHERE source='ICSD';

-- 3) In the new destination DB (icsd_merged.db): create table with your column order
CREATE TABLE IF NOT EXISTS Structures (
    structure_id TEXT PRIMARY KEY,
    space_group INTEGER,
    source TEXT,
    source_id TEXT,
    generation_commit TEXT,
    extinction_class TEXT,
    extinction_group INTEGER,
    crystal_system TEXT,
    composition TEXT,
    lattice_a REAL,
    lattice_b REAL,
    lattice_c REAL,
    alpha REAL,
    beta REAL,
    gamma REAL,
    wyckoff_positions TEXT,
    atom_sites_json TEXT,
    creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    publication_year INTEGER,
    ICSD_creation_date TEXT
);

CREATE INDEX IF NOT EXISTS idx_structures_source_id ON Structures(source_id);
CREATE INDEX IF NOT EXISTS idx_structures_source ON Structures(source);

-- Attach the three parts into the merged DB session
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part1.db' AS p1;
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part2.db' AS p2;
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part3.db' AS p3;

PRAGMA journal_mode = WAL;
PRAGMA synchronous = OFF;

BEGIN;
INSERT OR IGNORE INTO main.Structures SELECT * FROM p1.Structures;
INSERT OR IGNORE INTO main.Structures SELECT * FROM p2.Structures;
INSERT OR IGNORE INTO main.Structures SELECT * FROM p3.Structures;
COMMIT;

-- Verify
SELECT 'merged_total', COUNT(*) FROM main.Structures;
SELECT 'merged_ICSD', COUNT(*) FROM main.Structures WHERE source='ICSD';

-- Cleanup
DETACH DATABASE p1; DETACH DATABASE p2; DETACH DATABASE p3;
PRAGMA wal_checkpoint(TRUNCATE);
VACUUM;
