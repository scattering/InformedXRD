--1) Create three empty “part” databases with the same schema
-- open your main DB
-- sqlite3 crystal_xrd.db

-- attach three empty files (they’ll be created)
ATTACH DATABASE 'icsd_part1.db' AS part1;
ATTACH DATABASE 'icsd_part2.db' AS part2;
ATTACH DATABASE 'icsd_part3.db' AS part3;

--USE THIS INSTEAD
-- attach three empty files (they’ll be created)
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part1.db' AS part1;
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part2.db' AS part2;
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part3.db' AS part3;

-- create identical Structures tables in each part DB
CREATE TABLE IF NOT EXISTS part1.Structures AS SELECT * FROM main.Structures WHERE 0;
CREATE TABLE IF NOT EXISTS part2.Structures AS SELECT * FROM main.Structures WHERE 0;
CREATE TABLE IF NOT EXISTS part3.Structures AS SELECT * FROM main.Structures WHERE 0;

-- recreate your index (and any others you want) in each part DB
CREATE INDEX IF NOT EXISTS part1.idx_structures_source ON Structures(source);
CREATE INDEX IF NOT EXISTS part2.idx_structures_source ON Structures(source);
CREATE INDEX IF NOT EXISTS part3.idx_structures_source ON Structures(source);

--2) Split ICSD rows into thirds using rowid % 3

--This partitions deterministically without reading all rows into memory.
PRAGMA journal_mode = WAL;
PRAGMA synchronous = OFF;
BEGIN;

INSERT INTO part1.Structures
SELECT * FROM main.Structures
WHERE source='ICSD' AND (rowid % 3) = 0;

INSERT INTO part2.Structures
SELECT * FROM main.Structures
WHERE source='ICSD' AND (rowid % 3) = 1;

INSERT INTO part3.Structures
SELECT * FROM main.Structures
WHERE source='ICSD' AND (rowid % 3) = 2;

COMMIT;

-- sanity checks
SELECT 'main_ICSD',  COUNT(*) FROM main.Structures  WHERE source='ICSD';
SELECT 'part1_ICSD', COUNT(*) FROM part1.Structures WHERE source='ICSD';
SELECT 'part2_ICSD', COUNT(*) FROM part2.Structures WHERE source='ICSD';
SELECT 'part3_ICSD', COUNT(*) FROM part3.Structures WHERE source='ICSD';
--The three counts should sum to the main ICSD count.

--- ###------DESTINATION DATABASE FILE ---------###
--3) Create a fresh “merged target” DB and merge all parts into it
-- create the table with the same schema as your original
CREATE TABLE IF NOT EXISTS Structures AS
SELECT * FROM (SELECT * FROM main.sqlite_schema) WHERE 0; 

--create table in merged database
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

--and indexes
CREATE INDEX IF NOT EXISTS idx_structures_source_id ON Structures(source_id);
CREATE INDEX IF NOT EXISTS idx_structures_source ON Structures(source);

--Attach the three part databases by absolute path

--3.Attach the three part databases by absolute path


ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part1.db' AS p1;
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part2.db' AS p2;
ATTACH DATABASE 'C:/Users/UserD/OneDrive/ADVANCED ED/OrgB/OrgB-Program/OrgA/crystal_db/icsd_part3.db' AS p3;

PRAGMA journal_mode = WAL;
PRAGMA synchronous = OFF;


--4.Merge the rows into icsd_merged.db (Use a single transaction; OR IGNORE protects against accidental duplicates.)
BEGIN;

INSERT OR IGNORE INTO main.Structures
SELECT * FROM p1.Structures;

INSERT OR IGNORE INTO main.Structures
SELECT * FROM p2.Structures;

INSERT OR IGNORE INTO main.Structures
SELECT * FROM p3.Structures;

COMMIT;

--5.Verify counts
SELECT 'merged_total', COUNT(*) FROM main.Structures;
SELECT 'merged_ICSD', COUNT(*) FROM main.Structures WHERE source='ICSD';


--6. (Optional) Clean up / compact
-- See attached DBs
PRAGMA database_list;

-- Detach when done
DETACH DATABASE p1;
DETACH DATABASE p2;
DETACH DATABASE p3;

-- Compact the main file (close other connections first if needed)
PRAGMA wal_checkpoint(TRUNCATE);
VACUUM;