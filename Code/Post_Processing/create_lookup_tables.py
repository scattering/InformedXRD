"""
Generate canonical extinction group and space group lookup tables.

Outputs:
- final_canonical_extinction_table.csv
- spacegroup_to_canonical_lookup.csv

Requires:
    pip install requests beautifulsoup4 pandas gemmi
"""

import re
import requests
import pandas as pd
import gemmi
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import List


# ============================================
# 1. Fetch and parse EXPO extinction symbols
# ============================================

EXPO_URL = "https://www.ba.ic.cnr.it/softwareic/expo/extinction_symbols/"

def fetch_extinction_table(url: str = EXPO_URL) -> pd.DataFrame:
    """
    Fetches the extinction symbol table from the EXPO website and returns it as a DataFrame.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise RuntimeError("❌ Could not find extinction symbol table")

    rows = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != 3:
            continue

        idx = cells[0].get_text(strip=True)
        if not idx.isdigit():
            continue

        symbol = cells[1].get_text(strip=True).replace("\xa0", " ").replace("–", "-")
        sg_raw = cells[2].get_text(strip=True).replace("\xa0", " ").replace("–", "-")

        rows.append({
            "Extinction Group Number": int(idx),
            "Extinction Symbol": symbol,
            "Space Groups": sg_raw
        })

    return pd.DataFrame(rows)


# ============================================
# 2. Extract space group numbers
# ============================================

def extract_spacegroup_numbers(raw: str) -> List[int]:
    """
    Extracts international space group numbers from EXPO table strings.
    Handles single numbers, comma-separated lists, and closed ranges.
    """
    numbers: List[int] = []
    for segment in re.findall(r"\(([^)]*)\)", raw):
        for item in (part.strip() for part in segment.split(",")):
            if not item:
                continue
            if "‑" in item or "-" in item:
                start, end = re.split(r"[‑\-]", item)
                numbers.extend(range(int(start), int(end) + 1))
            else:
                numbers.append(int(item))
    return numbers


def simplify_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column with extracted space group numbers.
    """
    df = df.copy()
    df["Space Group Numbers"] = df["Space Groups"].apply(extract_spacegroup_numbers)
    df.drop(columns=["Space Groups"], inplace=True)
    return df


# ============================================
# 3. Assign canonical extinction groups
# ============================================

def assign_canonical_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns canonical extinction groups by grouping extinction symbols
    with identical sets of space group numbers.
    Adds a column with canonical group labels in readable format.
    """
    fp_to_labels = defaultdict(list)
    for _, row in df.iterrows():
        fingerprint = tuple(sorted(row["Space Group Numbers"]))
        label = f"{row['Extinction Group Number']:03d}-{row['Extinction Symbol']}"
        fp_to_labels[fingerprint].append(label)

    fp_to_canonical = {}
    for fp, labels in fp_to_labels.items():
        sorted_labels = sorted(labels)
        main_label = sorted_labels[0]
        _, main_sym = main_label.split("-", 1)
        equiv_syms = [lbl.split("-", 1)[1] for lbl in sorted_labels[1:]]

        if equiv_syms:
            display = f"{main_sym} (equiv: {', '.join(equiv_syms)})"
        else:
            display = main_sym

        fp_to_canonical[fp] = display

    df = df.copy()
    df["Canonical Extinction Group"] = df.apply(
        lambda row: fp_to_canonical[tuple(sorted(row["Space Group Numbers"]))],
        axis=1
    )
    return df


# ============================================
# 4. Build final canonical extinction group table
# ============================================

def build_final_canonical_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the final table with:
    - Index (sequential integer)
    - Canonical Extinction Group (symbol with equiv listing)
    - Space Group Numbers (sorted list)
    """
    grouped = df.groupby("Canonical Extinction Group")

    rows = []
    for idx, (canon, group) in enumerate(grouped):
        sg_nums = sorted({sg for sgs in group["Space Group Numbers"] for sg in sgs})
        rows.append({
            "Index": idx,
            "Canonical Extinction Group": canon,
            "Space Group Numbers": sg_nums
        })

    return pd.DataFrame(rows)


# ============================================
# 5. Space group → crystal system and Laue group mapping
# ============================================

def sg_to_crystal_system(sg_num: int) -> str:
    if 1 <= sg_num <= 2:
        return "Triclinic"
    elif 3 <= sg_num <= 15:
        return "Monoclinic"
    elif 16 <= sg_num <= 74:
        return "Orthorhombic"
    elif 75 <= sg_num <= 142:
        return "Tetragonal"
    elif 143 <= sg_num <= 167:
        return "Trigonal"
    elif 168 <= sg_num <= 194:
        return "Hexagonal"
    elif 195 <= sg_num <= 230:
        return "Cubic"
    else:
        return "Unknown"


def sg_to_laue_group(sg_num: int) -> str:
    sg = gemmi.SpaceGroup(sg_num)
    return sg.laue_str()


# ============================================
# 6. Build space group lookup table with index
# ============================================

def build_sg_to_canonical_lookup(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a table mapping each space group to:
    - Canonical Extinction Group (symbol)
    - Index (matching canonical table)
    - Laue Group
    - Crystal System
    """
    rows = []
    for _, row in canonical_df.iterrows():
        canonical = row["Canonical Extinction Group"]
        index = row["Index"]
        sg_nums = row["Space Group Numbers"]

        for sg_num in sg_nums:
            crystal_system = sg_to_crystal_system(sg_num)
            laue_group = sg_to_laue_group(sg_num)

            rows.append({
                "Space Group Number": sg_num,
                "Canonical Extinction Group": canonical,
                "Index": index,
                "Laue Group": laue_group,
                "Crystal System": crystal_system
            })

    lookup_df = pd.DataFrame(rows)
    lookup_df.sort_values("Space Group Number", inplace=True)
    return lookup_df


# ============================================
# 7. Main execution
# ============================================

def main():
    # Fetch and process extinction data
    df_raw = fetch_extinction_table()
    df_simplified = simplify_table(df_raw)
    df_with_canonicals = assign_canonical_groups(df_simplified)
    final_table = build_final_canonical_table(df_with_canonicals)

    # Save final canonical extinction group table
    final_table.to_csv("final_canonical_extinction_table.csv", index=False)
    print("✅ Saved final_canonical_extinction_table.csv")

    # Build and save space group lookup table with consistent index
    sg_lookup = build_sg_to_canonical_lookup(final_table)
    sg_lookup.to_csv("spacegroup_to_canonical_lookup.csv", index=False)
    print("✅ Saved spacegroup_to_canonical_lookup.csv")


if __name__ == "__main__":
    main()
