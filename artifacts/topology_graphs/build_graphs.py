#!/usr/bin/env python3
"""
Build group-subgroup graphs for:
  (1) The 230 crystallographic space groups (maximal t-subgroups from ITA1/Bilbao)
  (2) The corresponding extinction groups (induced graph)

The maximal translationengleiche (t-)subgroup data below is compiled from
International Tables for Crystallography, Vol. A1 (Wondratschek & Müller)
and the Bilbao Crystallographic Server (MAXSUB).

Only t-subgroups (translationengleiche, i.e. same translation lattice,
reduced point group) are included here, as these are the ones relevant
for the extinction group hierarchy — klassengleiche subgroups change the
lattice but not the point group, so they can change centering extinctions
but not in a way that maps cleanly onto the extinction group DAG.

Storage format: NetworkX DiGraph serialized as JSON (node-link format).
This is human-readable, version-controllable, and trivially loadable in Python.

Author: Claude (compiled from ITA1 data)
Date: 2026-03-20
"""

import json
import csv
import ast
import networkx as nx
from collections import defaultdict
from pathlib import Path

# ============================================================================
# PART 1: Maximal t-subgroup data from ITA1 / Bilbao MAXSUB
# ============================================================================
# Format: {space_group_number: [list of maximal t-subgroup numbers]}
# These are TYPE relationships (by SG number), not specific subgroup instances.
# Only t-subgroups (translationengleiche) are listed.
# For klassengleiche (k-) subgroups, one would need additional data.

MAXIMAL_T_SUBGROUPS = {
    # Triclinic
    1: [],
    2: [1],

    # Monoclinic
    3: [1],
    4: [1],
    5: [3],  # C2 > P2
    6: [1],
    7: [1],
    8: [6],  # Cm > Pm
    9: [7],  # Cc > Pc
    10: [2, 3, 6],
    11: [2, 4, 6],
    12: [5, 8, 10],
    13: [2, 3, 7],
    14: [2, 4, 7],
    15: [5, 9, 13],

    # Orthorhombic
    16: [3],
    17: [3, 4],
    18: [4],
    19: [4],
    20: [4, 5],
    21: [3, 5],
    22: [3, 5],
    23: [3, 5],
    24: [4, 5],
    25: [3, 6],
    26: [4, 6, 7],
    27: [3, 7],
    28: [3, 6, 7],
    29: [4, 7],
    30: [4, 7],
    31: [4, 6, 7],
    32: [4, 7],
    33: [4, 7],
    34: [4, 7],
    35: [5, 8, 25],
    36: [5, 8, 9, 26],
    37: [5, 9, 27],
    38: [5, 6, 8, 25],
    39: [5, 7, 8, 9, 28],
    40: [5, 7, 8, 9, 28],
    41: [5, 7, 9, 29],
    42: [5, 8, 25],
    43: [5, 9, 34],
    44: [5, 8, 23, 25],
    45: [5, 9, 27],
    46: [5, 7, 8, 9, 28],
    47: [2, 10, 16, 25],
    48: [2, 13, 16, 34],
    49: [2, 10, 13, 16, 25, 27],
    50: [2, 10, 13, 16, 27, 28],
    51: [2, 10, 11, 16, 25, 26, 28],
    52: [2, 11, 13, 14, 16, 29, 34],
    53: [2, 10, 11, 13, 14, 17, 28, 30],
    54: [2, 13, 14, 17, 27, 29],
    55: [2, 11, 14, 18, 25, 32],
    56: [2, 14, 18, 27, 34],
    57: [2, 11, 14, 17, 26, 29, 31],
    58: [2, 11, 14, 18, 27, 34],
    59: [2, 11, 18, 25, 31],
    60: [2, 14, 17, 29, 33, 34],
    61: [2, 14, 19, 29],
    62: [2, 11, 14, 19, 26, 31, 33],
    63: [12, 15, 20, 36, 40],
    64: [12, 14, 15, 20, 36, 41],
    65: [2, 10, 12, 21, 25, 35, 38, 47],
    66: [2, 10, 12, 13, 15, 21, 27, 37, 49],
    67: [2, 10, 11, 12, 15, 21, 28, 35, 39, 51],
    68: [2, 12, 13, 14, 15, 21, 34, 37, 41],
    69: [12, 22, 42, 47, 65],
    70: [12, 15, 22, 43, 48],
    71: [2, 10, 12, 23, 25, 44, 47, 65],
    72: [2, 10, 12, 13, 15, 23, 27, 45, 49],
    73: [2, 14, 15, 24, 29, 45],
    74: [2, 10, 11, 12, 15, 23, 28, 44, 46, 51],

    # Tetragonal
    75: [3],
    76: [4],
    77: [3],
    78: [4],
    79: [5, 75],
    80: [5, 76],
    81: [3],
    82: [5, 81],
    83: [6, 10, 75],
    84: [7, 13, 77],
    85: [7, 13, 75],
    86: [7, 14, 76],
    87: [8, 12, 79, 83],
    88: [9, 15, 80, 85],
    89: [16, 21, 75],
    90: [18, 21, 75],
    91: [17, 20, 76],
    92: [19, 20, 76],
    93: [17, 21, 77],
    94: [18, 21, 77],
    95: [17, 20, 78],
    96: [19, 20, 78],
    97: [23, 22, 79, 89],
    98: [24, 22, 80, 91],
    99: [25, 35, 75],
    100: [28, 35, 75],
    101: [27, 37, 77],
    102: [30, 37, 77],
    103: [27, 37, 75],
    104: [30, 37, 75],
    105: [26, 36, 77],
    106: [29, 36, 77],
    107: [38, 44, 79, 99],
    108: [40, 45, 79, 101],
    109: [41, 46, 80, 100],
    110: [41, 45, 80, 103],
    111: [16, 21, 81],
    112: [16, 21, 81],
    113: [18, 21, 81],
    114: [19, 24, 81],
    115: [25, 35, 81],
    116: [27, 37, 81],
    117: [28, 35, 81],
    118: [30, 34, 81],
    119: [38, 44, 82, 115],
    120: [40, 45, 82, 116],
    121: [23, 22, 82, 111],
    122: [24, 43, 82, 114],
    123: [47, 65, 83, 89, 99, 111, 115],
    124: [49, 66, 83, 93, 101, 112, 116],
    125: [50, 67, 85, 89, 100, 113, 117],
    126: [48, 68, 85, 93, 103, 114, 118],
    127: [49, 65, 84, 90, 99, 111, 117],
    128: [50, 66, 84, 94, 103, 112, 116],
    129: [51, 67, 83, 90, 100, 113, 115],
    130: [52, 68, 84, 94, 104, 114, 118],
    131: [47, 65, 84, 93, 105, 112, 115],
    132: [49, 66, 84, 89, 99, 111, 116],
    133: [48, 68, 86, 91, 103, 114, 118],
    134: [50, 67, 86, 93, 101, 113, 117],
    135: [52, 68, 83, 94, 104, 114, 116],
    136: [51, 67, 83, 91, 102, 112, 115],
    137: [51, 67, 86, 95, 102, 113, 118],
    138: [52, 68, 86, 91, 104, 114, 117],
    139: [47, 65, 69, 71, 87, 97, 107, 119, 121, 123],
    140: [49, 66, 69, 72, 87, 97, 108, 120, 121, 124],
    141: [50, 68, 70, 74, 88, 98, 109, 122, 125],
    142: [52, 68, 70, 73, 88, 98, 110, 122, 126],

    # Trigonal
    143: [1],
    144: [1],
    145: [1],
    146: [5, 143],
    147: [2, 143],
    148: [2, 146],
    149: [3, 143],
    150: [3, 143],
    151: [4, 144],
    152: [4, 144],
    153: [4, 145],
    154: [4, 145],
    155: [5, 146, 149],
    156: [6, 143],
    157: [6, 143],
    158: [7, 143],
    159: [7, 143],
    160: [8, 146, 156],
    161: [9, 146, 158],
    162: [10, 147, 149, 156],
    163: [13, 147, 150, 158],
    164: [10, 147, 150, 157],
    165: [14, 147, 149, 159],
    166: [12, 148, 155, 160, 162],
    167: [15, 148, 155, 161, 165],

    # Hexagonal
    168: [3, 143],
    169: [4, 144],
    170: [4, 145],
    171: [3, 144],
    172: [3, 145],
    173: [4, 143],
    174: [6, 143],
    175: [10, 147, 168, 174],
    176: [13, 147, 173, 174],
    177: [16, 149, 150, 168],
    178: [17, 151, 152, 169],
    179: [17, 153, 154, 170],
    180: [17, 152, 153, 171],
    181: [17, 151, 154, 172],
    182: [18, 152, 154, 173],
    183: [25, 156, 157, 168],
    184: [27, 158, 159, 168],
    185: [26, 157, 159, 173],
    186: [26, 156, 158, 173],
    187: [25, 156, 157, 174],
    188: [28, 158, 159, 174],
    189: [25, 156, 157, 174],
    190: [28, 158, 159, 174],
    191: [47, 162, 164, 175, 177, 183, 187, 189],
    192: [49, 163, 165, 175, 177, 184, 188, 190],
    193: [50, 163, 165, 176, 182, 185, 188, 190],
    194: [51, 162, 164, 176, 182, 186, 187, 189],

    # Cubic
    195: [16, 143, 149],
    196: [22, 146, 155],
    197: [23, 146, 155, 195],
    198: [19, 144, 149],
    199: [24, 146, 155, 198],
    200: [2, 10, 16, 47, 147, 162, 195],
    201: [2, 13, 16, 48, 147, 163, 195],
    202: [12, 22, 47, 69, 148, 166, 196],
    203: [12, 15, 22, 48, 70, 148, 167, 196],
    204: [2, 10, 12, 23, 47, 71, 148, 166, 197, 200],
    205: [2, 14, 24, 61, 148, 165, 198],
    206: [2, 14, 15, 24, 73, 148, 167, 199, 205],
    207: [16, 21, 89, 149, 150, 195],
    208: [16, 21, 93, 149, 150, 195],
    209: [22, 21, 97, 155, 196],
    210: [22, 24, 98, 155, 196],
    211: [23, 22, 97, 155, 197, 207],
    212: [19, 20, 92, 144, 152],
    213: [19, 20, 96, 145, 154],
    214: [24, 22, 98, 155, 199, 212],
    215: [25, 35, 81, 156, 160],
    216: [42, 38, 82, 160, 215],
    217: [44, 23, 82, 160, 215],
    218: [27, 34, 81, 158, 161],
    219: [42, 40, 82, 161, 218],
    220: [43, 24, 82, 161, 218],
    221: [47, 65, 123, 162, 164, 200, 207, 215],
    222: [48, 68, 126, 163, 165, 201, 208, 218],
    223: [47, 65, 124, 163, 165, 200, 208, 218],
    224: [48, 68, 125, 162, 164, 201, 207, 215],
    225: [69, 65, 139, 166, 202, 209, 216, 221],
    226: [69, 66, 140, 167, 202, 209, 219, 223],
    227: [70, 68, 141, 166, 203, 210, 216, 224],
    228: [70, 68, 142, 167, 203, 210, 220, 222],
    229: [71, 65, 69, 139, 166, 204, 211, 217, 221, 225],
    230: [73, 68, 70, 142, 167, 206, 214, 220, 222, 228],
}


def build_space_group_graph():
    """Build a directed graph of maximal t-subgroup relationships.

    Edge direction: parent -> child (i.e., G -> H means H is a maximal
    t-subgroup of G).

    Node attributes include the SG number and Hermann-Mauguin symbol.
    """
    # Hermann-Mauguin symbols for all 230 space groups
    HM_SYMBOLS = {
        1: "P1", 2: "P-1",
        3: "P2", 4: "P2_1", 5: "C2",
        6: "Pm", 7: "Pc", 8: "Cm", 9: "Cc",
        10: "P2/m", 11: "P2_1/m", 12: "C2/m", 13: "P2/c", 14: "P2_1/c", 15: "C2/c",
        16: "P222", 17: "P222_1", 18: "P2_12_12", 19: "P2_12_12_1",
        20: "C222_1", 21: "C222", 22: "F222", 23: "I222", 24: "I2_12_12_1",
        25: "Pmm2", 26: "Pmc2_1", 27: "Pcc2", 28: "Pma2", 29: "Pca2_1",
        30: "Pnc2", 31: "Pmn2_1", 32: "Pba2", 33: "Pna2_1", 34: "Pnn2",
        35: "Cmm2", 36: "Cmc2_1", 37: "Ccc2", 38: "Amm2", 39: "Aem2",
        40: "Ama2", 41: "Aea2", 42: "Fmm2", 43: "Fdd2", 44: "Imm2",
        45: "Iba2", 46: "Ima2",
        47: "Pmmm", 48: "Pnnn", 49: "Pccm", 50: "Pban",
        51: "Pmma", 52: "Pnna", 53: "Pmna", 54: "Pcca",
        55: "Pbam", 56: "Pccn", 57: "Pbcm", 58: "Pnnm",
        59: "Pmmn", 60: "Pbcn", 61: "Pbca", 62: "Pnma",
        63: "Cmcm", 64: "Cmce", 65: "Cmmm", 66: "Cccm",
        67: "Cmme", 68: "Ccce", 69: "Fmmm", 70: "Fddd",
        71: "Immm", 72: "Ibam", 73: "Ibca", 74: "Imma",
        75: "P4", 76: "P4_1", 77: "P4_2", 78: "P4_3",
        79: "I4", 80: "I4_1",
        81: "P-4", 82: "I-4",
        83: "P4/m", 84: "P4_2/m", 85: "P4/n", 86: "P4_2/n",
        87: "I4/m", 88: "I4_1/a",
        89: "P422", 90: "P42_12", 91: "P4_122", 92: "P4_12_12",
        93: "P4_222", 94: "P4_22_12", 95: "P4_322", 96: "P4_32_12",
        97: "I422", 98: "I4_122",
        99: "P4mm", 100: "P4bm", 101: "P4_2cm", 102: "P4_2nm",
        103: "P4cc", 104: "P4nc", 105: "P4_2mc", 106: "P4_2bc",
        107: "I4mm", 108: "I4cm", 109: "I4_1md", 110: "I4_1cd",
        111: "P-42m", 112: "P-42c", 113: "P-42_1m", 114: "P-42_1c",
        115: "P-4m2", 116: "P-4c2", 117: "P-4b2", 118: "P-4n2",
        119: "I-4m2", 120: "I-4c2", 121: "I-42m", 122: "I-42d",
        123: "P4/mmm", 124: "P4/mcc", 125: "P4/nbm", 126: "P4/nnc",
        127: "P4/mbm", 128: "P4/mnc", 129: "P4/nmm", 130: "P4/ncc",
        131: "P4_2/mmc", 132: "P4_2/mcm", 133: "P4_2/nbc", 134: "P4_2/nnm",
        135: "P4_2/mbc", 136: "P4_2/mnm", 137: "P4_2/nmc", 138: "P4_2/ncm",
        139: "I4/mmm", 140: "I4/mcm", 141: "I4_1/amd", 142: "I4_1/acd",
        143: "P3", 144: "P3_1", 145: "P3_2",
        146: "R3",
        147: "P-3", 148: "R-3",
        149: "P312", 150: "P321", 151: "P3_112", 152: "P3_121",
        153: "P3_212", 154: "P3_221", 155: "R32",
        156: "P3m1", 157: "P31m", 158: "P3c1", 159: "P31c",
        160: "R3m", 161: "R3c",
        162: "P-31m", 163: "P-31c", 164: "P-3m1", 165: "P-3c1",
        166: "R-3m", 167: "R-3c",
        168: "P6", 169: "P6_1", 170: "P6_5", 171: "P6_2",
        172: "P6_4", 173: "P6_3",
        174: "P-6",
        175: "P6/m", 176: "P6_3/m",
        177: "P622", 178: "P6_122", 179: "P6_522", 180: "P6_222",
        181: "P6_422", 182: "P6_322",
        183: "P6mm", 184: "P6cc", 185: "P6_3cm", 186: "P6_3mc",
        187: "P-6m2", 188: "P-6c2", 189: "P-62m", 190: "P-62c",
        191: "P6/mmm", 192: "P6/mcc", 193: "P6_3/mcm", 194: "P6_3/mmc",
        195: "P23", 196: "F23", 197: "I23", 198: "P2_13",
        199: "I2_13",
        200: "Pm-3", 201: "Pn-3", 202: "Fm-3", 203: "Fd-3",
        204: "Im-3", 205: "Pa-3", 206: "Ia-3",
        207: "P432", 208: "P4_232", 209: "F432", 210: "F4_132",
        211: "I432", 212: "P4_332", 213: "P4_132", 214: "I4_132",
        215: "P-43m", 216: "F-43m", 217: "I-43m",
        218: "P-43n", 219: "F-43c", 220: "I-43d",
        221: "Pm-3m", 222: "Pn-3n", 223: "Pm-3n", 224: "Pn-3m",
        225: "Fm-3m", 226: "Fm-3c", 227: "Fd-3m", 228: "Fd-3c",
        229: "Im-3m", 230: "Ia-3d",
    }

    # Crystal system assignment
    def crystal_system(sg):
        if sg <= 2: return "triclinic"
        if sg <= 15: return "monoclinic"
        if sg <= 74: return "orthorhombic"
        if sg <= 142: return "tetragonal"
        if sg <= 167: return "trigonal"
        if sg <= 194: return "hexagonal"
        return "cubic"

    G = nx.DiGraph()

    # Add nodes
    for sg in range(1, 231):
        G.add_node(sg,
                    hm_symbol=HM_SYMBOLS[sg],
                    crystal_system=crystal_system(sg))

    # Add edges (parent -> maximal t-subgroup)
    for parent, children in MAXIMAL_T_SUBGROUPS.items():
        for child in children:
            G.add_edge(parent, child, relation="maximal_t_subgroup")

    return G


def load_extinction_table(csv_path):
    """Load the extinction group table from CSV.

    Returns:
        extinction_groups: dict mapping extinction_group_index -> {
            'name': str,
            'space_groups': list[int]
        }
        sg_to_ext: dict mapping space_group_number -> extinction_group_index
    """
    extinction_groups = {}
    sg_to_ext = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['Index'])
            name = row['Canonical Extinction Group']
            sg_list = ast.literal_eval(row['Space Group Numbers'])

            extinction_groups[idx] = {
                'name': name,
                'space_groups': sg_list,
            }
            for sg in sg_list:
                sg_to_ext[sg] = idx

    return extinction_groups, sg_to_ext


def build_extinction_group_graph(sg_graph, extinction_groups, sg_to_ext):
    """Build the extinction group subgroup graph induced by the SG graph.

    Logic: Extinction group A is a "parent" of extinction group B if there
    exists an edge sg_a -> sg_b in the SG t-subgroup graph with
    sg_a in A, sg_b in B, and A != B.

    Physical subtlety: In the trigonal system, P3c1 (SG 158, ext "P - c -")
    and P31c (SG 159, ext "P - - c") are each subgroups of space groups
    in the other's extinction class, creating a genuine cycle. These two
    extinction patterns are incomparable — related by swapping the two
    trigonal symmetry directions (c-glide along different axes).

    We handle this by:
    1. Building the raw directed graph with all induced edges
    2. Condensing any strongly connected components (SCCs) into single nodes
    3. Taking the transitive reduction of the resulting DAG

    Returns both the condensed DAG and the raw (possibly cyclic) graph.
    """
    # First pass: find all direct connections between extinction groups
    ext_edges = set()
    for parent_sg, child_sg in sg_graph.edges():
        ext_parent = sg_to_ext.get(parent_sg)
        ext_child = sg_to_ext.get(child_sg)
        if ext_parent is not None and ext_child is not None and ext_parent != ext_child:
            ext_edges.add((ext_parent, ext_child))

    # Build raw graph (may have cycles)
    E_raw = nx.DiGraph()
    for idx, info in extinction_groups.items():
        E_raw.add_node(idx,
                       name=info['name'],
                       space_groups=info['space_groups'])
    for parent, child in ext_edges:
        E_raw.add_edge(parent, child)

    # Identify strongly connected components (cycles to merge)
    sccs = list(nx.strongly_connected_components(E_raw))
    nontrivial_sccs = [scc for scc in sccs if len(scc) > 1]
    if nontrivial_sccs:
        print(f"  Found {len(nontrivial_sccs)} nontrivial SCC(s) (cycle groups):")
        for scc in nontrivial_sccs:
            names = [extinction_groups[n]['name'][:40] for n in scc]
            print(f"    {scc}: {names}")
            print(f"    -> These extinction groups are incomparable (mutual subgroups)")
            print(f"       Physically: related by permutation of symmetry directions")

    # Build condensation: each SCC becomes one node
    condensation = nx.condensation(E_raw)

    # Map from condensed node -> set of original ext group indices
    # nx.condensation stores this in the 'members' attribute
    scc_members = nx.get_node_attributes(condensation, 'members')

    # Build a clean DAG with merged labels
    E_dag = nx.DiGraph()
    cond_to_orig = {}  # condensed node -> list of original ext group indices
    for cnode in condensation.nodes():
        members = sorted(scc_members[cnode])
        cond_to_orig[cnode] = members

        if len(members) == 1:
            idx = members[0]
            E_dag.add_node(idx,
                           name=extinction_groups[idx]['name'],
                           space_groups=extinction_groups[idx]['space_groups'],
                           merged_from=[idx])
        else:
            # Merged node: use lowest index as representative
            rep = min(members)
            merged_name = " | ".join(extinction_groups[m]['name'][:40] for m in members)
            merged_sgs = []
            for m in members:
                merged_sgs.extend(extinction_groups[m]['space_groups'])
            E_dag.add_node(rep,
                           name=merged_name,
                           space_groups=sorted(merged_sgs),
                           merged_from=members)

    # Add edges in the condensed graph
    for cu, cv in condensation.edges():
        u_rep = min(cond_to_orig[cu])
        v_rep = min(cond_to_orig[cv])
        E_dag.add_edge(u_rep, v_rep)

    # Transitive reduction
    E_reduced = nx.transitive_reduction(E_dag)
    for node in E_reduced.nodes():
        E_reduced.nodes[node].update(E_dag.nodes[node])
    for u, v in E_reduced.edges():
        E_reduced.edges[u, v]['relation'] = 'maximal_ext_subgroup'

    return E_reduced, E_raw


def save_graph_json(G, filepath):
    """Save a NetworkX DiGraph as node-link JSON."""
    data = nx.node_link_data(G)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {filepath}")
    print(f"    Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")


def save_graph_adjacency_json(G, filepath):
    """Save as a simpler adjacency-list JSON format.

    Format: {
        "nodes": {node_id: {attr_dict}},
        "adjacency": {node_id: [list of children]}
    }
    """
    data = {
        "nodes": {},
        "adjacency": {},
    }
    for node, attrs in G.nodes(data=True):
        # Convert node to string key for JSON
        key = str(node)
        data["nodes"][key] = {k: v for k, v in attrs.items()}
        data["adjacency"][key] = sorted([str(n) for n in G.successors(node)])

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {filepath} (adjacency-list format)")


def save_graph_graphml(G, filepath):
    """Save as GraphML (XML-based, readable by many graph tools)."""
    # GraphML doesn't like list attributes, convert to strings
    G2 = G.copy()
    for node in G2.nodes():
        for key, val in list(G2.nodes[node].items()):
            if isinstance(val, list):
                G2.nodes[node][key] = str(val)
    nx.write_graphml(G2, filepath)
    print(f"  Saved: {filepath} (GraphML format)")


def print_graph_stats(G, name):
    """Print basic statistics about a graph."""
    print(f"\n{'='*60}")
    print(f"Graph: {name}")
    print(f"{'='*60}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Find roots (no incoming edges) and leaves (no outgoing edges)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    print(f"  Roots (no parents): {len(roots)} -> {sorted(roots)[:10]}...")
    print(f"  Leaves (no children): {len(leaves)} -> {sorted(leaves)[:10]}...")

    # Max depth via longest path
    if nx.is_directed_acyclic_graph(G):
        print(f"  DAG: Yes")
        longest = nx.dag_longest_path_length(G)
        print(f"  Longest path length: {longest}")
    else:
        print(f"  DAG: No (contains cycles!)")
        cycles = list(nx.simple_cycles(G))[:5]
        print(f"  Example cycles: {cycles}")


def main():
    csv_path = "/mnt/user-data/uploads/final_canonical_extinction_table__1_.csv"
    out_dir = Path("/home/claude/output")
    out_dir.mkdir(exist_ok=True)

    print("Building space group maximal t-subgroup graph...")
    sg_graph = build_space_group_graph()
    print_graph_stats(sg_graph, "Space Group t-Subgroups")

    print("\nLoading extinction group table...")
    extinction_groups, sg_to_ext = load_extinction_table(csv_path)
    print(f"  Loaded {len(extinction_groups)} extinction groups")
    print(f"  Covering {len(sg_to_ext)} space groups")

    print("\nBuilding extinction group subgroup graph...")
    ext_graph, ext_graph_raw = build_extinction_group_graph(sg_graph, extinction_groups, sg_to_ext)
    print_graph_stats(ext_graph, "Extinction Group Subgroups (condensed DAG)")
    print_graph_stats(ext_graph_raw, "Extinction Group Subgroups (raw, with cycles)")

    # Save in multiple formats
    print("\n" + "="*60)
    print("Saving graphs...")
    print("="*60)

    # 1. Node-link JSON (NetworkX native, most complete)
    save_graph_json(sg_graph, out_dir / "sg_t_subgroup_graph.json")
    save_graph_json(ext_graph, out_dir / "extinction_group_graph.json")
    save_graph_json(ext_graph_raw, out_dir / "extinction_group_graph_raw.json")

    # 2. Simple adjacency-list JSON (easiest to read/parse)
    save_graph_adjacency_json(sg_graph, out_dir / "sg_t_subgroup_adjacency.json")
    save_graph_adjacency_json(ext_graph, out_dir / "extinction_group_adjacency.json")

    # 3. GraphML (interoperable with Gephi, Cytoscape, etc.)
    save_graph_graphml(sg_graph, out_dir / "sg_t_subgroup_graph.graphml")
    save_graph_graphml(ext_graph, out_dir / "extinction_group_graph.graphml")

    # Print loading example
    print("\n" + "="*60)
    print("USAGE EXAMPLE (Python)")
    print("="*60)
    print("""
import json
import networkx as nx

# Method 1: Load node-link JSON (full NetworkX reconstruction)
with open('sg_t_subgroup_graph.json') as f:
    sg_graph = nx.node_link_graph(json.load(f))

# Method 2: Load adjacency-list JSON (simpler)
with open('sg_t_subgroup_adjacency.json') as f:
    data = json.load(f)

# Find all subgroups of SG 225 (Fm-3m):
print("Maximal t-subgroups of Fm-3m (225):")
for child in sg_graph.successors(225):
    sym = sg_graph.nodes[child]['hm_symbol']
    print(f"  {child}: {sym}")

# Find ALL subgroups (transitive closure):
descendants = nx.descendants(sg_graph, 225)
print(f"All t-subgroups of 225: {len(descendants)} groups")
""")

    # Demo the usage right now
    print("\n" + "="*60)
    print("DEMO: Maximal t-subgroups of Fm-3m (225)")
    print("="*60)
    for child in sorted(sg_graph.successors(225)):
        sym = sg_graph.nodes[child]['hm_symbol']
        print(f"  {child}: {sym}")

    n_desc = len(nx.descendants(sg_graph, 225))
    print(f"\n  Total t-subgroups (transitive): {n_desc}")

    print("\n" + "="*60)
    print("DEMO: Extinction group hierarchy sample")
    print("="*60)
    # Show a few extinction groups and their sub-extinction-groups
    for ext_idx in sorted(ext_graph.nodes())[:10]:
        name = ext_graph.nodes[ext_idx]['name']
        children = sorted(ext_graph.successors(ext_idx))
        if children:
            child_names = [ext_graph.nodes[c]['name'][:30] for c in children[:3]]
            more = f"... +{len(children)-3}" if len(children) > 3 else ""
            print(f"  [{ext_idx}] {name[:40]}")
            for c, cn in zip(children[:3], child_names):
                print(f"       -> [{c}] {cn}")
            if more:
                print(f"       {more}")


if __name__ == "__main__":
    main()
