#!/bin/bash
#SBATCH -J topo-flow-champion
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A CDA24014
#SBATCH -o /scratch/ALLOC3/UserE/topology_flow_assets/topo_flow_champion_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/topology_flow_assets/topo_flow_champion_%j.err

set -euo pipefail

ASSET_DIR=/scratch/ALLOC3/UserE/topology_flow_assets
OUT_DIR=/scratch/ALLOC3/UserE/topology_flow_assets/output

mkdir -p "$OUT_DIR"

python3 "$ASSET_DIR/analyze_topological_error_distance.py" \
  --graph-json "$ASSET_DIR/extinction_group_adjacency.json" \
  --failure-json "$ASSET_DIR/mixed2500k_compare_325_failure_modes_655279.json" \
  --output-json "$ASSET_DIR/mixed2500k_topological_error_summary_325_655279.json"

python3 "$ASSET_DIR/plot_extinction_topology_flow.py" \
  --graph-json "$ASSET_DIR/extinction_group_adjacency.json" \
  --canonical-csv "$ASSET_DIR/canonical_extinction_to_space_group.csv" \
  --model "Champion_Mixed_2500k:$ASSET_DIR/mixed2500k_compare_325_failure_modes_655279.json:$ASSET_DIR/mixed2500k_topological_error_summary_325_655279.json" \
  --output-dir "$OUT_DIR" \
  --animation-frames

echo "Outputs:"
ls -lah "$OUT_DIR"
