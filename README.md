# Closing the Loop: Superloop and Outer London bus connectivity (COMP0123)

This repository contains code and processed outputs for a COMP0123 (Complex Networks and Online Social Networks) coursework project analysing the structural impact of TfL’s Superloop on Outer London bus connectivity.

## Research question
How does adding Superloop-exclusive connectivity change shortest-path structure and global efficiency for travel between outer-borough stops, compared to an otherwise identical network without those edges?

## Summary of approach
- Build a stop-level **L-space** graph: nodes are stops; edges connect consecutive stops in published route sequences.
- Construct two graphs:
  - **Baseline**: remove *Superloop-exclusive* edges (served by Superloop and by no non-Superloop route).
  - **With Superloop**: full graph including Superloop routes.
- Evaluate changes in shortest-path hop distance and global efficiency via OD sampling.
- Run robustness variants:
  - Paths computed on outer-induced graph vs full London graph (OD pairs remain outer-only)
  - Unweighted hops vs haversine-weighted (km) shortest paths
- Additional diagnostics: degree/mixing, null-model small-world comparison, rich-club, and approximate betweenness rank shifts.

## Repository structure
- `code/` — analysis scripts (graph construction, sampling, figures, tables)
- `data/boroughs/` — London borough boundaries used for outer/inner filtering
- `data/cache_tfl/` — cached route sequences
- `outputs/graphs/` — GEXF graphs used for analysis and Gephi
- `outputs/tables/` — CSV outputs used in the report
- `outputs/figures/` — generated figures
- `figures/` — final figures referenced in the report

## Setup
Tested on macOS with Python 3.

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
