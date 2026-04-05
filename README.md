# Project 1 — CrystalGINE: GNN Property Predictor

A multimodal Graph Neural Network that predicts **band gap** and **formation energy** of crystalline materials directly from atomic structure, trained on stable semiconductors from the [Materials Project](https://materialsproject.org/).

## Data

**Source:** Materials Project via `mp_api`

**Query filters:**
- Band gap: `0.5 – 5.0 eV` (semiconductor range relevant for photovoltaic / LED applications)
- `is_stable=True` — thermodynamically stable phases only

**Features extracted per material:**

| Feature | Type | Description |
|---|---|---|
| `structure` | pymatgen `Structure` | Full atomic positions and lattice |
| `density` | float | Bulk density (g/cm³) |
| `nelements` | int | Number of distinct elements |
| `crystal_system` | str → int | One of 7 Bravais lattice types |
| `band_gap` | float | DFT-computed band gap (eV) — **target 1** |
| `formation_energy_per_atom` | float | Formation energy (eV/atom) — **target 2** |

## Setup

```bash
pip install torch torch-geometric mp-api pymatgen tqdm pandas
```

Set your Materials Project API key:

```bash
export MP_API_KEY="your_key_here"
```

## Context in the Larger Pipeline

This model is **Project 1 of 5** in a computational materials discovery pipeline. Downstream projects build on this:

- **Project 2** (materials literature RAG) a Q&A system that uses a variety of materials science papers to provide cited answers
- **Project 3** (autonomous screener) calls the GNN as a tool to rapidly evaluate candidate materials without DFT
- **Project 4** (multi-agent pipeline) combines both the GNN and RAG system as a callable tool any LLM agent can invoke
- **Project 5** (MCP server) packages the pipeline as a protocol that can be callable by agents in the future