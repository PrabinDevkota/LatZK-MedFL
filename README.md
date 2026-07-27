# LatZK-MedFL

**Dual-verification federated learning with lattice-based proofs for IoMT hospital networks.**

Public code companion for the IEEE Access manuscript:

> *LatZK-MedFL: Dual-Verification Federated Learning with Lattice-Based Proofs for IoMT Hospital Networks*

Repository: [https://github.com/PrabinDevkota/LatZK-MedFL](https://github.com/PrabinDevkota/LatZK-MedFL)

## What this is

A **reproducible protocol prototype** (not a production zk-SNARK / NIST Level-3 library) that:

1. Runs non-IID hospital FedAvg on Breast Cancer Wisconsin data  
2. Attaches lattice-style **client norm proofs** + **aggregator correctness proofs**  
3. Evaluates FedAvg / Krum / DP-FL under large-norm, label-flip, hybrid, and aggregator-forgery attacks  
4. Exports journal figures (PNG+PDF) and CSV/JSON tables under `results/`

## Quick start (Google Colab Pro recommended)

1. Open [`implementation/LatZK_MedFL_Colab.ipynb`](implementation/LatZK_MedFL_Colab.ipynb) in Colab  
2. Set `FAST_MODE = False` for the 7-seed journal campaign (~45–90 min GPU)  
3. Set `FAST_MODE = True` for a short smoke test  
4. Download `results/results.zip` when finished  

### Local run

```bash
pip install -r requirements.txt
# then open the notebook in Jupyter, or rebuild it:
python implementation/build_colab_notebook.py
```

## Repository layout

```
implementation/
  LatZK_MedFL_Colab.ipynb     # primary experiment notebook
  build_colab_notebook.py     # regenerates the notebook
  run_strengthened_experiments.py
  polish_figures.py
requirements.txt
README.md
LICENSE
```

Large literature dumps, local result folders, and PDF build artifacts are excluded via `.gitignore` (re-run the notebook to regenerate figures/tables).

## Headline results (7 seeds, demo lattice params)

| Setting | Acc | Client reject | Agg reject |
|---------|-----|---------------|------------|
| LatZK-MedFL clean | 0.957 | 0.00 | 0.00 |
| FedAvg hybrid | 0.331 | 0.00 | 0.00 |
| LatZK-MedFL hybrid | 0.946 | 0.25 | 0.00 |
| LatZK-MedFL aggregator forgery | fail-closed | 0.00 | 1.00 |
| Communication vs FedAvg | ~1.17× | | |

## Honest limitations

- Demo lattice parameters — **not** claimed as production post-quantum security  
- Pure label-flip inside the ℓ₂ ball is **not** detected by the norm gate  
- Primary dataset is clinical tabular (Wisconsin); secondary task is synthetic EHR-like  

## Citation

If you use this code, please cite the accompanying manuscript (IEEE Access submission) and this repository.

## License

MIT — see [LICENSE](LICENSE).
