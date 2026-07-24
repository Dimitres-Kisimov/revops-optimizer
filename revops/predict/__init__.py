"""revops.predict — trained ML models whose outputs feed the optimizers.

  * forecast.py   — global demand forecaster (PyTorch MLP on lag+calendar feats)
  * elasticity.py — per-category log-log ridge (from-scratch numpy)
  * risk.py       — SKU decline classifier (from-scratch numpy logistic reg.)
"""
