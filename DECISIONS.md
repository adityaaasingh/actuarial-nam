# Decisions

- **torch 2.2.2 and numpy<2, pinned everywhere.** 

Paper: not specified. Choice: the last PyTorch release with Intel Mac wheels, used on every platform so local, Colab, Docker and CI runs match.

- **pyarrow pinned to 21.0.0.** Newer pyarrow requires NumPy 2, which torch 2.2.2 (last Intel Mac build) can't use.

Schema bounds are plausible ranges, wider than the observed min/max; category levels are strict.