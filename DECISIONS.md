# Decisions

- **torch 2.2.2 and numpy<2, pinned everywhere.** Paper: not specified. Choice: the last PyTorch release with Intel Mac wheels, used on every platform so local, Colab, Docker and CI runs match.