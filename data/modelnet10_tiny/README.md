# ModelNet10 Tiny Slice

This folder contains a tiny CPU-light slice of ModelNet10-derived voxel arrays used only for the real-shape benchmark tier.

- `chair.npy.gz`: compressed voxelized chair shapes.
- `monitor.npy.gz`: compressed voxelized monitor shapes.

The arrays were downloaded from the public `SomTambe/ModelNet10-dataset` mirror, which derives from the Princeton ModelNet10 dataset introduced with 3D ShapeNets. The benchmark runner deterministically selects a small subset from each file so the repository can test sparse-view candidate selection on recognized real object categories without downloading the full dataset.
