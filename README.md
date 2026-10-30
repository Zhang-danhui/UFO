# UFO

🎉 Official implementation of UFO: A Unified Flow-Oriented Framework for Robust Continual Graph Learning (NeurIPS 2026).

## Datasets

Use `--dataset` to choose one of the supported datasets. The recommended `--n_cls_per_task` values are:

- `CoraFull-CL`: use `--n_cls_per_task 7`
- `CS-CL`: use `--n_cls_per_task 3`
- `WikiCS-CL`: use `--n_cls_per_task 2`
- `Photo-CL`: use `--n_cls_per_task 2`

## Quick Start

Basic example:

```bash
python train.py --method UFO --dataset CS-CL --n_cls_per_task 3 --label-noise 0.3
```

For pair flipping noise, add `--label-noise-type pair_cycle`:

```bash
python train.py --method UFO --dataset CS-CL --n_cls_per_task 3 --label-noise 0.3 --label-noise-type pair_cycle
```

## References

This project is adapted from [CGLB](https://github.com/QueuQ/CGLB).
The normalizing-flow components are adapted from [nflows](https://github.com/bayesiains/nflows).
We sincerely thank the authors for making their valuable code publicly available.
