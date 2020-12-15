# Clothes_Change_Person_ReID

A baseline for clothes-change Person ReID task. 

The code is based on [Fast-reid](https://github.com/JDAI-CV/fast-reid). 

## News
+ [2020.12.15] LTCC datasets is supported.

## Installation

See [INSTALL.md](https://github.com/JDAI-CV/fast-reid/blob/master/docs/INSTALL.md).

## Datasets

The clothes-change datasets include:

+ [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html#)
+ [VC-Clothes](https://wanfb.github.io/dataset.html)

## Quick Start

**Train:**

```bash
python tools/train_net.py --config-file ./configs/LTCC/bagtricks_R50.yml --num-gpus 2
```

**Test:**

```bash
python tools/train_net.py --config-file ./configs/LTCC/bagtricks_R50.yml --eval-only \
MODEL.WEIGHTS /path/to/checkpoint_file MODEL.DEVICE "cuda:0"
```

For more options, see `./tools/train_net.py -h`.