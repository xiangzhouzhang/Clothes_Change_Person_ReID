# Clothes_Change_Person_ReID

A baseline for clothes-change Person ReID task. 

This code is implemented based on [JD Fast-reid](https://github.com/JDAI-CV/fast-reid). 

## News
+ [2020.12.15] LTCC dataset is supported.
+ [2020.12.16] Add evaluation code for clothes-change setting.
+ [2020.12.16] Train without/with clothes labels.

## Todo list

- [ ] Add human keypoints 
- [ ] Add human mask

## Installation

See [INSTALL.md](https://github.com/JDAI-CV/fast-reid/blob/master/docs/INSTALL.md).

## Datasets

The clothes-change datasets include:

+ LTCC [[dataset](https://naiq.github.io/LTCC_Perosn_ReID.html#)] [[paper](https://arxiv.org/abs/2005.12633)] : 17119 images from 152 identities
+ VC-Clothes [[dataset](https://wanfb.github.io/dataset.html)] [[paper](https://arxiv.org/pdf/2003.04070.pdf)]: 19060 images from 512 identities (Virtual) + 4324 images from 28 identities (Real)
+ PRCC [[dataset](https://www.isee-ai.cn/~yangqize/clothing.html)] [[paper](https://www.isee-ai.cn/~yangqize/main_document.pdf)]: 33698 images from 221 identities

Our experiments are based on LTCC. You can place LTCC dataset to make the data folder like:

  ~~~
  ${ROOT}
  |-- datasets
  `-- |-- LTCC_ReID
      `-- |-- train
          |--- test
          |--- query
          `--- info
      |-- Other_Dataset
  ~~~

## Quick Start

**Train:**

Single-GPU

```bash
python tools/train_net.py --config-file ./configs/LTCC/bagtricks_R50.yml MODEL.DEVICE "cuda:0"
```

Multi-GPU

```bash
python tools/train_net.py --config-file ./configs/LTCC/bagtricks_R50.yml --num-gpus 4
```

**Test:**

Standard Setting: The images with the same identity and the same camera view are discarded during testing.

```bash
python tools/train_net.py --config-file ./configs/LTCC/bagtricks_R50.yml --eval-only \
MODEL.WEIGHTS /path/to/checkpoint_file MODEL.DEVICE "cuda:0"
```

Cloth-changing Setting: The images with same identity, camera view and *clothes* are discarded during testing.

```
python tools/train_net.py --config-file ./configs/LTCC/bagtricks_R50.yml --eval-only --cconly \
MODEL.WEIGHTS /path/to/checkpoint_file MODEL.DEVICE "cuda:0"
```

For more options, see `python ./tools/train_net.py -h`.

## Baseline Results 

We provide some baseline results and trained models available for download:

The model is trained for 120 epochs with batch size=64 on 2 GeForce GTX TITAN X.

<table>
<thead>
  <tr>
    <th rowspan="3" align="center">Method</th>
    <th rowspan="3" align="center">Backbone</th>
    <th colspan="3" align="center">Standard</th>
    <th colspan="3" align="center">Cloth-changing</th>  
    <th rowspan="3" align="center">download</th>
  </tr>
  <tr>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
    <td align="center">mAP</td>
    <td align="center">Rank@5</td>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td nowrap align="center">Baseline</td>
    <td align="center">R50</td>
    <td align="center">67.55%</td>
    <td align="center">77.48%</td>
    <td align="center">32.64%</td>
    <td align="center">33.93%</td>
    <td align="center">49.49%</td>
    <td align="center">15.57%</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td nowrap align="center">Baseline(w/ clo)</td>
    <td align="center">R50</td>
    <td align="center">73.43%</td>
    <td align="center">81.74%</td>
    <td align="center">38.54%</td>
    <td align="center">31.89%</td>
    <td align="center">48.47%</td>
    <td align="center">15.47%</td>
    <td align="center"><a href="https://drive.google.com/file/d/1w2qYOpWVzZInYZlkUZmxRTC4uW1I33-Y/view?usp=sharing">Model</a></td>
  </tr>
</tbody>
</table>

