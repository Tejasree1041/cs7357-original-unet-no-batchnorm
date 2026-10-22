# CS7357 model/training additions

This is the team's original repository with **minimal additions**. Luke's `data_scripts/prepare_data.py` and `data/splits/split_seed42.json` are unchanged. `train/train.py` is also unchanged to avoid overwriting teammate-owned validation/checkpoint code.

## Our scope
- `models/unet.py`: existing skip ON/OFF architecture; removed BatchNorm for a closer 2015 reproduction and added He initialization. Uses same-padding as the research plan.
- `train/losses.py`: weighted cross-entropy consuming the existing `.npy` weight maps, with unweighted mode for the no-border ablation.
- `train/augment.py`: synchronized flips, 90-degree rotations and elastic deformation of image/mask/weight maps, with intensity jitter applied to image only. No-elastic mode retains other transforms.
- `train/dataset.py`: optional transform hook (no change to the original dataset path or split handling).
- `train/trainer.py`: simple configurable one-epoch/full-epoch training loop. Evaluation/checkpointing/logging remain with the assigned teammates.

## Important existing data-prep limitation
The current `prepare_data.py` computes weights from the **merged binary boundary**, not from distances to the two nearest separate nucleus instances. This differs from the original U-Net paper's border weight formula. Discuss a fix with Luke, then **regenerate weight maps only**, while keeping the committed split JSON unchanged. Do not rerun its `main()` blindly: it reshuffles the filesystem enumeration and could overwrite the split.

## Running in Colab or locally
From the repo root, after Luke's processed dataset is available in `data/processed/{images,masks,weights}`:

```bash
pip install -r requirements.txt
python -m train.trainer --model full --epochs 1 --batch-size 1
```

Once one epoch succeeds, run with `--epochs 50 --batch-size 8` (reduce batch size if out of memory). Other modes: `no_skips`, `no_elastic`, `no_border`. For the full experiment repeat seeds 42, 43, 44 **without changing the split JSON**.

## Important
No real-dataset training or metrics have been run in this package. A synthetic CPU smoke test was performed separately. The full 64–1024-channel U-Net can require substantial RAM/VRAM.
