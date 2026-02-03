# VisionTransformer (CIFAR-10)

Minimal Vision Transformer implementation and training script for CIFAR-10.

**Project layout**
- `VisionTransformer_main.py`: training entrypoint (builds model, runs training/validation, saves checkpoint).
- `Utils.py`: dataset transforms and DataLoader helpers.
- `models/`: transformer components (`SimpleTransformer`, `TransformerBlock`, etc.).
- `TrainValidateWrapper.py`: training/validation wrapper used by the main script.
- `checkpoint/`: folder where model checkpoints are written (`visiontrans_model.pt`).

**Quick start**

1. (Recommended) Create a conda env and install dependencies. Example using pip:

```bash
python -m pip install torch torchvision tqdm numpy
```

2. Run training (uses CUDA if available):

```bash
python VisionTransformer_main.py
```

3. Inspect saved checkpoint:

```bash
python - <<PY
import torch
ckpt = torch.load('checkpoint/visiontrans_model.pt')
print('epoch', ckpt.get('epoch'))
print('test_acc', ckpt.get('test_acc'))
PY
```

**Evaluation / export**
- Add/modify a small `evaluate.py` to load the checkpoint and run the test set.
- For deployment, export with TorchScript or ONNX from the loaded model.

**Notes & recommendations**
- The training script will save checkpoints to `checkpoint/visiontrans_model.pt` when validation improves.
- Large model checkpoints and dataset files are excluded from version control by `.gitignore`. Use Git LFS for checkpoint files if you want to store them in the repo:

```bash
git lfs install
git lfs track "checkpoint/*.pt"
git add .gitattributes
```

- To increase GPU throughput, consider enabling AMP (automatic mixed precision) and `torch.backends.cudnn.benchmark = True` in `VisionTransformer_main.py`.

**Pushing to GitHub (example)**

```bash
git init
git add .
git commit -m "Initial commit: vision transformer training"
git remote add origin <your-repo-url>
git push -u origin main
```

Replace `<your-repo-url>` with your repository URL. If you tracked checkpoints with LFS, they'll be pushed via LFS.

**Contact / next steps**
- Run a final evaluation with the saved checkpoint and share the reported `test_acc` if you want help exporting or pruning the model for deployment.
