## Dynafomer – Copilot Instructions for AI Coding Agents

Purpose: Make agents productive quickly in this repo by codifying the architecture, workflows, and repo-specific conventions. Keep changes minimal and aligned with existing patterns.

### Big Picture
- Graphormer-based model on Fairseq: Custom Fairseq model `graphormer` with 3D encoders in `Dynaformer/dynaformer`. Tasks and criterions are Fairseq-registered and invoked via `fairseq-train`.
- Data flow (PyG): PyG/DGL graphs → `collator.py` pads/assembles tensors → Fairseq task (`graph_prediction[_with_flag]`) → `models/graphormer.py` → loss in `criterions/*`.
- Evaluation: `evaluate/evaluate.py` loads checkpoints from a folder, runs inference, applies a fixed linear rescaling to predictions, writes CSVs and logs metrics.

Key directories/files:
- `Dynaformer/dynaformer/models/graphormer.py`: Fairseq model + 3D distance head, fingerprint fusion.
- `Dynaformer/dynaformer/tasks/graph_prediction.py`: Fairseq tasks `graph_prediction` and `graph_prediction_with_flag` (FLAG adversarial training).
- `Dynaformer/dynaformer/criterions/*.py`: `*_with_flag` variants match tasks with FLAG.
- `Dynaformer/dynaformer/data/pyg_datasets/pyg_dataset_lookup_table.py`: Parses dataset spec strings and builds PyG datasets (PDBBind, MD, Hybrid, Custom).
- `Dynaformer/dynaformer/data/collator.py`: Batching/padding; adds node/edge features and 2040-dim fingerprints when enabled.
- `Dynaformer/dynaformer/evaluate/evaluate.py`: Enumerates `*.pt` in `--save-dir`, evaluates, rescales predictions, writes CSV.

### Environment & Build
- Use provided installer from project root:
  ```bash
  cd Dynaformer
  conda create -n dynaformer python=3.9 -y
  conda activate dynaformer
  ./install.sh  # pins torch 1.10.0 + cu113 and builds fairseq + Cython
  ```
- `install.sh` compiles Fairseq extensions and Cython module `dynaformer/data/algos.pyx`. CUDA 11.3-compatible environment expected.

### Datasets (Spec Strings)
- Datasets are specified as strings parsed by `PYGDatasetLookupTable.GetPYGDataset`:
  - PDBBind: `pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=0`
  - MD: `mddata:set_name=md-refined2019-5-5-5,seed=2022`
  - Hybrid: `hybrid:set_name=md-refined2019-5-5-5+general-set-2019-coreset-2016,cutoffs=5-5-5,seed=2022`
  - Custom (preprocessed pickle): `custom:path=/abs/path/to/data.pkl`
- Data will be downloaded/processed into `--data-path` (default similar to `./data`).

### Training
- Reference script: `Dynaformer/examples/md_pretrain/md_train.sh` (a more general version of root `md_train.sh`). It launches Fairseq with distributed training:
  ```bash
  python -m torch.distributed.launch --nproc_per_node=${n_gpu} \
    $(which fairseq-train) \
    --user-dir "$(realpath ./dynaformer)" \
    --task graph_prediction_with_flag --criterion l2_loss_with_flag \
    --dataset-source pyg --dataset-name "<dataset_spec>" --data-path "<path>" \
    --arch graphormer_base --num-classes 1 \
    --encoder-layers 4 --encoder-attention-heads 32 \
    --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
    --lr 1e-4 --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \
    --warmup-updates <steps> --total-num-update <steps> --max-update <steps> \
    --batch-size 20 --update-freq 1 --weight-decay 1e-5 --clip-norm 5 \
    --max-nodes 600 --dist-head gbf3d --num-dist-head-kernel 256 --num-edge-types 16384 \
    --fingerprint --fp16 --save-dir "<save_dir>"
  ```
- Flags and conventions:
  - `--user-dir` must point to `Dynaformer/dynaformer` to load custom Fairseq components.
  - `*_with_flag` task/criterion pairs enable FLAG adversarial training; set with `--flag-m/--flag-step-size/--flag-mag`.
  - 3D head: `--dist-head {none,gbf,gbf3d,bucket,embed3d}`; Dynaformer uses `gbf3d` with `--num-dist-head-kernel 256`, `--num-edge-types 16384`.
  - `--fingerprint` enables a 2040-dim feature branch (RFScore+GBScore+ECIF) fused in the model (`fpnn` in `graphormer.py`).
  - Scripts construct a descriptive `save_dir` and set wandb env vars (`WANDB_NAME`, `WANDB_SAVE_DIR`).

### Evaluation & Custom Input
- Evaluate checkpoints: from project root
  ```bash
  ./run_evaluate.sh  # CASF-2016 and CASF-2013 using checkpoints in ./checkpoint
  ```
- Script `Dynaformer/examples/evaluate/evaluate.sh` wraps `evaluate.py` with fixed model dims and `--fingerprint`.
- Important: `evaluate.py` rescales predictions before metrics: `y_pred = y_pred * 1.9919705951218716 + 6.529300030461668`.
- Custom input flow:
  1) Build `.pkl` with `preprocess/custom_input.py` from PDB/SDF and a CSV schema; 2) Run `./run_custom_input.sh` which calls `evaluate.sh` with `dataset-name "custom:path=...pkl"`.

### Extending the Codebase
- New dataset:
  - Prefer adding a spec in `pyg_dataset_lookup_table.py` to parse a new `name:` and return train/valid/test PyG datasets.
  - Alternatively register via `DATASET_REGISTRY` in `dynaformer/data/__init__.py` and pass `--user-data-dir`.
- New model/head:
  - Extend `dynaformer/modules/*` and wire in `dynaformer/models/graphormer.py`; expose args via `add_args` and default via `@register_model_architecture`.
- New loss:
  - Add under `dynaformer/criterions/` and register with `@register_criterion("name")`; add a `*_with_flag` variant if needed.

### Gotchas
- Keep `--user-dir` correct; otherwise Fairseq won’t find custom tasks/models.
- Max nodes: training/eval scripts use `--max-nodes 600`; `collator.py` asserts no sample exceeds this after filtering.
- Version pins in `install.sh` matter (Torch 1.10/CUDA 11.3; torch-scatter/torch-sparse versions); changing them can break builds.

### Quick Commands
- Install env: see above.
- Evaluate checkpoints: `./run_evaluate.sh`.
- Train (example): `bash Dynaformer/examples/md_pretrain/md_train.sh` (override env vars like `dataset_name`, `data_path`, `save_path`, `n_gpu`, etc.).
