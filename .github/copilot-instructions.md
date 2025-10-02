## Dynaformer – Copilot Instructions for AI Coding Agents## Dynaformer – Copilot Instructions for AI Coding Agents## Dynaformer – Copilot Instructions for AI Coding Agents## Dynaformer – Copilot Instructions for AI Coding Agents



Purpose: Modernize Dynaformer by extracting core components from the legacy Fairseq implementation and building a PyTorch-native training pipeline with custom datasets.



### Project GoalPurpose: Modernize Dynaformer by extracting core components from the legacy Fairseq implementation and building a PyTorch-native training pipeline with custom datasets.



**Legacy → Modern**: Extract Dynaformer's core neural architecture from the deprecated Fairseq framework and implement custom PyTorch training with domain-specific datasets. The original Fairseq-based implementation has compatibility issues; this modernization focuses on the essential model components.



### Architecture Overview### Project GoalPurpose: Modernize Dynaformer by extracting core components from the legacy Fairseq implementation and building a PyTorch-native training pipeline with custom datasets.Purpose: Modernize Dynaformer by extractin### Gotchas



- **Core Model**: Graphormer with 3D encoders extracted to `src/models/graphormer.py` (✅ **Modernized** - Pure PyTorch implementation)**Legacy → Modern**: Extract Dynaformer's core neural architecture from the deprecated Fairseq framework and implement custom PyTorch training with domain-specific datasets. The original Fairseq-based implementation has compatibility issues; this modernization focuses on the essential model components.

- **Data Flow**: PyG graphs → `src/data/collator.py` (batching/padding) → Graphormer model → loss functions in `src/criterions/`

- **Custom Datasets**: Merge Dynaformer's PyG dataset patterns (`src/data/pyg_datasets/`) with domain-specific splits from `bingsong_project/`- Keep `--user-dir` correct; otherwise Fairseq won't find custom tasks/models.



### Extracted Core Components### Architecture Overview



- `src/models/graphormer.py`: Main Graphormer model (✅ **Modernized** - Pure PyTorch implementation)- **Core Model**: Graphormer with 3D encoders extracted to `src/models/graphormer.py` (✅ **Modernized** - Pure PyTorch implementation)### Project Goal- Max nodes: training/eval scripts use `--max-nodes 600`; `collator.py` asserts no sample exceeds this after filtering.

- `src/modules/`: Graphormer layers (✅ **Modernized** - PyTorch 2.6+ compatible)

- `src/data/collator.py`: Graph batching with 2040-dim fingerprint support and 3D position handling- **Data Flow**: PyG graphs → `src/data/collator.py` (batching/padding) → Graphormer model → loss functions in `src/criterions/`

- `src/data/pyg_datasets/`: PyG dataset loaders for molecular data (PDBBind patterns)

- `src/criterions/`: Loss functions (L1, L2, binary) with normalization constants (✅ **Modernized** - Pure PyTorch implementation)- **Custom Datasets**: Merge Dynaformer's PyG dataset patterns (`src/data/pyg_datasets/`) with domain-specific splits from `bingsong_project/`**Legacy → Modern**: Extract Dynaformer's core neural architecture from the deprecated Fairseq framework and implement custom PyTorch training with domain-specific datasets. The original Fairseq-based implementation has compatibility issues; this modernization focuses on the essential model components.- Version pins in `install.sh` matter (Torch 1.10/CUDA 11.3; torch-scatter/torch-sparse versions); changing them can break builds.

- `src/evaluate/evaluate.py`: Model evaluation with prediction rescaling

- `bingsong_project/`: Custom dataset definitions with train-test splitting logic



### Modernization Tasks### Extracted Core Components



1. **✅ Remove Fairseq Dependencies**: Convert `src/models/graphormer.py` from `FairseqEncoderModel` to pure PyTorch `nn.Module`- `src/models/graphormer.py`: Main Graphormer model (✅ **Modernized** - Pure PyTorch implementation)

2. **✅ Modernize Modules**: Convert `src/modules/` to PyTorch 2.6+ compatible implementations

3. **✅ Modernize Criterions**: Convert `src/criterions/` to pure PyTorch loss functions without Fairseq dependencies- `src/modules/`: Graphormer layers (✅ **Modernized** - PyTorch 2.6+ compatible)### Architecture Overview### Quick Commands

4. **Custom Training Loop**: Replace `fairseq-train` with PyTorch training logic using modern criterions

5. **Dataset Integration**: Merge `bingsong_project/dataset_GIGN_benchmark_davis_complete.py` patterns with `src/data/pyg_datasets/`- `src/data/collator.py`: Graph batching with 2040-dim fingerprint support and 3D position handling

6. **Configuration System**: Replace Fairseq args parsing with modern config management (hydra/argparse)

- `src/data/pyg_datasets/`: PyG dataset loaders for molecular data (PDBBind patterns)- **Core Model**: Graphormer with 3D encoders extracted to `src/models/graphormer.py` (still has Fairseq dependencies to remove)- Install env: see above.

### Model Architecture Details

- `src/criterions/`: Loss functions (L1, L2, binary) with normalization constants

- **3D Encoders**: `src/modules/graphormer_3d_encoder.py` handles spatial relationships with GBF3D distance heads

- **Attention**: Custom multihead attention in `src/modules/multihead_attention.py` - `src/evaluate/evaluate.py`: Model evaluation with prediction rescaling- **Data Flow**: PyG graphs → `src/data/collator.py` (batching/padding) → Graphormer model → loss functions in `src/criterions/`- Evaluate checkpoints: `./run_evaluate.sh`.

- **Fingerprints**: 2040-dim molecular fingerprints (RFScore+GBScore+ECIF) integrated via `collator.py`

- **Graph Encoding**: Node/edge features processed through `src/modules/graphormer_graph_encoder.py`- `bingsong_project/`: Custom dataset definitions with train-test splitting logic

- **Key Parameters**: 

  - `max_nodes=600`, `encoder_layers=4`, `encoder_attention_heads=32`- **Custom Datasets**: Merge Dynaformer's PyG dataset patterns (`src/data/pyg_datasets/`) with domain-specific splits from `bingsong_project/`- Train (example): `bash Dynaformer/examples/md_pretrain/md_train.sh` (override env vars like `dataset_name`, `data_path`, `save_path`, `n_gpu`, etc.).mponents from the legacy Fairseq implementation and building a PyTorch-native training pipeline with custom datasets.

  - `dist_head=gbf3d`, `num_dist_head_kernel=256`, `num_edge_types=16384`

### Modernization Tasks

### Modern Model Usage

1. **✅ Remove Fairseq Dependencies**: Convert `src/models/graphormer.py` from `FairseqEncoderModel` to pure PyTorch `nn.Module`

```python

from src.models import GraphormerModel, GraphormerConfig, get_graphormer_base_config2. **Custom Training Loop**: Replace `fairseq-train` with PyTorch training logic using `src/criterions/` losses



# Create model with predefined config3. **Dataset Integration**: Merge `bingsong_project/dataset_GIGN_benchmark_davis_complete.py` patterns with `src/data/pyg_datasets/`### Extracted Core Components### Project Goal

config = get_graphormer_base_config()

config.fingerprint = True  # Enable fingerprint features4. **Configuration System**: Replace Fairseq args parsing with modern config management (hydra/argparse)

model = GraphormerModel(config)

- `src/models/graphormer.py`: Main Graphormer model (needs Fairseq dependency removal)**Legacy → Modern**: Extract Dynaformer's core neural architecture from the deprecated Fairseq framework and implement custom PyTorch training with domain-specific datasets. The original Fairseq-based implementation has compatibility issues; this modernization focuses on the essential model components.

# Or create from legacy args (for compatibility)

model = GraphormerModel.from_args(args)### Model Architecture Details



# Forward pass- **3D Encoders**: `src/modules/graphormer_3d_encoder.py` handles spatial relationships with GBF3D distance heads- `src/modules/`: Graphormer layers (3D encoders, attention, graph encoder)

output = model(batched_data)

```- **Attention**: Custom multihead attention in `src/modules/multihead_attention.py` 



### Modern Criterion Usage- **Fingerprints**: 2040-dim molecular fingerprints (RFScore+GBScore+ECIF) integrated via `collator.py`- `src/data/collator.py`: Graph batching with 2040-dim fingerprint support and 3D position handling### Architecture Overview



```python- **Graph Encoding**: Node/edge features processed through `src/modules/graphormer_graph_encoder.py`

from src.criterions import get_criterion, L2Loss, L1LossWithFlag

- **Key Parameters**: - `src/data/pyg_datasets/`: PyG dataset loaders for molecular data (PDBBind patterns)- **Core Model**: Graphormer with 3D encoders extracted to `src/models/graphormer.py` (still has Fairseq dependencies to remove)

# Create criterion instances

l2_criterion = L2Loss()  - `max_nodes=600`, `encoder_layers=4`, `encoder_attention_heads=32`

l1_flag_criterion = L1LossWithFlag()

  - `dist_head=gbf3d`, `num_dist_head_kernel=256`, `num_edge_types=16384`- `src/criterions/`: Loss functions (L1, L2, binary) with normalization constants- **Data Flow**: PyG graphs → `src/data/collator.py` (batching/padding) → Graphormer model → loss functions in `src/criterions/`

# Or use registry

criterion = get_criterion("l2_loss_with_flag")()



# Training loop### Modern Model Usage- `src/evaluate/evaluate.py`: Model evaluation with prediction rescaling- **Custom Datasets**: Merge Dynaformer's PyG dataset patterns (`src/data/pyg_datasets/`) with domain-specific splits from `bingsong_project/`

for batch in dataloader:

    result = criterion(model, batch)```python

    loss = result.loss

    loss.backward()from src.models import GraphormerModel, GraphormerConfig, get_graphormer_base_config- `bingsong_project/`: Custom dataset definitions with train-test splitting logic

    optimizer.step()

```



### Dataset Patterns & Integration# Create model with predefined config### Extracted Core Components



- **Legacy Specs**: String-based dataset specs like `"pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5"`config = get_graphormer_base_config()

- **Custom Splits**: `bingsong_project/dta_davis_complete.py` provides domain-specific train/test splitting:

  - `create_fold()`: Standard k-fold cross-validationconfig.fingerprint = True  # Enable fingerprint features### Modernization Tasks- `src/models/graphormer.py`: Main Graphormer model (needs Fairseq dependency removal)

  - `create_fold_setting_cold()`: Cold-start drug/target scenarios  

  - `create_seq_identity_fold()`: Sequence identity-based splitsmodel = GraphormerModel(config)

- **Data Processing**: PyG `Data` objects with molecular graphs, 3D coordinates, and binding affinity targets

- **Normalization**: Targets normalized as `(y - 6.529300030461668) / 1.9919705951218716` in loss functions1. **Remove Fairseq Dependencies**: Convert `src/models/graphormer.py` from `FairseqEncoderModel` to pure PyTorch `nn.Module`- `src/modules/`: Graphormer layers (3D encoders, attention, graph encoder)



### Environment & Build# Or create from legacy args (for compatibility)



- Use provided installer from project root:model = GraphormerModel.from_args(args)2. **Custom Training Loop**: Replace `fairseq-train` with PyTorch training logic using `src/criterions/` losses- `src/data/collator.py`: Graph batching with 2040-dim fingerprint support and 3D position handling

  ```bash

  cd Dynaformer

  conda create -n dynaformer python=3.9 -y

  conda activate dynaformer# Forward pass3. **Dataset Integration**: Merge `bingsong_project/dataset_GIGN_benchmark_davis_complete.py` patterns with `src/data/pyg_datasets/`- `src/data/pyg_datasets/`: PyG dataset loaders for molecular data (PDBBind patterns)

  ./install.sh  # pins torch 1.10.0 + cu113 and builds fairseq + Cython

  ```output = model(batched_data)

- `install.sh` compiles Fairseq extensions and Cython module `dynaformer/data/algos.pyx`. CUDA 11.3-compatible environment expected.

```4. **Configuration System**: Replace Fairseq args parsing with modern config management (hydra/argparse)- `src/criterions/`: Loss functions (L1, L2, binary) with normalization constants

### Datasets (Spec Strings)



- Datasets are specified as strings parsed by `PYGDatasetLookupTable.GetPYGDataset`:

  - PDBBind: `pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=0`### Dataset Patterns & Integration- `src/evaluate/evaluate.py`: Model evaluation with prediction rescaling

  - MD: `mddata:set_name=md-refined2019-5-5-5,seed=2022`

  - Hybrid: `hybrid:set_name=md-refined2019-5-5-5+general-set-2019-coreset-2016,cutoffs=5-5-5,seed=2022`- **Legacy Specs**: String-based dataset specs like `"pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5"`

  - Custom (preprocessed pickle): `custom:path=/abs/path/to/data.pkl`

- Data will be downloaded/processed into `--data-path` (default similar to `./data`).- **Custom Splits**: `bingsong_project/dta_davis_complete.py` provides domain-specific train/test splitting:### Model Architecture Details- `bingsong_project/`: Custom dataset definitions with train-test splitting logic



### Training  - `create_fold()`: Standard k-fold cross-validation



- Reference script: `Dynaformer/examples/md_pretrain/md_train.sh` (a more general version of root `md_train.sh`). It launches Fairseq with distributed training:  - `create_fold_setting_cold()`: Cold-start drug/target scenarios  - **3D Encoders**: `src/modules/graphormer_3d_encoder.py` handles spatial relationships with GBF3D distance heads

- Finetuning: `Dynaformer/examples/finetune/finetune.sh` starts from a pretrained checkpoint with `--finetune-from-model`

  ```bash  - `create_seq_identity_fold()`: Sequence identity-based splits

  python -m torch.distributed.launch --nproc_per_node=${n_gpu} \

    $(which fairseq-train) \- **Data Processing**: PyG `Data` objects with molecular graphs, 3D coordinates, and binding affinity targets- **Attention**: Custom multihead attention in `src/modules/multihead_attention.py` ### Modernization Tasks

    --user-dir "$(realpath ./dynaformer)" \

    --task graph_prediction_with_flag --criterion l2_loss_with_flag \- **Normalization**: Targets normalized as `(y - 6.529300030461668) / 1.9919705951218716` in loss functions

    --dataset-source pyg --dataset-name "<dataset_spec>" --data-path "<path>" \

    --arch graphormer_base --num-classes 1 \- **Fingerprints**: 2040-dim molecular fingerprints (RFScore+GBScore+ECIF) integrated via `collator.py`1. **Remove Fairseq Dependencies**: Convert `src/models/graphormer.py` from `FairseqEncoderModel` to pure PyTorch `nn.Module`

    --encoder-layers 4 --encoder-attention-heads 32 \

    --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \### Environment & Build

    --lr 1e-4 --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \

    --warmup-updates <steps> --total-num-update <steps> --max-update <steps> \- Use provided installer from project root:- **Graph Encoding**: Node/edge features processed through `src/modules/graphormer_graph_encoder.py`2. **Custom Training Loop**: Replace `fairseq-train` with PyTorch training logic using `src/criterions/` losses

    --batch-size 20 --update-freq 1 --weight-decay 1e-5 --clip-norm 5 \

    --max-nodes 600 --dist-head gbf3d --num-dist-head-kernel 256 --num-edge-types 16384 \  ```bash

    --fingerprint --fp16 --save-dir "<save_dir>"

  ```  cd Dynaformer- **Key Parameters**: 3. **Dataset Integration**: Merge `bingsong_project/dataset_GIGN_benchmark_davis_complete.py` patterns with `src/data/pyg_datasets/`

- Flags and conventions:

  - `--user-dir` must point to `Dynaformer/dynaformer` to load custom Fairseq components.  conda create -n dynaformer python=3.9 -y

  - `*_with_flag` task/criterion pairs enable FLAG adversarial training; set with `--flag-m/--flag-step-size/--flag-mag`.

  - 3D head: `--dist-head {none,gbf,gbf3d,bucket,embed3d}`; Dynaformer uses `gbf3d` with `--num-dist-head-kernel 256`, `--num-edge-types 16384`.  conda activate dynaformer  - `max_nodes=600`, `encoder_layers=4`, `encoder_attention_heads=32`4. **Configuration System**: Replace Fairseq args parsing with modern config management (hydra/argparse)

  - `--fingerprint` enables a 2040-dim feature branch (RFScore+GBScore+ECIF) fused in the model (`fpnn` in `graphormer.py`).

  - Scripts construct a descriptive `save_dir` and set wandb env vars (`WANDB_NAME`, `WANDB_SAVE_DIR`).  ./install.sh  # pins torch 1.10.0 + cu113 and builds fairseq + Cython



### Evaluation & Custom Input  ```  - `dist_head=gbf3d`, `num_dist_head_kernel=256`, `num_edge_types=16384`



- Evaluate checkpoints: from project root- `install.sh` compiles Fairseq extensions and Cython module `dynaformer/data/algos.pyx`. CUDA 11.3-compatible environment expected.

  ```bash

  ./run_evaluate.sh  # CASF-2016 and CASF-2013 using checkpoints in ./checkpoint### Model Architecture Details

  ```

- Script `Dynaformer/examples/evaluate/evaluate.sh` wraps `evaluate.py` with fixed model dims and `--fingerprint`.### Datasets (Spec Strings)

- Important: `evaluate.py` rescales predictions before metrics: `y_pred = y_pred * 1.9919705951218716 + 6.529300030461668`.

- Custom input flow:- Datasets are specified as strings parsed by `PYGDatasetLookupTable.GetPYGDataset`:### Dataset Patterns & Integration- **3D Encoders**: `src/modules/graphormer_3d_encoder.py` handles spatial relationships with GBF3D distance heads

  1) Install additional dependencies: `conda install -c conda-forge pymol-open-source openbabel -y` under `dynaformer` environment

  2) Build `.pkl` with `preprocess/custom_input.py` from PDB/SDF and a CSV schema  - PDBBind: `pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=0`

  3) Run `./run_custom_input.sh` which calls `evaluate.sh` with `dataset-name "custom:path=...pkl"`

- Custom input processes pockets on-the-fly using PyMOL/OpenBabel, so results may differ slightly from CASF evaluation (which uses pre-provided pocket files).  - MD: `mddata:set_name=md-refined2019-5-5-5,seed=2022`- **Legacy Specs**: String-based dataset specs like `"pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5"`- **Attention**: Custom multihead attention in `src/modules/multihead_attention.py` 



### Gotchas  - Hybrid: `hybrid:set_name=md-refined2019-5-5-5+general-set-2019-coreset-2016,cutoffs=5-5-5,seed=2022`



- Keep `--user-dir` correct; otherwise Fairseq won't find custom tasks/models.  - Custom (preprocessed pickle): `custom:path=/abs/path/to/data.pkl`- **Custom Splits**: `bingsong_project/dta_davis_complete.py` provides domain-specific train/test splitting:- **Fingerprints**: 2040-dim molecular fingerprints (RFScore+GBScore+ECIF) integrated via `collator.py`

- Max nodes: training/eval scripts use `--max-nodes 600`; `collator.py` asserts no sample exceeds this after filtering.

- Version pins in `install.sh` matter (Torch 1.10/CUDA 11.3; torch-scatter/torch-sparse versions); changing them can break builds.- Data will be downloaded/processed into `--data-path` (default similar to `./data`).



### Quick Commands  - `create_fold()`: Standard k-fold cross-validation- **Graph Encoding**: Node/edge features processed through `src/modules/graphormer_graph_encoder.py`



- Install env: see above.### Training

- Evaluate checkpoints: `./run_evaluate.sh`.

- Train (example): `bash Dynaformer/examples/md_pretrain/md_train.sh` (override env vars like `dataset_name`, `data_path`, `save_path`, `n_gpu`, etc.).- Reference script: `Dynaformer/examples/md_pretrain/md_train.sh` (a more general version of root `md_train.sh`). It launches Fairseq with distributed training:  - `create_fold_setting_cold()`: Cold-start drug/target scenarios  - **Key Parameters**: 

- Finetuning: `Dynaformer/examples/finetune/finetune.sh` starts from a pretrained checkpoint with `--finetune-from-model`

  ```bash  - `create_seq_identity_fold()`: Sequence identity-based splits  - `max_nodes=600`, `encoder_layers=4`, `encoder_attention_heads=32`

  python -m torch.distributed.launch --nproc_per_node=${n_gpu} \

    $(which fairseq-train) \- **Data Processing**: PyG `Data` objects with molecular graphs, 3D coordinates, and binding affinity targets  - `dist_head=gbf3d`, `num_dist_head_kernel=256`, `num_edge_types=16384`

    --user-dir "$(realpath ./dynaformer)" \

    --task graph_prediction_with_flag --criterion l2_loss_with_flag \- **Normalization**: Targets normalized as `(y - 6.529300030461668) / 1.9919705951218716` in loss functions

    --dataset-source pyg --dataset-name "<dataset_spec>" --data-path "<path>" \

    --arch graphormer_base --num-classes 1 \### Dataset Patterns & Integration

    --encoder-layers 4 --encoder-attention-heads 32 \

    --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \### Environment & Build- **Legacy Specs**: String-based dataset specs like `"pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5"`

    --lr 1e-4 --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \

    --warmup-updates <steps> --total-num-update <steps> --max-update <steps> \- Use provided installer from project root:- **Custom Splits**: `bingsong_project/dta_davis_complete.py` provides domain-specific train/test splitting:

    --batch-size 20 --update-freq 1 --weight-decay 1e-5 --clip-norm 5 \

    --max-nodes 600 --dist-head gbf3d --num-dist-head-kernel 256 --num-edge-types 16384 \  ```bash  - `create_fold()`: Standard k-fold cross-validation

    --fingerprint --fp16 --save-dir "<save_dir>"

  ```  cd Dynaformer  - `create_fold_setting_cold()`: Cold-start drug/target scenarios  

- Flags and conventions:

  - `--user-dir` must point to `Dynaformer/dynaformer` to load custom Fairseq components.  conda create -n dynaformer python=3.9 -y  - `create_seq_identity_fold()`: Sequence identity-based splits

  - `*_with_flag` task/criterion pairs enable FLAG adversarial training; set with `--flag-m/--flag-step-size/--flag-mag`.

  - 3D head: `--dist-head {none,gbf,gbf3d,bucket,embed3d}`; Dynaformer uses `gbf3d` with `--num-dist-head-kernel 256`, `--num-edge-types 16384`.  conda activate dynaformer- **Data Processing**: PyG `Data` objects with molecular graphs, 3D coordinates, and binding affinity targets

  - `--fingerprint` enables a 2040-dim feature branch (RFScore+GBScore+ECIF) fused in the model (`fpnn` in `graphormer.py`).

  - Scripts construct a descriptive `save_dir` and set wandb env vars (`WANDB_NAME`, `WANDB_SAVE_DIR`).  ./install.sh  # pins torch 1.10.0 + cu113 and builds fairseq + Cython- **Normalization**: Targets normalized as `(y - 6.529300030461668) / 1.9919705951218716` in loss functions



### Evaluation & Custom Input  ```

- Evaluate checkpoints: from project root

  ```bash- `install.sh` compiles Fairseq extensions and Cython module `dynaformer/data/algos.pyx`. CUDA 11.3-compatible environment expected.### Environment & Build

  ./run_evaluate.sh  # CASF-2016 and CASF-2013 using checkpoints in ./checkpoint

  ```- Use provided installer from project root:

- Script `Dynaformer/examples/evaluate/evaluate.sh` wraps `evaluate.py` with fixed model dims and `--fingerprint`.

- Important: `evaluate.py` rescales predictions before metrics: `y_pred = y_pred * 1.9919705951218716 + 6.529300030461668`.### Datasets (Spec Strings)  ```bash

- Custom input flow:

  1) Install additional dependencies: `conda install -c conda-forge pymol-open-source openbabel -y` under `dynaformer` environment- Datasets are specified as strings parsed by `PYGDatasetLookupTable.GetPYGDataset`:  cd Dynaformer

  2) Build `.pkl` with `preprocess/custom_input.py` from PDB/SDF and a CSV schema

  3) Run `./run_custom_input.sh` which calls `evaluate.sh` with `dataset-name "custom:path=...pkl"`  - PDBBind: `pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=0`  conda create -n dynaformer python=3.9 -y

- Custom input processes pockets on-the-fly using PyMOL/OpenBabel, so results may differ slightly from CASF evaluation (which uses pre-provided pocket files).

  - MD: `mddata:set_name=md-refined2019-5-5-5,seed=2022`  conda activate dynaformer

### Gotchas

- Keep `--user-dir` correct; otherwise Fairseq won't find custom tasks/models.  - Hybrid: `hybrid:set_name=md-refined2019-5-5-5+general-set-2019-coreset-2016,cutoffs=5-5-5,seed=2022`  ./install.sh  # pins torch 1.10.0 + cu113 and builds fairseq + Cython

- Max nodes: training/eval scripts use `--max-nodes 600`; `collator.py` asserts no sample exceeds this after filtering.

- Version pins in `install.sh` matter (Torch 1.10/CUDA 11.3; torch-scatter/torch-sparse versions); changing them can break builds.  - Custom (preprocessed pickle): `custom:path=/abs/path/to/data.pkl`  ```



### Quick Commands- Data will be downloaded/processed into `--data-path` (default similar to `./data`).- `install.sh` compiles Fairseq extensions and Cython module `dynaformer/data/algos.pyx`. CUDA 11.3-compatible environment expected.

- Install env: see above.

- Evaluate checkpoints: `./run_evaluate.sh`.

- Train (example): `bash Dynaformer/examples/md_pretrain/md_train.sh` (override env vars like `dataset_name`, `data_path`, `save_path`, `n_gpu`, etc.).
### Training### Datasets (Spec Strings)

- Reference script: `Dynaformer/examples/md_pretrain/md_train.sh` (a more general version of root `md_train.sh`). It launches Fairseq with distributed training:- Datasets are specified as strings parsed by `PYGDatasetLookupTable.GetPYGDataset`:

- Finetuning: `Dynaformer/examples/finetune/finetune.sh` starts from a pretrained checkpoint with `--finetune-from-model`  - PDBBind: `pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=0`

  ```bash  - MD: `mddata:set_name=md-refined2019-5-5-5,seed=2022`

  python -m torch.distributed.launch --nproc_per_node=${n_gpu} \  - Hybrid: `hybrid:set_name=md-refined2019-5-5-5+general-set-2019-coreset-2016,cutoffs=5-5-5,seed=2022`

    $(which fairseq-train) \  - Custom (preprocessed pickle): `custom:path=/abs/path/to/data.pkl`

    --user-dir "$(realpath ./dynaformer)" \- Data will be downloaded/processed into `--data-path` (default similar to `./data`).

    --task graph_prediction_with_flag --criterion l2_loss_with_flag \

    --dataset-source pyg --dataset-name "<dataset_spec>" --data-path "<path>" \### Training

    --arch graphormer_base --num-classes 1 \- Reference script: `Dynaformer/examples/md_pretrain/md_train.sh` (a more general version of root `md_train.sh`). It launches Fairseq with distributed training:

    --encoder-layers 4 --encoder-attention-heads 32 \- Finetuning: `Dynaformer/examples/finetune/finetune.sh` starts from a pretrained checkpoint with `--finetune-from-model`

    --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \  ```bash

    --lr 1e-4 --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \  python -m torch.distributed.launch --nproc_per_node=${n_gpu} \

    --warmup-updates <steps> --total-num-update <steps> --max-update <steps> \    $(which fairseq-train) \

    --batch-size 20 --update-freq 1 --weight-decay 1e-5 --clip-norm 5 \    --user-dir "$(realpath ./dynaformer)" \

    --max-nodes 600 --dist-head gbf3d --num-dist-head-kernel 256 --num-edge-types 16384 \    --task graph_prediction_with_flag --criterion l2_loss_with_flag \

    --fingerprint --fp16 --save-dir "<save_dir>"    --dataset-source pyg --dataset-name "<dataset_spec>" --data-path "<path>" \

  ```    --arch graphormer_base --num-classes 1 \

- Flags and conventions:    --encoder-layers 4 --encoder-attention-heads 32 \

  - `--user-dir` must point to `Dynaformer/dynaformer` to load custom Fairseq components.    --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \

  - `*_with_flag` task/criterion pairs enable FLAG adversarial training; set with `--flag-m/--flag-step-size/--flag-mag`.    --lr 1e-4 --end-learning-rate 1e-9 --lr-scheduler polynomial_decay --power 1 \

  - 3D head: `--dist-head {none,gbf,gbf3d,bucket,embed3d}`; Dynaformer uses `gbf3d` with `--num-dist-head-kernel 256`, `--num-edge-types 16384`.    --warmup-updates <steps> --total-num-update <steps> --max-update <steps> \

  - `--fingerprint` enables a 2040-dim feature branch (RFScore+GBScore+ECIF) fused in the model (`fpnn` in `graphormer.py`).    --batch-size 20 --update-freq 1 --weight-decay 1e-5 --clip-norm 5 \

  - Scripts construct a descriptive `save_dir` and set wandb env vars (`WANDB_NAME`, `WANDB_SAVE_DIR`).    --max-nodes 600 --dist-head gbf3d --num-dist-head-kernel 256 --num-edge-types 16384 \

    --fingerprint --fp16 --save-dir "<save_dir>"

### Evaluation & Custom Input  ```

- Evaluate checkpoints: from project root- Flags and conventions:

  ```bash  - `--user-dir` must point to `Dynaformer/dynaformer` to load custom Fairseq components.

  ./run_evaluate.sh  # CASF-2016 and CASF-2013 using checkpoints in ./checkpoint  - `*_with_flag` task/criterion pairs enable FLAG adversarial training; set with `--flag-m/--flag-step-size/--flag-mag`.

  ```  - 3D head: `--dist-head {none,gbf,gbf3d,bucket,embed3d}`; Dynaformer uses `gbf3d` with `--num-dist-head-kernel 256`, `--num-edge-types 16384`.

- Script `Dynaformer/examples/evaluate/evaluate.sh` wraps `evaluate.py` with fixed model dims and `--fingerprint`.  - `--fingerprint` enables a 2040-dim feature branch (RFScore+GBScore+ECIF) fused in the model (`fpnn` in `graphormer.py`).

- Important: `evaluate.py` rescales predictions before metrics: `y_pred = y_pred * 1.9919705951218716 + 6.529300030461668`.  - Scripts construct a descriptive `save_dir` and set wandb env vars (`WANDB_NAME`, `WANDB_SAVE_DIR`).

- Custom input flow:

  1) Install additional dependencies: `conda install -c conda-forge pymol-open-source openbabel -y` under `dynaformer` environment### Evaluation & Custom Input

  2) Build `.pkl` with `preprocess/custom_input.py` from PDB/SDF and a CSV schema- Evaluate checkpoints: from project root

  3) Run `./run_custom_input.sh` which calls `evaluate.sh` with `dataset-name "custom:path=...pkl"`  ```bash

- Custom input processes pockets on-the-fly using PyMOL/OpenBabel, so results may differ slightly from CASF evaluation (which uses pre-provided pocket files).  ./run_evaluate.sh  # CASF-2016 and CASF-2013 using checkpoints in ./checkpoint

  ```

### Gotchas- Script `Dynaformer/examples/evaluate/evaluate.sh` wraps `evaluate.py` with fixed model dims and `--fingerprint`.

- Keep `--user-dir` correct; otherwise Fairseq won't find custom tasks/models.- Important: `evaluate.py` rescales predictions before metrics: `y_pred = y_pred * 1.9919705951218716 + 6.529300030461668`.

- Max nodes: training/eval scripts use `--max-nodes 600`; `collator.py` asserts no sample exceeds this after filtering.- Custom input flow:

- Version pins in `install.sh` matter (Torch 1.10/CUDA 11.3; torch-scatter/torch-sparse versions); changing them can break builds.  1) Install additional dependencies: `conda install -c conda-forge pymol-open-source openbabel -y` under `dynaformer` environment

  2) Build `.pkl` with `preprocess/custom_input.py` from PDB/SDF and a CSV schema

### Quick Commands  3) Run `./run_custom_input.sh` which calls `evaluate.sh` with `dataset-name "custom:path=...pkl"`

- Install env: see above.- Custom input processes pockets on-the-fly using PyMOL/OpenBabel, so results may differ slightly from CASF evaluation (which uses pre-provided pocket files).

- Evaluate checkpoints: `./run_evaluate.sh`.

- Train (example): `bash Dynaformer/examples/md_pretrain/md_train.sh` (override env vars like `dataset_name`, `data_path`, `save_path`, `n_gpu`, etc.).### Gotchas
- Keep `--user-dir` correct; otherwise Fairseq won’t find custom tasks/models.
- Max nodes: training/eval scripts use `--max-nodes 600`; `collator.py` asserts no sample exceeds this after filtering.
- Version pins in `install.sh` matter (Torch 1.10/CUDA 11.3; torch-scatter/torch-sparse versions); changing them can break builds.

### Quick Commands
- Install env: see above.
- Evaluate checkpoints: `./run_evaluate.sh`.
- Train (example): `bash Dynaformer/examples/md_pretrain/md_train.sh` (override env vars like `dataset_name`, `data_path`, `save_path`, `n_gpu`, etc.).
