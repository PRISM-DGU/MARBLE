# HyperAttentionDTI

Drug-Target Interaction prediction using Cross-Attention mechanism.

Based on: "HyperAttentionDTI: improving drug–protein interaction prediction by sequence-based deep learning with attention mechanism"

## Directory Structure

```
hyperattentiondti/
├── config.yaml              # Configuration file
├── Dockerfile               # Docker image definition
├── requirements.txt         # Python dependencies
├── components/              # Encoder/Decoder components (registry pattern)
│   ├── __init__.py
│   ├── registry.py          # Component registry
│   ├── base.py              # Base classes
│   ├── drug_encoder_hyperattentiondti.py    # SMILES CNN encoder
│   ├── protein_encoder_hyperattentiondti.py # Protein CNN encoder
│   └── decoder_hyperattentiondti.py         # Cross-attention + MLP decoder
└── src/                     # Model source code
    ├── main.py              # Training entry point
    ├── model.py             # Unified model
    └── utils.py             # Utilities
```

## Model Architecture

- **Drug Encoder**: Character-level SMILES → Embedding → 3-layer CNN (kernel: [4,6,8])
- **Protein Encoder**: Character-level protein sequence → Embedding → 3-layer CNN (kernel: [4,8,12])
- **Decoder**: Cross-Attention mechanism + MLP classifier (1024→1024→512→2)

## Datasets

Supported datasets (located at `/workspace/datasets/hyperattentiondti/`):
- KIBA
- Davis
- DrugBank

Data format:
```
Drug_ID Protein_ID SMILES Protein_sequence Label
```

## Docker Usage

### Build
```bash
docker build -t autodrp/hyperattentiondti:latest .
```

### Run
```bash
docker run --gpus all -v /path/to/datasets:/workspace/datasets \
    autodrp/hyperattentiondti:latest \
    python /app/hyperattentiondti_models/main.py --config /app/config.yaml
```

## Configuration

Key parameters in `config.yaml`:
- `data.dataset`: Dataset name (KIBA, Davis, DrugBank)
- `model.drug_encoder.architecture`: Drug CNN settings
- `model.protein_encoder.architecture`: Protein CNN settings
- `training.epochs`: Number of epochs (default: 200)
- `training.batch_size`: Batch size (default: 32)
- `training.learning_rate`: Learning rate (default: 5e-5)

## Command Line Arguments

```bash
python main.py --config ../config.yaml --epochs 100 --batch_size 64 --dataset KIBA
```

## Dependencies

- Python 3.10
- PyTorch 2.4.1 + CUDA 12.1
- NumPy, SciPy, scikit-learn
- tqdm, PyYAML, TensorBoardX
