"""Baseline performance data for supported models.

Measured original model performance (mean values).

Performance metrics by domain:
- DRP (Drug Response Prediction): rmse, mae, pcc, scc
- Spatial: ari, nmi, silhouette
- DTI (Drug-Target Interaction): accuracy, auroc, auprc, f1
"""

MODEL_BASELINES = {
    # === Spatial ===
    "deepst": {
        "description": "DeepST: Spatial Transcriptomics with Deep Learning",
        "domain": "spatial",
        "performance": {
            "ari": 0.526,
        },
    },
    "stagate": {
        "description": "STAGATE: Spatial Transcriptomics Analysis with Graph Attention auto-Encoder",
        "domain": "spatial",
        "performance": {
            "ari": 0.4957,
        },
    },
    # === DTI ===
    "hyperattentiondti": {
        "description": "HyperAttentionDTI: Drug-Target Interaction with Hypergraph",
        "domain": "dti",
        "performance": {
            "auprc": 0.7706,
        },
    },
    "dlm-dti": {
        "description": "DLM-DTI: Dual Language Model for Drug-Target Interaction",
        "domain": "dti",
        "performance": {
            "auprc": 0.8238,
        },
    },
    # === DRP ===
    "deeptta": {
        "description": "DeepTTA: Drug response prediction using Transfer Learning",
        "domain": "drp",
        "performance": {
            "rmse": 1.2946,
        },
    },
    "deepdr": {
        "description": "DeepDR: Drug Response Prediction using Graph + MPG (Message Passing Graph)",
        "domain": "drp",
        "performance": {
            "rmse": 1.2006,  # Baseline - update after actual test
        },
    },
}
