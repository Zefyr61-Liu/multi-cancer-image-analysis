# Data sources and evaluation

## Upstream sources

- The historical 26-label experiment used [Multi Cancer on Kaggle](https://www.kaggle.com/datasets/obulisainaren/multi-cancer/).
- The 11-label experiment used original images or traceable derivatives from the [blood image mirror](https://huggingface.co/datasets/pritam06/leukemia-cancer-data/tree/9696248a8da664c8edae05a1ca95fb654466b2af), [BiomedCoOp kidney data mirror](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/tree/c50b6304ea49623188dc7f0f470c511b48538108), [lymphoma derivative archive](https://github.com/atlan-antillia/EfficientNet-Malignant-Lymphoma/tree/172aa41c62b84987c16578295fef3d92274e4896), and [oral images, version 2](https://data.mendeley.com/datasets/ftmp4cvtmb/2).
- The external lung experiment used public [TCGA data through the NCI Genomic Data Commons](https://portal.gdc.cancer.gov/). Follow each upstream dataset's terms when obtaining or reusing images.

## 11-label evaluation

The fixed evaluation involved 880 images across the blood, kidney, lymphoma and oral categories. Train/validation/test counts were 628/127/125. Related images were grouped using documented original-image provenance or a conservative filename grouping for the lymphoma subset. Both models started from random initialization at 96×96 pixels and used the same split with model selection by minimum validation loss. CancerCNN selected epoch 8 and ResNet-18 selected epoch 4. The original grouping procedure does not establish patient-level independence.

The aggregate records in `results/` let readers inspect the metrics, class support, confusion matrices and learning curves. The public repository contains the runnable pipeline and aggregate evidence. The smoke test uses generated images. Reproducing the historical scores requires the original fixed split, per-image data and saved weights.

## External lung evaluation

The external TCGA cohort included lung adenocarcinoma and lung squamous cell carcinoma. Ten of twelve preselected cases had eligible tiles, yielding 80 tiles. Case predictions were computed by averaging probabilities over each case's tiles. The baseline classified 3.75% of tiles and 0 of 10 cases correctly. The adapted model classified 57.50% of tiles and 5 of 10 cases correctly. Tissue tiles inherited case labels. This small cohort describes performance in the observed evaluation.
