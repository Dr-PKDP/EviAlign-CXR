---
title: "Supplementary Material: Evidence-Aligned Vision–Language Explanations for Chest Radiograph Diagnosis"
---

# Supplementary Note 1. Data preparation

**ReXGradient-160K.** The release lists images in zstd-compressed tar parts and carries view labels in a JSON field for part of the images. Patients under 18 years were excluded by the age field (years, months, days or weeks). For each study the frontal image was chosen as PA if present, otherwise AP; where the labelled images were lateral only and an unlabelled image was present, that image was taken as frontal. A pairwise check showed that the image-path and view-position fields are aligned and that studies with an unlabelled image beside a lateral image behave like labelled frontal–lateral pairs, which supports the inference. A zero-shot BiomedCLIP prompt did not separate frontal from lateral views, so a trained probe was used: balanced logistic regression (C = 10) on BiomedCLIP image embeddings, trained on labelled frontal and lateral images and evaluated on held-out images (accuracy 0.904, AUROC 0.968). Images with p(frontal) below 0.5 were dropped. Of 100,859 studies with a labelled PA or AP image, 92,837 passed; of 22,562 with an inferred frontal image, 20,314 passed; 113,151 studies remained (train 98,770, validation 7,166, test 7,215), with no patient in more than one split. Lung and heart masks from the TorchXRayVision PSPNet were produced for every retained image.

**IU X-ray.** The probe transferred poorly to the IU images (7 of 7,470 kept), so for IU it was adapted: pseudo-labels were taken from within-report ranking of the probe score, a new classifier was fitted on cross-fitted folds split by report, and the best-scoring image per report was kept if its adapted probability was at least 0.5. A random audit of 120 selected images found no lateral image (0 of 120; 95% Clopper–Pearson upper bound 3.0%), which bounds contamination of the IU results.

**RSNA Pneumonia.** The adjudicated MD.ai export was used: an image is positive if it carries at least one box of the final "Lung Opacity" label, giving 29,684 images, 7,106 of them positive with boxes.  DICOM files were converted to PNG.

**Licences.** ReXGradient-160K: non-commercial research use, no redistribution, attribution to Gradient Health. RSNA Pneumonia: attribution to the RSNA. TBX11K: CC BY 4.0 (training and validation images are released; the test set is not).

# Supplementary Note 2. Worked numerical example

Take four patches, one finding, contributions $\phi=[2.0, 1.0, 0.5, -0.4]$ and bias $0.3$, so $s=3.4$. The evidence distribution is $\tilde a\approx[0.571, 0.286, 0.143, 0.000]$; the fourth patch carries counter-evidence and receives only the floor. A sentence whose grounding is $\tilde b=[0.70, 0.20, 0.08, 0.02]$ gives $\mathrm{JS}=0.018$, and a sentence grounded on the wrong side, $\tilde b'=[0.10, 0.15, 0.25, 0.50]$, gives $\mathrm{JS}=0.282$, against an upper bound of $\ln 2=0.693$. The directional divergence separates the two far more sharply, $\mathrm{KL}(\tilde b\Vert\tilde a)=0.112$ against $\mathrm{KL}(\tilde b'\Vert\tilde a)=3.67$, because $\tilde b'$ puts half its mass on a patch with almost no evidence; this sensitivity is why KL is used as an ablation and not as the default. For the instance term, a second image with the same finding and evidence $[0.05, 0.15, 0.30, 0.50]$ has centred cosine similarity $-0.83$ with $\tilde b$, against $0.97$ for the correct image, and the InfoNCE loss at $\tau=0.1$ is below $10^{-7}$; it becomes large only when the grounding fits other images as well as its own. If the fourth patch lies outside the lungs, $\mathcal{L}_{\mathrm{anat}}=0.0002$. Two alternatives fail on the same numbers. A Dice overlap computed between two normalised distributions does not vanish when they are identical: for $g=[0.65, 0.20, 0.15]$, $1-2\sum g^2/(\sum g+\sum g)=0.515$, so mass inside the anatomical region is the quantity used instead. And applying a softmax to $\phi_+=[2, 1, 0.5]$ gives $[0.63, 0.23, 0.14]$, whereas the same pattern multiplied by ten gives $[1.00, 0.00, 0.00]$.

# Supplementary tables

| Finding | CheXbert positives | Rule positives | Agreement | Cohen's κ |
|---|---|---|---|---|
| atelectasis | 3,437 | 3,380 | 0.995 | 0.982 |
| cardiomegaly | 4,792 | 3,778 | 0.943 | 0.833 |
| consolidation | 587 | 513 | 0.993 | 0.865 |
| edema | 1,621 | 1,538 | 0.987 | 0.912 |
| pleural effusion | 2,331 | 2,449 | 0.989 | 0.950 |
| pneumonia | 870 | 762 | 0.992 | 0.902 |
| pneumothorax | 316 | 317 | 0.998 | 0.931 |
| nodule or mass | 738 | 590 | 0.989 | 0.826 |
| lung opacity | 4,178 | 4,273 | 0.899 | 0.698 |

Table: **Table S1.** Agreement between the rule labeler and CheXbert on 20,000 randomly drawn ReXGradient reports (`tools/labeler_agreement.py`).

| Finding | ReXGradient test | NIH-14 | ChestX-Det | RSNA | IU X-ray |
|---|---|---|---|---|---|
| atelectasis | | | | – | |
| cardiomegaly | | | – | – | |
| consolidation | | | | – | |
| edema | | | – | – | |
| pleural effusion | | | | – | |
| pneumonia | | | – | | |
| pneumothorax | | | | – | |
| nodule or mass | | | | – | |
| lung opacity | | | – | | |
| emphysema | | | | – | |
| fibrosis | | | | – | |
| pleural thickening | | | – | – | |
| hernia | | | – | – | |

Table: **Table S2.** Per-finding AUROC of EviAlign-CXR (full protocol) with 95% patient-level bootstrap CI.

| Metric | Set | Full | − alignment | − anatomy | Unguided | Full, detached decoder input |
|---|---|---|---|---|---|---|
| JS(text, evidence) | internal | 0.026 ± 0.004 | 0.366 ± 0.049 | 0.022 ± 0.001 | 0.332 ± 0.093 | 0.035 ± 0.004 |
| JS(text, evidence) | external | 0.029 ± 0.005 | 0.342 ± 0.034 | 0.030 ± 0.005 | 0.292 ± 0.064 | 0.047 ± 0.018 |
| Win rate vs random | internal | 0.710 ± 0.098 | 0.540 ± 0.124 | 0.688 ± 0.157 | 0.535 ± 0.127 | 0.683 ± 0.161 |
| Win rate vs random | external | 0.442 ± 0.189 | 0.347 ± 0.014 | 0.422 ± 0.072 | 0.291 ± 0.034 | 0.339 ± 0.021 |
| Effusion AUROC | internal | 0.999 ± 0.001 | 0.999 ± 0.001 | 0.998 ± 0.002 | 0.997 ± 0.003 | 0.999 ± 0.001 |
| Effusion AUROC | external | 0.992 ± 0.006 | 0.989 ± 0.009 | 0.986 ± 0.012 | 0.973 ± 0.008 | 0.989 ± 0.012 |
| System | Set | Pairs | Excess drop, text (95% CI) | Win rate vs random | Excess drop, evidence | ECS gap (95% CI) | JS (text, evidence)† |

Table: **Table S3.** Controlled synthetic study, additional metrics (JS: Jensen–Shannon distance between text grounding and evidence, optimised during training; win rate: share of pairs in which removing the named tokens lowers the logit more than random removal; effusion AUROC on the cue-shifted set). Mean ± s.d. over three seeds.

| Re-initialised | Median Spearman ρ (evidence maps) | Interquartile range |
|---|---|---|
| evidence head | [R] | [R] |
| head and last four encoder blocks | [R] | [R] |

Table: **Table S4.** Model-randomisation check on NIH test images.

| Removal operator | System | Excess drop, text | Win rate | Normalised TGD | ECS gap |
|---|---|---|---|---|---|
| Blur, 5% of tokens | Unguided VLM | 0.49 | 0.66 | 0.49 | 0.064 |
| Blur, 5% of tokens | EviAlign-CXR | 2.78 | 0.98 | 0.96 | 0.233 |
| Blur, 30% | Unguided VLM | 1.60 | 0.89 | 0.83 | 0.119 |
| Blur, 30% | EviAlign-CXR | 6.29 | 1.00 | 1.02 | 0.356 |
| Image-mean infill, 15% | Unguided VLM | 1.00 | 0.77 | 0.80 | 0.058 |
| Image-mean infill, 15% | EviAlign-CXR | 5.32 | 1.00 | 1.01 | 0.408 |
| Noise infill, 15% | Unguided VLM | 1.29 | 0.84 | 0.79 | 0.165 |
| Noise infill, 15% | EviAlign-CXR | 1.05 | 0.77 | 1.09 | 0.063 |

Table: **Table S5.** Text-guided deletion and counterfactual sensitivity on NIH ChestX-ray14 (fixed subset, finding-conditioned explanations, full protocol) under removal operators other than the 15% blur used in training. In the full NIH set with 15% blur, EviAlign-CXR reached an excess drop of 5.76 (win rate 1.00, normalised 1.01, ECS gap 0.394).

| Set | System | Deletion area, evidence | Insertion area, evidence | Deletion area, Grad-CAM | Insertion area, Grad-CAM |
|---|---|---|---|---|---|
| ReXGradient test | Classifier only | 0.430 | 0.681 | 0.470 | 0.669 |
| ReXGradient test | EviAlign-CXR | 0.070 | 0.737 | 0.090 | 0.743 |
| NIH-14 | Classifier only | 0.446 | 0.807 | 0.523 | 0.780 |
| NIH-14 | EviAlign-CXR | 0.088 | 0.863 | 0.145 | 0.859 |
| ChestX-Det | Classifier only | 0.408 | 0.801 | 0.494 | 0.766 |
| ChestX-Det | EviAlign-CXR | 0.081 | 0.876 | 0.150 | 0.871 |
| RSNA | Classifier only | 0.423 | 0.761 | 0.485 | 0.738 |
| RSNA | EviAlign-CXR | 0.073 | 0.825 | 0.126 | 0.821 |
| IU X-ray | Classifier only | 0.375 | 0.628 | 0.392 | 0.621 |
| IU X-ray | EviAlign-CXR | 0.057 | 0.699 | 0.066 | 0.706 |

Table: **Table S6.** Deletion and insertion areas for the top predicted finding with blurred infill (lower deletion and higher insertion are better). The classifier-only model did not train with the sufficiency–comprehensiveness term.
