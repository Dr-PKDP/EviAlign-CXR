# 4. Experimental design

## 4.1 Data

All corpora can be downloaded without credentialing or a data-access committee (Table 2). ReXGradient-160K is the training corpus: 160,000 studies from 109,487 patients at 79 sites of three US health systems, distributed as 16-bit PNG images at a quarter of their original resolution with Findings and Impression sections [@zhangx2025rexgradient]. Access is an automatically approved click-through licence for non-commercial research with attribution to Gradient Health; the data are not redistributed here.

Studies were restricted to adults (18 years or older) with one frontal image each. The release carries view labels for part of the images, so the PA image was preferred, then AP. In studies where the only labelled images were lateral, an unlabelled image was taken as frontal. A view probe, a balanced logistic regression on BiomedCLIP embeddings [@zhang2025biomedclip] trained on labelled views (held-out accuracy 0.90, AUROC 0.97), scored every selected image, and images with a frontal probability below 0.5 were dropped. Of 123,421 candidate studies, 92,837 of 100,859 with a labelled PA or AP image and 20,314 of 22,562 with an inferred frontal image passed, leaving 113,151 studies. The released splits were kept and no patient crosses them: 98,770 studies for training, 7,166 for validation (threshold calibration and checkpoint selection) and 7,215 as the internal test set. Reports were labelled with CheXbert [@smit2020] and, for emphysema, fibrosis, pleural thickening and hernia, with the rule labeler of Section 3.4. Lung and heart masks were available for all images (Supplementary Note 1 gives the probe and selection details).

| Corpus | Country (institutions) | Images used | Labels | Reports | Localisation | Role |
|---|---|---|---|---|---|---|
| ReXGradient-160K [@zhangx2025rexgradient] | USA (3 systems, 79 sites) | 113,151 frontal | CheXbert + rules | English | – | train / val / internal test |
| NIH ChestX-ray14 [@wang2017] | USA (NIH CC) | 25,596 (official test list) | 14, NLP-mined | – | boxes, 8 findings | external |
| ChestX-Det [@liu2020chestxdet] | USA (NIH subset) | 3,543 (2,633 with boxes) | 10 categories, radiologist boxes | – | boxes, 7 mapped findings | external |
| RSNA Pneumonia [@shih2019] | USA (NIH subset) | 29,684 | lung opacity, adjudicated | – | opacity boxes (7,106 images) | external |
| IU X-ray [@demner2016] | USA (Indiana) | [N-IU] frontal, 3,851 reports | CheXbert on reports | English | – | external |
| Shenzhen [@jaeger2014] | China | 662 | TB vs normal | – | – | external |
| Montgomery [@jaeger2014] | USA (Maryland) | 138 | TB vs normal | – | – | external |
| TBX11K, validation split [@liu2024tbx11k] | China | [N-TBX] | healthy / sick / TB | – | TB boxes | external |

Table: **Table 2.** Corpora. Bracketed counts are fixed after view filtering and removal of unreadable files.

Because training uses ReXGradient, every NIH-derived annotation set is external: ChestX-Det and RSNA Pneumonia relabel NIH images with radiologist boxes, and none of their images or patients occurs in training. ChestX-Det images are drawn from ChestX-ray14, so the NIH and ChestX-Det results overlap in images and are not independent samples. The backbone follows from the same constraint. DINOv2 [@oquab2024] was pretrained on natural images, whereas RAD-DINO and some medical VLMs saw NIH or other test corpora and would leak image-level information [@perez2025].

Label spaces differ between corpora, and Table 3 maps them to thirteen canonical findings. Two mappings are approximate and are reported separately: NIH "Infiltration" is mapped to lung opacity, and NIH "Mass" and "Nodule" are merged into nodule or mass (the CheXpert "Lung Lesion" class). A finding that a corpus does not annotate is masked from its evaluation, not treated as negative. The tuberculosis corpora have no matching finding; there the abnormality score $1-\prod_c(1-\hat y_c)$ over thoracic findings is evaluated against the abnormal label and the probability-weighted evidence $\sum_c\hat y_c[\phi_c]_+$ against TB boxes. This is a transfer test to an unseen disease and not a TB classifier.

| Canonical finding | ReXGradient (CheXbert / rule) | NIH-14 | ChestX-Det | RSNA | IU (CheXbert) |
|---|---|---|---|---|---|
| atelectasis | Atelectasis | Atelectasis | Atelectasis | – | Atelectasis |
| cardiomegaly | Cardiomegaly | Cardiomegaly | – | – | Cardiomegaly |
| consolidation | Consolidation | Consolidation | Consolidation | – | Consolidation |
| edema | Edema | Edema | – | – | Edema |
| pleural effusion | Pleural Effusion | Effusion | Effusion | – | Pleural Effusion |
| pneumonia | Pneumonia | Pneumonia | – | Lung Opacity | Pneumonia |
| pneumothorax | Pneumothorax | Pneumothorax | Pneumothorax | – | Pneumothorax |
| nodule or mass | Lung Lesion | Mass, Nodule | Mass, Nodule | – | Lung Lesion |
| lung opacity | Lung Opacity | Infiltration (approx.) | – | Lung Opacity | Lung Opacity |
| emphysema | rule | Emphysema | Emphysema | – | rule |
| fibrosis | rule | Fibrosis | Fibrosis | – | rule |
| pleural thickening | rule | Pleural_Thickening | – | – | rule |
| hernia | rule | Hernia | – | – | rule |

Table: **Table 3.** Harmonised label space. "rule" denotes the transparent labeler of Section 3.4. RSNA positives are adjudicated lung-opacity cases and are scored for both pneumonia and lung opacity.

## 4.2 Implementation and training protocols

Images were letterboxed to $448\times448$ and normalised with ImageNet statistics. The encoder was a DINOv2 ViT-B/14 [@oquab2024] with gradient checkpointing, giving $32\times32$ patch tokens. The evidence head had four heads of dimension 256. The decoder was Qwen2.5-1.5B-Instruct [@qwen2025] in bfloat16 with eager attention, LoRA rank 16 on query, key, value and output projections [@hu2022lora], and a $16\times16$-token visual prefix. Optimisation used AdamW with weight decay 0.05, learning rates of $10^{-5}$ (encoder), $10^{-3}$ (head) and $2\times10^{-4}$ (projector and LoRA), 1,000 warm-up steps with cosine decay, batch size 32 and gradient clipping at 1.0. Augmentation was rotation within ±7°, scaling within ±10%, translation within ±5% and small intensity changes, applied identically to images and anatomy masks. Decision thresholds were set per finding to maximise F1 on the ReXGradient validation split and transferred unchanged to every external corpus. Explanations were decoded greedily with at most 96 new tokens. Models were trained in PyTorch with timm, Transformers and PEFT on one NVIDIA H200 GPU, using about 56 GB at batch size 32 once the language model is active.

Two protocols were used because of cost. The full protocol (all 98,770 training studies, twelve epochs, about eight hours) trains the main model, the unguided VLM and the classifier-only model. The reduced protocol (a fixed 30,000-study subset, a 3,000-study validation subset, six epochs, with the staged schedule compressed to one classifier-only epoch and alignment from the third epoch) trains every ablation and the seed replicates. Each ablation is compared with the reduced-protocol main model and never with the full-protocol model, and the full model, the model without alignment and the unguided model are replicated with three seeds under the reduced protocol. Code, configurations and the synthetic benchmark are released with the paper.

## 4.3 Compared systems

Each comparison differs from the full model in one respect, within the same encoder, data and schedule. The classifier-only model has the same encoder and evidence head and no decoder. Its evidence is compared with Grad-CAM on patch tokens [@selvaraju2017], class-query attention and Integrated Gradients with a blurred baseline [@sundararajan2017], and its predictions are verbalised by templates ("There is a pleural effusion."). The template system is faithful to the classifier by construction and uninformative beyond the labels, which bounds what prediction–explanation consistency can show. The unguided VLM trains classifier and decoder jointly with $\mathcal{L}_{\mathrm{cls}}+\mathcal{L}_{\mathrm{LM}}$ only and is the reference for every faithfulness comparison. The aggregated-alignment model replaces finding-specific alignment by one class-agnostic evidence map (the label-weighted mean of $\tilde a_c$) aligned with the mean grounding of all target tokens, which reproduces the mechanism of CAMANet [@wang2024camanet] inside this architecture. An off-the-shelf open VLM generating findings zero-shot is scored with the same text metrics as a reference for explanation content without classifier coupling.

Component ablations remove, one at a time, the alignment and instance terms, the instance term alone, the anatomical term and the sufficiency–comprehensiveness term; replace the stop-gradient by bidirectional alignment; replace JS by $\mathrm{KL}(\tilde b\Vert\tilde a)$; let the decoder losses update the shared encoder; replace the LLM decoder by the GPT-2 cross-attention decoder; and replace DINOv2 by the BiomedCLIP vision tower [@zhang2025biomedclip].

## 4.4 Evaluation

Classification is reported as per-finding and macro AUROC and AUPRC with 95% confidence intervals from 1,000 patient-level bootstrap resamples, F1, sensitivity and specificity at the transferred thresholds, and expected calibration error over 15 bins.

Prediction–explanation consistency compares the findings the classifier predicts at its F1-optimal thresholds with those the explanation asserts (rule labeler applied to generated text), for free generation and for finding-conditioned generation (Section 3.9). Because the second mode takes its findings from the classifier, it shows coverage and the cost of the 90%-specificity gate, while free generation shows whether the decoder volunteers the classifier's findings unaided. It is summarised by macro F1 between the two binary vectors, the unsupported rate (share of asserted findings the classifier does not predict) and the omission rate (share of predicted findings not asserted). The hallucination rate is the share of asserted findings that are negative in the reference labels, and text clinical F1 compares asserted findings with the reference labels. Where reference reports exist (ReXGradient, IU X-ray), BLEU-4 and ROUGE-L against the evidence sentences are secondary measures, since n-gram overlap tracks radiologist judgement poorly [@yu2023].

Counterfactual faithfulness uses two interventions on the $16\times16$ visual-token grid, both restricted to pairs $(i,c)$ that the classifier predicts and the explanation asserts. Neither is a training loss, but the sufficiency–comprehensiveness term (Section 3.8) teaches the classifier to lose score when blurred evidence is removed, which is the operation they perform, so a model trained with it can score well on blur-based tests without its explanations being more faithful. Text-guided deletion (TGD) removes the 15% of tokens with the largest text grounding $\tilde b_{i,c}$ and records the drop in the classifier logit, $\Delta^{\mathrm{text}}=s_c(I)-s_c(I^{\setminus\tilde b})$. The same drop is recorded for removal by the classifier's own evidence, $\Delta^{\mathrm{evid}}$, and for three random removals of equal area, $\Delta^{\mathrm{rand}}$. The normalised score

$$\mathrm{TGD}=\frac{\mathbb{E}[\Delta^{\mathrm{text}}]-\mathbb{E}[\Delta^{\mathrm{rand}}]}{\mathbb{E}[\Delta^{\mathrm{evid}}]-\mathbb{E}[\Delta^{\mathrm{rand}}]}$$

is 0 when the explanation points no better than chance and 1 when it points as well as the classifier's own evidence. The ratio is unstable when evidence removal barely exceeds random removal, so the primary statistic is the excess drop $\mathbb{E}[\Delta^{\mathrm{text}}-\Delta^{\mathrm{rand}}]$ with a bootstrap CI over pairs, reported with the win rate $P(\Delta^{\mathrm{text}}>\Delta^{\mathrm{rand}})$. Explanation counterfactual sensitivity (ECS) asks the converse of the decoder: after removing the top 15% of $c$'s evidence, how far does the per-token log-likelihood of the generated sentences asserting $c$ fall, relative to a random removal of equal area,

$$\mathrm{ECS}=\mathbb{E}\big[\ell(E_c\mid I)-\ell(E_c\mid I^{\setminus\tilde a})\big]-\mathbb{E}\big[\ell(E_c\mid I)-\ell(E_c\mid I^{\setminus\mathrm{rand}})\big].$$

A positive ECS means the sentence depends on the evidence region more than on an arbitrary region of the same size. TGD tests whether the text points at what the classifier uses; ECS tests whether the text uses it. The Jensen–Shannon distance between $\tilde a_c$ and $\tilde b_c$ is also reported, labelled as a consistency statistic because it is optimised.

The interventions are therefore run with removal operators the regulariser never saw: Gaussian noise with the image's mean and standard deviation (the primary operator, since it does not smooth the image), flat image-mean infill, and blur at 5% and 30% of tokens in addition to the 15% used in training. Absolute values are interpreted through matched comparisons within the reduced protocol, in which the full objective is compared with the model without alignment and instance terms, the model without the sufficiency–comprehensiveness term, and the unguided model, because a difference that survives an unseen operator and a matched ablation is attributable to the component removed. These passes use a fixed subset (the first 3,200 images of NIH, ChestX-Det and RSNA), so every model sees the same pairs.

Localisation is scored on radiologist boxes for every map type (evidence, Grad-CAM, class-query attention, Integrated Gradients on a subsample, and the text grounding of the generated explanation) by the pointing game with a 15-pixel tolerance [@zhang2018eb], the share of map mass inside the boxes, and that share divided by the box-area fraction (1 is chance). Deletion and insertion areas for the top predicted finding [@petsiuk2018] use blurred infill. A model-randomisation check [@adebayo2018] compares evidence maps before and after re-initialising the head and the last four encoder blocks.

## 4.5 Reader study and statistics

Two board-certified radiologists, blinded to the system, rate 200 external images (100 NIH, 100 IU X-ray, stratified by finding), each shown with the explanation and grounding overlay from the full model and the unguided VLM in random order. For each asserted finding they rate the statement (present, absent, uncertain) and whether the overlay covers the finding (yes, partly, no); for each image they rate usefulness on a five-point scale and list omitted findings. Agreement is Cohen's κ, with paired Wilcoxon tests on usefulness and McNemar tests on per-finding correctness.

Differences between the full model and each comparator are tested with a paired patient-level bootstrap (1,000 resamples) for AUROC and paired permutation tests over (image, finding) pairs for TGD and ECS. Seed results are mean and standard deviation. Significance is assessed at $\alpha=0.05$ after Holm correction within each table.
