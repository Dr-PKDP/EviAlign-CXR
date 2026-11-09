# 5. Results

## 5.1 Controlled study with known evidence

The clinical corpora do not reveal which pixels a finding occupies, so the first experiment uses synthetic radiograph-like images in which they are known. Each $64\times64$ image holds two lung fields and a heart silhouette. Five findings (pleural effusion, consolidation, cardiomegaly, nodule, pneumothorax) are placed independently with probability 0.3 at a random side and zone, and the report states each with its side and zone. Boxes and anatomy masks come from the generator. A spurious cue, a bright $6\times6$ square in a corner outside the lungs, appears in 90% of training images with effusion and 10% of those without; in the external set the cue is independent of the label and intensity is scaled by 1.25. There are 1,200 training images (70/15/15 split by patient) and 400 external images. The model is a four-block ViT with $8\times8$ patches trained from scratch with a two-layer cross-attention decoder on a $4\times4$ visual grid for eight epochs and three seeds. Four variants let the decoder losses reach the shared encoder: the full objective, the full objective without alignment and instance terms, without the anatomical term, and the unguided model. A fifth variant kept the full objective and passed the visual tokens to the decoder through a stop-gradient, the configuration adopted for the clinical experiments.

| Metric | Set | Full | − alignment | − anatomy | Unguided | Full, detached decoder input |
|---|---|---|---|---|---|---|
| Macro-AUROC | internal | 0.828 ± 0.008 | 0.852 ± 0.010 | 0.834 ± 0.011 | 0.856 ± 0.004 | 0.852 ± 0.005 |
| Macro-AUROC | external | 0.846 ± 0.009 | 0.856 ± 0.010 | 0.847 ± 0.011 | 0.836 ± 0.008 | 0.854 ± 0.007 |
| Text grounding, hit | internal | 0.866 ± 0.025 | 0.631 ± 0.144 | 0.831 ± 0.102 | 0.629 ± 0.085 | 0.894 ± 0.015 |
| Text grounding, hit | external | 0.883 ± 0.016 | 0.576 ± 0.161 | 0.878 ± 0.051 | 0.596 ± 0.085 | 0.891 ± 0.020 |
| Text grounding, mass/area | internal | 3.980 ± 0.412 | 1.015 ± 0.006 | 3.250 ± 0.734 | 1.053 ± 0.063 | 3.854 ± 0.236 |
| Text grounding, mass/area | external | 3.843 ± 0.509 | 1.010 ± 0.002 | 3.264 ± 1.237 | 1.060 ± 0.063 | 3.359 ± 0.228 |
| Excess drop, text | internal | 0.290 ± 0.014 | 0.145 ± 0.354 | 0.266 ± 0.217 | -0.020 ± 0.169 | 0.282 ± 0.021 |
| Excess drop, text | external | 0.013 ± 0.092 | -0.077 ± 0.039 | -0.004 ± 0.122 | -0.159 ± 0.045 | -0.076 ± 0.044 |
| Excess drop, evidence | internal | 0.325 ± 0.065 | 0.318 ± 0.131 | 0.268 ± 0.219 | -0.094 ± 0.229 | 0.343 ± 0.092 |
| Excess drop, evidence | external | 0.012 ± 0.085 | -0.083 ± 0.050 | -0.006 ± 0.165 | -0.220 ± 0.087 | -0.083 ± 0.036 |
| ECS gap | internal | 0.001 ± 0.002 | -0.000 ± 0.002 | -0.001 ± 0.001 | -0.001 ± 0.001 | 0.000 ± 0.001 |
| ECS gap | external | -0.000 ± 0.002 | -0.001 ± 0.002 | -0.001 ± 0.002 | -0.001 ± 0.004 | -0.001 ± 0.002 |
| Evidence, mass/area | internal | 6.846 ± 0.377 | 6.735 ± 0.418 | 5.210 ± 1.454 | 5.991 ± 1.542 | 6.656 ± 0.505 |
| Evidence, mass/area | external | 6.245 ± 0.385 | 6.193 ± 0.324 | 5.224 ± 1.364 | 5.334 ± 1.465 | 6.127 ± 0.394 |

Table: **Table 4.** Controlled synthetic study, mean ± s.d. over three seeds (Supplementary Table S3 adds JS, win rate and effusion AUROC). Hit: pointing game; mass/area: map mass inside the lesion box divided by the box-area fraction (1 = chance); excess drop: classifier-logit drop after removing the top visual tokens of the named map minus the drop for random removal; ECS gap: fall in per-token log-likelihood of the asserted sentence after removing its evidence minus the fall after random removal.

Alignment did what it was designed to do on the property it targets. With the alignment and instance terms, the grounding of each finding sentence fell on the lesion box at a mass-to-area ratio of 3.98 ± 0.41 internally and 3.84 ± 0.51 externally. Without them the ratio was 1.02 ± 0.01 and 1.01 ± 0.00, which is chance, and the unguided model was equally at chance (1.05, 1.06). The pointing-game hit rate of the text grounding rose from 0.63 to 0.87 internally and from 0.58 to 0.88 externally. Text-guided deletion moved the same way on the internal split: removing the tokens the explanation named lowered the classifier logit by 0.29 ± 0.01 more than random removal, against 0.15 ± 0.35 without alignment and −0.02 ± 0.17 for the unguided model (win rates over random removal 0.71, 0.54 and 0.54).

Three results qualify this. The decoder-side test was flat in every variant (ECS gap between −0.001 and 0.001): removing a lesion did not change the likelihood of the sentence describing it more than removing a random region did. The decoder attended where it was told to while its wording did not depend on that region, so attention alignment alone did not produce causal dependence of the text on the evidence. On the shifted external set no map, the classifier's own evidence included, beat random removal reliably (evidence excess drop 0.01 ± 0.09 for the full model), so perturbation tests lose discriminative power under covariate shift and must be read against the evidence-removal reference, which is why Section 4.4 reports both. And alignment cost internal macro-AUROC (0.828 ± 0.008 against 0.852 ± 0.010 without it) although the gradient on the evidence is stopped, because the encoder still received gradients from the decoder's attention keys. Detaching the decoder input closed that path and recovered the AUROC (0.852 ± 0.005 internally, 0.854 ± 0.007 externally) while keeping the grounding (hit 0.89, mass-to-area 3.85 internally) and the internal excess drop (0.28 ± 0.02). All of these removal tests use the blurred infill on which the sufficiency–comprehensiveness term trains, so the clinical comparison across removal operators in Section 5.4 is needed to separate what alignment contributes from what that term teaches the classifier.

The anatomical term raised the evidence mass-to-area ratio from 5.21 ± 1.45 to 6.85 ± 0.38 internally and from 5.22 ± 1.36 to 6.25 ± 0.39 externally, and cut its variance across seeds about fourfold. The planted cue barely affected external effusion AUROC in any variant (0.973 to 0.992) because the synthetic effusion was easy to detect. The synthetic study therefore supports the localisation effect of the anatomical term and is not evidence that it removes a shortcut, which the clinical corpora with their real confounders can address better.

## 5.2 Diagnostic performance on internal and external data

Table 5 reports macro-AUROC for the three systems trained under the full protocol. The decoder and the alignment objective left diagnostic accuracy unchanged: on every corpus the three systems fall inside overlapping 95% intervals, and RSNA, where EviAlign-CXR reached 0.849 (0.844–0.853) against 0.840 and 0.839, is the only corpus with a visible difference. On NIH ChestX-ray14 the paired patient-level bootstrap difference between EviAlign-CXR and the classifier-only model was 0.000 (95% CI −0.003 to 0.003). Expected calibration error was 0.098 on the internal split, 0.209 on NIH, 0.143 on ChestX-Det and 0.138 on RSNA, and 0.076 on IU X-ray, so probabilities and thresholds fixed on ReXGradient transferred poorly to the NIH-derived corpora although the rankings did. On the tuberculosis corpora, which contain no training finding, the abnormality score reached AUROCs of [R] (Shenzhen), [R] (Montgomery) and [R] (TBX11K). Per-finding AUROCs are in Supplementary Table S2.

| System | ReXGradient test | NIH-14 | ChestX-Det | RSNA | IU X-ray | Shenzhen* | Montgomery* | TBX11K* |
|---|---|---|---|---|---|---|---|---|
| Classifier only | 0.881 (0.874–0.888) | 0.804 (0.797–0.810) | 0.845 (0.836–0.854) | 0.840 (0.836–0.845) | 0.881 (0.866–0.895) | [R] | [R] | [R] |
| Unguided VLM | 0.883 (0.876–0.889) | 0.799 (0.792–0.805) | 0.843 (0.834–0.851) | 0.839 (0.835–0.844) | 0.883 (0.869–0.896) | [R] | [R] | [R] |
| EviAlign-CXR | 0.881 (0.874–0.888) | 0.804 (0.797–0.810) | 0.844 (0.835–0.852) | 0.849 (0.844–0.853) | 0.882 (0.867–0.896) | [R] | [R] | [R] |

Table: **Table 5.** Macro-AUROC with 95% patient-level bootstrap CI over the harmonised findings each corpus annotates, full protocol. *Abnormality AUROC.

## 5.3 Agreement between predictions and explanations

Table 6 separates two questions that clinical F1 conflates: whether the explanation states what the classifier predicts, and whether what it states is true. In free generation the decoder volunteered few of the classifier's findings. The normal statement was the whole explanation for 54% of NIH images, and the text omitted 77–78% of the findings the classifier predicted, for the unguided and the aligned model alike (PEC F1 0.287 and 0.276). Alignment therefore changed where the decoder looked and not what it chose to say. Finding-conditioned generation raised PEC F1 to 0.635 and cut omission to 39%, at a price: 31% of the findings it asserted were not predicted at the F1 thresholds, because the 90%-specificity gate admits more positives, and 77% of asserted findings were negative in the NIH labels. That rate combines classifier precision at the gate with the label noise of report-mined NIH annotations [@oakden2020]. The template system states exactly what the classifier predicts, so its consistency F1 is 1 by construction and its hallucination rate equals the classifier's false-positive share. On IU X-ray the free-generation systems did not differ in BLEU-4 (0.387 and 0.386) or ROUGE-L (0.718 and 0.717).

| System | PEC F1 | Unsupported (%) | Omitted (%) | Hallucination vs labels (%) | Text clinical F1 | BLEU-4 (IU) | ROUGE-L (IU) |
|---|---|---|---|---|---|---|---|
| Template from classifier | 1.000 | 0 | 0 | [R] | [R] | – | – |
| Zero-shot open VLM | – | – | – | [R] | [R] | [R] | [R] |
| Unguided VLM, free generation | 0.287 | 10.4 | 76.9 | 60.0 | 0.202 | 0.387 | 0.718 |
| EviAlign-CXR, free generation | 0.276 | 12.0 | 78.2 | 61.4 | 0.200 | 0.386 | 0.717 |
| Unguided VLM, finding-conditioned | [R] | [R] | [R] | [R] | [R] | – | – |
| EviAlign-CXR, finding-conditioned | 0.635 | 31.2 | 38.8 | 76.7 | 0.263 | – | – |

Table: **Table 6.** Prediction–explanation consistency (PEC) and text clinical F1 on NIH ChestX-ray14, and text quality on IU X-ray. N-gram overlap is reported for free generation only.

## 5.4 Counterfactual faithfulness

With the blur on which the classifier is trained, the effects were large. On NIH, removing the region a sentence named lowered the logit of EviAlign-CXR by 5.76 more than random removal (win rate 1.00), and removing a finding's evidence lowered the likelihood of its sentence by 0.39 nats per token more than random removal. These values mostly reflect the sufficiency–comprehensiveness term. In the reduced protocol the blur excess drop fell from 3.65 with the full objective to 1.14 without that term and to 1.13 without alignment, and the deletion area of Grad-CAM maps, which the term does not touch, fell from 0.52 for the classifier-only model to 0.15 for EviAlign-CXR (Supplementary Table S5). The classifier had learned to lose score when blurred regions are removed, whichever map picks the region.

Under noise infill the effect shrank to a fifth and separated the variants (Table 7). EviAlign-CXR reached an excess drop of 1.05 (95% CI 1.01–1.09) on NIH with a win rate of 0.77, and 1.07 on ChestX-Det. The unguided VLM reached 1.29 and 1.53, so in the full protocol alignment did not raise text-guided deletion above that of the unguided model. Relative to the effect of the classifier's own evidence, the named region of the aligned model was as consequential as the evidence (normalised score 1.09 and 0.99) and that of the unguided model somewhat less so (0.79 and 0.90), even though its text grounding lies far from the evidence (Jensen–Shannon distance 0.53 against 0.03). The matched reduced-protocol comparison is sharper. With the full objective the excess drop was 1.92 (1.88–1.96) on NIH and the normalised score 0.98; without the alignment and instance terms they fell to 0.67 (0.64–0.71) and 0.37, and for the unguided model to 0.36 (0.33–0.40) and 0.25. Removing the sufficiency–comprehensiveness term lowered the excess drop to 1.46 (1.43–1.51) and left the normalised score at 0.99, so alignment accounts for most of the effect that survives an unseen operator. ChestX-Det showed the same order. [R: seed-to-seed variation of the reduced-protocol rows.] Why the margin of alignment is smaller in the full protocol than in the reduced one cannot be settled with these runs, since the two protocols differ in training length and data as well as in the decoder.

| System | Protocol | Set | Pairs | Excess drop, text (95% CI) | Win rate vs random | Normalised TGD | ECS gap (95% CI) | JS (text, evidence)† |
|---|---|---|---|---|---|---|---|---|
| Unguided VLM | full | NIH-14 | 4,618 | 1.29 (1.25–1.34) | 0.84 | 0.79 | 0.165 (0.158–0.171) | 0.533 |
| Unguided VLM | full | ChestX-Det | 3,943 | 1.53 (1.48–1.59) | 0.84 | 0.90 | 0.135 (0.127–0.142) | 0.550 |
| EviAlign-CXR | full | NIH-14 | 5,270 | 1.05 (1.01–1.09) | 0.77 | 1.09 | 0.063 (0.056–0.069) | 0.031 |
| EviAlign-CXR | full | ChestX-Det | 4,749 | 1.07 (1.02–1.11) | 0.78 | 0.99 | 0.060 (0.053–0.067) | 0.034 |
| Unguided VLM | reduced | NIH-14 | 4,803 | 0.36 (0.33–0.40) | 0.62 | 0.25 | 0.074 (0.069–0.078) | 0.509 |
| Unguided VLM | reduced | ChestX-Det | 4,225 | 0.45 (0.41–0.49) | 0.63 | 0.32 | 0.059 (0.054–0.064) | 0.498 |
| − alignment and instance | reduced | NIH-14 | 4,784 | 0.67 (0.64–0.71) | 0.72 | 0.37 | 0.033 (0.028–0.037) | 0.492 |
| − alignment and instance | reduced | ChestX-Det | 4,285 | 0.50 (0.46–0.53) | 0.66 | 0.27 | 0.036 (0.032–0.041) | 0.452 |
| − sufficiency / comprehensiveness | reduced | NIH-14 | 5,015 | 1.46 (1.43–1.51) | 0.85 | 0.99 | 0.068 (0.063–0.073) | 0.041 |
| − sufficiency / comprehensiveness | reduced | ChestX-Det | 4,326 | 1.24 (1.19–1.28) | 0.81 | 0.89 | 0.053 (0.049–0.059) | 0.041 |
| EviAlign-CXR | reduced | NIH-14 | 4,946 | 1.92 (1.88–1.96) | 0.92 | 0.98 | 0.064 (0.060–0.068) | 0.044 |
| EviAlign-CXR | reduced | ChestX-Det | 4,518 | 1.83 (1.79–1.87) | 0.90 | 0.90 | 0.038 (0.034–0.042) | 0.050 |

Table: **Table 7.** Counterfactual faithfulness under Gaussian-noise infill (15% of tokens) on a fixed subset of (image, finding) pairs that are predicted and asserted; explanations are finding-conditioned. Excess drop: classifier-logit drop after removing the named tokens minus the drop for random removal of equal area. Normalised TGD: that excess divided by the same quantity for removal of the classifier's own evidence. ECS gap: fall in per-token log-likelihood of the sentence asserting the finding after removing its evidence, minus the fall after random removal. Reduced-protocol rows show seed 0. †Optimised during training; a consistency statistic, not a faithfulness one.

The decoder-side test did not favour alignment. The ECS gap of EviAlign-CXR was 0.063 (0.056–0.069) on NIH against 0.165 (0.158–0.171) for the unguided model in the full protocol, and 0.064 against 0.074 in the reduced protocol; every reduced variant lay between 0.033 and 0.084. Under blur the reduced-protocol gap was 0.19 for the full objective, 0.19 without alignment and 0.04 without the sufficiency–comprehensiveness term, so it followed the regulariser and not the alignment. As in the synthetic study, the decoder grounded its sentences on the evidence without the likelihood of the sentence depending on it more than on an arbitrary region. [R: aggregated alignment, text-guided deletion and ECS gap under noise infill.]

## 5.5 Localisation against radiologist boxes

On NIH boxes the exact evidence of EviAlign-CXR reached a pointing-game hit rate of 0.58, against 0.30 for Grad-CAM (Table 8); on ChestX-Det the figures were 0.73 and 0.34 and on RSNA 0.69 and 0.48, and mass-to-area ratios of the evidence were 5.6, 4.5 and 4.8 against 2.8, 2.5 and 3.5 for Grad-CAM. The auxiliary terms did not improve where the evidence points: the classifier-only model reached 0.67, 0.71 and 0.75 with mass-to-area ratios of 5.4, 4.6 and 4.8. The text grounding of EviAlign-CXR in free generation reached 0.71, 0.82 and 0.85, at or above its evidence maps, and 0.63, 0.74 and 0.77 in finding-conditioned generation. The text grounding of the unguided VLM never peaked inside a box (hit 0.00 on all three corpora) and placed less mass inside the boxes than a uniform map would (mass-to-area 0.4–0.6). By finding, the finding-conditioned grounding was strong for cardiomegaly (hit 1.00 on NIH), emphysema (0.92 on ChestX-Det) and effusion (0.87) and weak for pneumothorax (0.33 on NIH, 0.27 on ChestX-Det). On the TB boxes of TBX11K, where no training finding exists, the probability-weighted evidence reached [R]. After re-initialising the evidence head the median Spearman correlation with the trained maps was [R], and [R] after also re-initialising the last four encoder blocks (Supplementary Table S4).

| Map | NIH hit | NIH mass/area | ChestX-Det hit | ChestX-Det mass/area | RSNA hit | TBX11K hit |
|---|---|---|---|---|---|---|
| Grad-CAM (classifier only) | 0.30 | 1.84 | 0.38 | 1.91 | 0.51 | [R] |
| Grad-CAM (EviAlign-CXR) | 0.30 | 2.75 | 0.34 | 2.48 | 0.48 | [R] |
| Integrated Gradients (EviAlign-CXR)† | 0.64 | 2.87 | 0.81 | 4.07 | 0.73 | [R] |
| Evidence (classifier only) | 0.67 | 5.43 | 0.71 | 4.61 | 0.75 | [R] |
| Evidence (unguided VLM) | 0.66 | 5.21 | 0.70 | 4.00 | 0.76 | [R] |
| Evidence (EviAlign-CXR) | 0.58 | 5.57 | 0.73 | 4.51 | 0.69 | [R] |
| Text grounding (unguided VLM) | 0.00 | 0.61 | 0.00 | 0.57 | 0.00 | [R] |
| Text grounding (EviAlign-CXR) | 0.71 | 4.72 | 0.82 | 4.04 | 0.85 | [R] |

Table: **Table 8.** Localisation. Hit: pointing game with 15-pixel tolerance. Mass/area: share of map mass inside boxes divided by the box-area fraction (1 = chance). Text grounding is from free generation. †Subsample of 500 images per corpus.

## 5.6 What each component contributes

Table 9 reports the matched ablations under noise infill. Removing the alignment and instance terms lowered the excess drop from 1.92 to 0.67 and the text-grounding hit rate from 0.63 to 0.05, the level of the unguided model (0.36 and 0.00). Removing only the instance term changed neither (1.77 and 0.75), so these data give no support for the location-prior shortcut that the term is designed to block. Removing the anatomical prior lowered the excess drop to 1.62 and raised the evidence hit rate from 0.60 to 0.68, so the term did not improve pointing on NIH. Removing the sufficiency–comprehensiveness term lowered the excess drop to 1.46, left the normalised score at 0.99 and raised the text-grounding hit rate to 0.73. Macro-AUROC on the subset stayed between 0.779 and 0.784 in all variants, and PEC F1 between 0.57 and 0.62 with no pattern. [R: bidirectional alignment, KL, the encoder-update variant, the GPT-2 cross-attention decoder, the BiomedCLIP backbone and the aggregated-alignment control; seed variation.] All ablations use the reduced protocol and are compared with the reduced-protocol main model.

| Variant | Macro-AUROC (NIH) | PEC F1 | Excess drop, text | Normalised TGD | ECS gap | Text hit (NIH) | Evidence hit (NIH) |
|---|---|---|---|---|---|---|---|
| EviAlign-CXR (seed 0) | 0.781 | 0.598 | 1.92 | 0.98 | 0.064 | 0.63 | 0.60 |
| − alignment and instance | 0.779 | 0.571 | 0.67 | 0.37 | 0.033 | 0.05 | 0.64 |
| − instance | 0.781 | 0.588 | 1.77 | 0.98 | 0.084 | 0.75 | 0.63 |
| − anatomical prior | 0.781 | 0.621 | 1.62 | 0.92 | 0.083 | 0.76 | 0.68 |
| − sufficiency / comprehensiveness | 0.782 | 0.580 | 1.46 | 0.99 | 0.068 | 0.73 | 0.63 |
| Unguided VLM (seed 0) | 0.784 | 0.572 | 0.36 | 0.25 | 0.074 | 0.00 | 0.62 |
| Aggregated alignment (CAMANet-style) | [R] | [R] | [R] | [R] | [R] | [R] | [R] |
| bidirectional alignment | [R] | [R] | [R] | [R] | [R] | [R] | [R] |
| KL(b ‖ a) instead of JS | [R] | [R] | [R] | [R] | [R] | [R] | [R] |
| decoder losses update the encoder | [R] | [R] | [R] | [R] | [R] | [R] | [R] |
| GPT-2 cross-attention decoder | [R] | [R] | [R] | [R] | [R] | [R] | [R] |
| BiomedCLIP backbone | [R] | [R] | [R] | [R] | [R] | [R] | [R] |

Table: **Table 9.** Ablations on a fixed NIH ChestX-ray14 subset under the reduced protocol, with noise infill and finding-conditioned explanations. Rows show seed 0.

## 5.7 Qualitative examples and reader study

Figure 2 shows NIH and ChestX-Det cases with radiologist boxes, classifier evidence, explanation grounding and Grad-CAM. [R: two or three sentences on observed agreement and failure cases, including one where the explanation grounds correctly but the finding is false.] In the reader study (Table 10), radiologists judged [R]% of asserted findings correct for EviAlign-CXR and [R]% for the unguided VLM (McNemar p = [R]) and judged the overlay to cover the finding in [R]% and [R]% of cases; inter-reader κ was [R].

[FIGURE 2: qualitative grid, rows = cases, columns = radiograph with boxes, classifier evidence, explanation grounding, Grad-CAM; generated explanation printed under each row.]

| System | Correct statements (%) | Overlay covers finding (%) | Usefulness, median (IQR) | Omitted findings per image |
|---|---|---|---|---|
| Unguided VLM | | | | |
| EviAlign-CXR | | | | |

Table: **Table 10.** Reader study (two radiologists, 200 external images).

# 6. Discussion

Joint training with the alignment objective left diagnostic accuracy unchanged and moved the decoder's grounding onto the classifier's evidence, with a Jensen–Shannon distance of 0.03 against 0.53 for the unguided decoder. The evidence itself, an exact decomposition of the logits, located findings better than Grad-CAM on all three box-annotated corpora, which supports using a decomposable head where an explanation will be shown next to a diagnosis. The auxiliary terms did not improve that localisation: the classifier-only model pointed as well as the evidence of the full model, and the anatomical prior raised neither the hit rate nor the accuracy. Viviano et al. found that correcting attributions need not track improved generalisation [@viviano2021], and the present results agree: the terms are justified by what they do to the explanation and not by any gain in accuracy.

The faithfulness result is narrower than the blur-based numbers suggest. Text-guided deletion and counterfactual sensitivity under the blur used in training were five times larger than under noise, and Grad-CAM maps also scored a deletion area about 3.5 times smaller, which marks the effect as a property of the classifier trained with the sufficiency–comprehensiveness term and not of the explanations. The measurement that survives is the noise-infill comparison. In matched reduced-protocol ablations, the sentence-named region mattered as much as the classifier's own evidence with alignment (normalised score 0.98) and about a third as much without it (0.37), and removing the sufficiency–comprehensiveness term kept the score at 0.99. The full protocol narrowed the contrast, because the unguided decoder trained for twelve epochs also named regions that mattered (0.79 and 0.90), and a claim about the effect of alignment beyond the reduced protocol would need seed replicates at full scale. These numbers describe agreement between where a sentence points and where the classifier's score depends; they do not establish that the classifier's evidence is the right evidence, which only the box-based localisation addresses and then only for the findings with boxes.

The decoder-side test showed no benefit. The likelihood of a generated sentence fell no more after removing the finding's evidence than after removing an arbitrary region, in every variant, and the gap of the unguided model was as large as or larger than that of the aligned model in both protocols, as in the synthetic study. A decoder can attend to the right place while generating words from other cues, the gap between where a model looks and what it uses that work on attention as explanation describes [@jain2019; @wiegreffe2019] and that faithfulness tests for natural-language explanations are built to detect [@atanasova2023; @parcalabescu2024]. An attention-level loss cannot by itself force causal use of the attended region, and a method that sets out to do so needs an objective on the decoder's output, for example a counterfactual consistency loss, which this study does not test.

Free generation also exposed a coverage problem that the alignment did not touch. The decoder wrote the normal statement for 54% of NIH images and omitted 77–78% of the classifier's findings, so a sentence-level explanation produced in this way is silent about most of what the classifier detects. Finding-conditioned generation corrects this by letting the classifier choose what is explained, at the price of inheriting the classifier's false positives, and it makes agreement between prediction and text a property of the design and not a result. The aggregated-alignment control places the contribution relative to CAMANet [@wang2024camanet]: a single class-agnostic map can make text attention resemble classifier evidence on average, but finding-specific sentences then share one region, visible in [R].

Several limitations bound the conclusions. Training labels come from CheXbert and inherit its errors [@smit2020]. It marks "stable heart size" as cardiomegaly in some reports, for example, and the rule labeler that assigns sentences to findings agreed with it with κ between 0.70 (lung opacity) and 0.98 (atelectasis), finding fewer cardiomegaly cases than CheXbert (3,778 against 4,792 in 20,000 reports; Supplementary Table S1). Labeler errors therefore put alignment pressure on the wrong tokens for some sentences. In 18% of retained studies the frontal image had no view label and was inferred by a probe with 0.90 held-out accuracy. NIH labels are mined from reports, the mapping of "Infiltration" to lung opacity is approximate [@oakden2020], and the 77% share of asserted findings that were negative in NIH labels cannot be separated from label error. Calibration did not transfer to the NIH-derived corpora (expected calibration error 0.14–0.21), and the 90%-specificity gate was set on ReXGradient. Anatomy masks come from a segmentation model, and a finding whose evidence lies partly outside the dilated lung and heart fields, such as a subpulmonic effusion, receives a small penalty for correct evidence. The head decomposition is exact for the head and not for the encoder. Blurred infill reduces but does not remove the distribution shift of removal tests [@rong2022], and noise infill introduces a shift of its own, so no single operator is definitive. The text grounding of the unguided decoder placed less mass in the boxes than a uniform map; a mechanism such as attention concentrating on a fixed position is plausible but was not tested. ReXGradient images are distributed at a quarter of their original resolution, which limits small-nodule detection. Ablations and seed replicates use a reduced protocol, so their absolute values are not comparable with the full-protocol model. The external corpora come from the United States and China, only adults were studied, and prospective evaluation in a reading workflow is outside the scope of this study.

# 7. Conclusion

Training an explanation decoder against the exact evidence of a co-trained classifier moves each sentence's grounding onto that evidence without changing how well the classifier diagnoses. An exact patch decomposition of the logits defines the evidence, and a stop-gradient Jensen–Shannon term with an instance-level contrast aligns each sentence to it. Whether the resulting explanations are faithful depends on how faithfulness is measured. Removal tests that use blur reward a classifier trained to react to blur, and with an operator the classifier had not seen, the region a sentence named mattered as much as the classifier's own evidence in the matched comparison with alignment and much less without it, while the wording of the sentence did not come to depend on that evidence. Reporting removal tests under more than one operator and against matched ablations, and not only agreement between maps optimised to agree, is the practice this work recommends for explainable diagnostic vision–language models.

# Data and code availability

All corpora are downloadable from their providers under the licences listed in Section 4.1. ReXGradient-160K requires acceptance of its non-commercial licence on Hugging Face and attribution to Gradient Health, and RSNA Pneumonia requires attribution to the RSNA. Code, configurations, corpus adapters, the synthetic benchmark and evaluation scripts are released at [repository URL].
