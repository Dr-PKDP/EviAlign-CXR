# Lit survey B (CXR VLMs, grounded reporting, evaluation) — verified by subagent

## 1. Classification & generalization
[1] Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8. CVPR 2017. doi:10.1109/CVPR.2017.369
[2] Rajpurkar P et al. CheXNet. arXiv:1711.05225, 2017.
[3] Irvin J et al. CheXpert. AAAI 33, 2019. doi:10.1609/aaai.v33i01.3301590
[4] Bustos A, Pertusa A, Salinas JM, de la Iglesia-Vayá M. PadChest. Med Image Anal 66:101797, 2020. doi:10.1016/j.media.2020.101797
[5] Nguyen HQ et al. VinDr-CXR. Sci Data 9:429, 2022. doi:10.1038/s41597-022-01498-w
[6] Johnson AEW et al. MIMIC-CXR. Sci Data 6:317, 2019. doi:10.1038/s41597-019-0322-0
[7] Zech JR et al. Variable generalization performance ... pneumonia. PLoS Med 15:e1002683, 2018. doi:10.1371/journal.pmed.1002683
[8] Pooch EHP, Ballester PL, Barros RC. Can we trust deep learning models diagnosis? Domain shift CXR. arXiv:1909.01940 (TIA 2020 LNCS 12502)
[9] Cohen JP, Hashir M, Brooks R, Bertrand H. On the limits of cross-domain generalization in automated X-ray prediction. MIDL, PMLR 121, 2020. https://proceedings.mlr.press/v121/cohen20a.html
[10] Oakden-Rayner L. Exploring large-scale public medical image datasets. Acad Radiol 27(1), 2020. doi:10.1016/j.acra.2019.10.006
[11] DeGrave AJ, Janizek JD, Lee SI. AI for radiographic COVID-19 detection selects shortcuts over signal. Nat Mach Intell 3, 2021. doi:10.1038/s42256-021-00338-7
[12] Arun N et al. Assessing the trustworthiness of saliency maps ... Radiol AI 3:e200267, 2021. doi:10.1148/ryai.2021200267
[13] Saporta A et al. Benchmarking saliency methods for chest X-ray interpretation. Nat Mach Intell 4, 2022. doi:10.1038/s42256-022-00536-x
[14] Sellergren AB et al. Simplified transfer learning for chest radiography models using less data. Radiology 305, 2022. doi:10.1148/radiol.212482
[15] Xu S et al. ELIXR. arXiv:2308.01317, 2023.
[16] Pérez-García F et al. Exploring scalable medical image encoders beyond text supervision (RAD-DINO). Nat Mach Intell 2025. doi:10.1038/s42256-024-00965-w
[17] Yang Z et al. CheXFound. IEEE TMI 2025. doi:10.1109/TMI.2025.3581907
[18] Cohen JP et al. TorchXRayVision. MIDL PMLR 172, 2022. https://proceedings.mlr.press/v172/cohen22a.html

## 2. VLP
[19] Zhang Y, Jiang H, Miura Y, Manning CD, Langlotz CP. ConVIRT. MLHC PMLR 182, 2022. https://proceedings.mlr.press/v182/zhang22a.html
[20] Huang SC, Shen L, Lungren MP, Yeung S. GLoRIA. ICCV 2021. doi:10.1109/ICCV48922.2021.00391
[21] Boecking B et al. BioViL. ECCV 2022. doi:10.1007/978-3-031-20059-5_1
[22] Bannur S et al. BioViL-T. CVPR 2023. doi:10.1109/CVPR52729.2023.01442
[23] Wang Z, Wu Z, Agarwal D, Sun J. MedCLIP. EMNLP 2022. doi:10.18653/v1/2022.emnlp-main.256
[24] Tiu E et al. CheXzero. Nat Biomed Eng 6, 2022. doi:10.1038/s41551-022-00936-9
[25] Wu C, Zhang X, Zhang Y, Wang Y, Xie W. MedKLIP. ICCV 2023. doi:10.1109/ICCV51070.2023.01954
[26] Zhang X, Wu C, Zhang Y, Xie W, Wang Y. KAD. Nat Commun 14, 2023. doi:10.1038/s41467-023-40260-7
[27] Zhang S et al. BiomedCLIP. NEJM AI 2025. doi:10.1056/AIoa2400640
[28] Lin W et al. PMC-CLIP. MICCAI 2023. doi:10.1007/978-3-031-43993-3_51
[29] You K et al. CXR-CLIP. MICCAI 2023. doi:10.1007/978-3-031-43895-0_10
[30] Codella NCF et al. MedImageInsight. arXiv:2410.06542, 2024.

## 3. Report generation / VLMs / explainable
[31] Chen Z, Song Y, Chang TH, Wan X. R2Gen. EMNLP 2020. doi:10.18653/v1/2020.emnlp-main.112
[32] Wang X, Peng Y, Lu L, Lu Z, Summers RM. TieNet. CVPR 2018. doi:10.1109/CVPR.2018.00943
[33] Jing B, Xie P, Xing E. On the automatic generation of medical imaging reports. ACL 2018. doi:10.18653/v1/P18-1240
[34] Tanida T, Müller P, Kaissis G, Rueckert D. RGRG. CVPR 2023. doi:10.1109/CVPR52729.2023.00718
[35] Jin H, Che H, Lin Y, Chen H. PromptMRG. AAAI 38, 2024. doi:10.1609/aaai.v38i3.28038
[36] Müller P, Kaissis G, Rueckert D. ChEX. ECCV 2024. doi:10.1007/978-3-031-72664-4_6
[37] Hyland SL et al. MAIRA-1. arXiv:2311.13668, 2023.
[38] Bannur S et al. MAIRA-2: Grounded radiology report generation. arXiv:2406.04449, 2024.
[39] Chen Z et al. CheXagent. arXiv:2401.12208, 2024.
[40] Li C et al. LLaVA-Med. NeurIPS D&B 2023. doi:10.52202/075280-1240
[41] Zambrano Chaves JM et al. LLaVA-Rad. Nat Commun 2025. doi:10.1038/s41467-025-58344-x
[42] Tu T et al. Towards generalist biomedical AI (Med-PaLM M). NEJM AI 2024. doi:10.1056/AIoa2300138
[43] Sellergren A et al. MedGemma technical report. arXiv:2507.05201, 2025.
[44] Deperrois N et al. RadVLM. arXiv:2502.03333, 2025.
[45] Chambon P et al. CheXpert Plus. arXiv:2405.19538, 2024.
[46] Zhang X, Acosta JN, Miller J, Huang O, Rajpurkar P. ReXGradient-160K. arXiv:2505.00228, 2025.
[47] Zhang X et al. ReXrank. arXiv:2411.15122, 2024.
[48] Kayser M, Emde C, Camburu OM, Parsons G, Papiez B, Lukasiewicz T. Explaining chest X-ray pathologies in natural language. MICCAI 2022. doi:10.1007/978-3-031-16443-9_67
[49] Yang Y et al. LaBo. CVPR 2023. doi:10.1109/CVPR52729.2023.01839
[50] Yang Y et al. KnoBo (textbook remedy). NeurIPS 2024. doi:10.52202/079017-2879
[51] Yan A et al. Robust and interpretable medical image classifiers via CBMs. arXiv:2310.03182, 2023.
[52] Liu B et al. GEMeX. ICCV 2025. doi:10.1109/ICCV51701.2025.01979
[53] Han T et al. CLEAR auditable foundation model. Nat Biomed Eng 2026. doi:10.1038/s41551-026-01741-4

## 4. Evaluation
[54] Smit A et al. CheXbert. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.117
[55] Jain S et al. RadGraph. NeurIPS D&B 2021.
[56] Yu F et al. Evaluating progress in automatic CXR report generation (RadGraph F1, RadCliQ). Patterns 4:100802, 2023. doi:10.1016/j.patter.2023.100802
[57] Ostmeier S et al. GREEN. Findings EMNLP 2024. doi:10.18653/v1/2024.findings-emnlp.21
[58] Zhao W et al. RaTEScore. EMNLP 2024. doi:10.18653/v1/2024.emnlp-main.836
[60] Ramesh V, Chi NA, Rajpurkar P. Removing hallucinated references to non-existent priors. ML4H PMLR 193, 2022.
[62] Miura Y et al. Improving factual completeness and consistency. NAACL 2021. https://aclanthology.org/2021.naacl-main.416
[63] Liu G et al. Clinically accurate CXR report generation. MLHC PMLR 106, 2019.
[64] Delbrouck JB et al. Semantic rewards. Findings EMNLP 2022. doi:10.18653/v1/2022.findings-emnlp.319

## Novelty threats
- CAMANet: Wang J, Bhalerao A, Yin T, See S, He Y. IEEE JBHI 2024. doi:10.1109/JBHI.2024.3354712 — CAM-supervised cross-modal attention + visual-textual attention consistency. CLOSEST.
- WeNLEX: Rio-Torto I, Cardoso JS, Teixeira LF. arXiv:2603.18752, 2026 — faithfulness-constrained NLEs for multilabel CXR classifier, in-model joint training.
- AdaMatch-Cyclic: Chen W et al. ACL 2024. doi:10.18653/v1/2024.acl-long.514
- FG-CXR/Gen-XAI: Pham TT et al. ACCV 2024. doi:10.1007/978-981-96-0960-4_5 (gaze)
- CheXPoT: Guo J, Ansari R, Youssef A. arXiv:2611.00785, 2026 — post-hoc predicate verification vs Grad-CAM evidence.
- Attention without grounding: Sadanandan B, Behzadan V. arXiv:2607.18577, 2026 — VLM heatmaps not faithful vs PadChest boxes/CheXlocalize/occlusion.
- XBench: Luo H et al. arXiv:2510.19599, 2025.
- TieNet [32], PromptMRG [35] joint cls + generation.
