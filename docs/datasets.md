# Dataset decision (verified by subagent, Nov 2026)
TRAIN (Config A, recommended): ReXGradient-160K (Zhang, Acosta, Miller, Huang, Rajpurkar; arXiv:2505.00228). HF rajpurkarlab/ReXGradient-160K, auto-approved click-through, Harvard non-commercial licence (no redistribution; A5 "no download beyond needed"). 3 US health systems, 79 sites. Train 140k studies/238,968 img/95,716 pts; val 10k/17,007/6,964; public test 10k/17,029/6,807; private test 10k (ReXrank). 16-bit PNG, 25% downsampled. Reports: Indication, Comparison, Findings, Impression. No structured labels -> CheXbert (StanfordAIMI/RRG_scorers chexbert.pth, not gated) + rule labeler for NIH extras. ~400 boxes interstitial.
TRAIN (Config B, strictly open): NIH ChestX-ray14 (no reports).
EXTERNAL (fully open):
- NIH ChestX-ray14 (Wang 2017 doi:10.1109/CVPR.2017.369): 112,120 img / 30,805 pts; official test_list; BBox_List_2017.csv (984 boxes, 8 classes, unverified count). Licence: no restrictions, cite + acknowledge NIH CC.
- ChestX-Det (Liu, Lian, Yu arXiv:2006.10550; Deepwise GitHub, Apache-2.0): 3,543 NIH images, 13 classes boxes+masks by 3 radiologists.
- RSNA Pneumonia (Shih et al. Radiol AI 2019 doi:10.1148/ryai.2019180041): NIH-derived, pneumonia boxes; research+commercial with attribution.
- IU X-ray / OpenI (Demner-Fushman et al. JAMIA 23(2):304–310, 2016, doi:10.1093/jamia/ocv080): ~3,955 reports/7,470 img (unverified; JAMIA abstract 3,996/8,121), CC BY-NC-ND 4.0, reports + MeSH.
- Shenzhen (662; 336 TB) & Montgomery (138; 58 TB) (Jaeger et al. QIMS 4(6), 2014, PMC4256233).
- TBX11K (Liu et al. IEEE TPAMI 46(4):2316–2332, CC BY 4.0): 11,200 img, TB boxes.
- CheXmask (Gaggion et al., PhysioNet doi:10.13026/3705-zg36, CC BY 4.0): lung/heart masks for NIH, CheXpert, MIMIC, PadChest, VinDr (not ReXGradient).
GREY (click-through): VinDr-CXR via Kaggle/vindr.ai (Nguyen et al. Sci Data 9:429 2022), 18k, 22 local findings boxes, Vietnam.
OUT: MIMIC-CXR, BRAX, VinDr PhysioNet (credentialed), CANDID-PTX; CheXpert/CheXpert Plus/CheXlocalize/PadChest (RUA) grey.
WEIGHTS: avoid RAD-DINO (pretrained on all NIH + PadChest), MedGemma (possible NIH), LLaVA-Rad (VinDr, China). Safe: DINOv2 (Apache), BiomedCLIP (MIT, PMC-15M), CXR-BERT/BioViL (MIMIC only). CheXagent unverified.
