# Dynamic Multi-modal Causal Graph Emotion System (DMCGES)
## A Dynamic Multimodal Causal Graph Framework for Standardized Emotion Recognition in Conversations

### Highlights
- Proposal of **DMCGES**, a **Dynamic Multimodal Causal Graph Emotion System** for **Emotion Recognition in Conversation (ERC)**.
- Explicit enforcement of **temporal causality**, preventing information leakage from future dialogue turns.
- Integration of a **speaker-specific GRU memory module** to model individual emotional trajectories.
- Robust **multimodal fusion** of text, audio, and vision, enhanced with a **cross-modal Variational Autoencoder (VAE)**.
- Introduction of a **standardized JSONL-based multimodal representation**, improving reproducibility and interoperability.
- Extensive evaluation on **IEMOCAP** and **MELD**, outperforming state-of-the-art models such as **SACCMA**.
- Alignment with the **IEEE 7010-2020** standard for human-centric and well-being–aware AI systems.

---

### Authors

- **Ronghao Pan** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=80lntLMAAAAJ) · [ORCID](https://orcid.org/0009-0008-7317-7145)

- **José Antonio García-Díaz** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=ek7NIYUAAAAJ) · [ORCID](https://orcid.org/0000-0002-3651-2660)

- **Rafael Valencia-García** — University of Murcia  
  [Google Scholar](https://scholar.google.com/citations?user=GLpBPNMAAAAJ) · [ORCID](https://orcid.org/0000-0003-2457-1791)  

---

### Publication
Preprint submitted to *Computer Standards & Interfaces* (Elsevier).

---

### Abstract
Understanding emotions in conversations is a fundamental challenge in affective computing. Emotional expressions evolve dynamically across dialogue turns and depend on multimodal cues such as speech, text, and facial behavior. However, existing multimodal models often rely on global attention mechanisms that overlook causal constraints. This allows information leakage from future turns and neglect of the speaker's emotional evolution. To address these limitations, we propose the  Dynamic Multimodal Causal Graph Emotion System (DMCGES). DMCGES integrates a restricted dynamic causal graph to ensure temporal coherence, as well as a speaker-specific memory module (GRU) to capture affective trajectories and enhance multimodal alignment and robustness. The framework aligns with the IEEE 7010-2020 standard, which emphasizes integrating human well-being as a fundamental design principle in autonomous and intelligent systems. Experiments on the IEMOCAP and MELD benchmark datasets demonstrate that DMCGES outperforms state-of-the-art approaches in terms of accuracy and F1 score. On the IEMOCAP dataset, DMCGES achieved an accuracy of 69.36\% and an F1 score of 69.49\%, representing relative improvements of 1.95\% and 2.36\%, respectively. On the MELD dataset, our model achieved an accuracy of 62.38\% and an F1 score of 62.03\%, improving upon SACCMA's results by 0.1\% in accuracy and 2.73\% in F1 score.

---

### System Architecture

The following figure illustrates the overall architecture of **DMCGES**. The framework processes multimodal inputs (text, audio, and vision) through modality-specific encoders, combines them with a speaker-aware memory module, and performs causal reasoning using a restricted dynamic causal graph. A cross-modal VAE further enhances robustness by reconstructing acoustic features from text and visual cues.
![DMCGES Architecture](architecture.pdf)

---

### Methods
1. **Multimodal Feature Extraction**
   - **Text:** BERT embeddings (CLS token).
   - **Audio:** MFCC-based acoustic features.
   - **Vision:** MTCNN-based face detection followed by ViT facial expression embeddings.
   - All features are stored in a **standardized JSONL schema** preserving dialogue structure and speaker identity.

2. **Speaker-Specific Emotional Memory**
   - A **GRU-based memory module** tracks the emotional state of each speaker across dialogue turns.
   - Speaker memories are updated independently to capture personal emotional trajectories.

3. **Restricted Dynamic Causal Graph**
   - Dialogue turns are modeled as nodes in a directed graph.
   - Attention-based message passing is constrained by a **causal temporal window**, preventing access to future turns.
   - A temporal mask enforces causal consistency during self-attention.

4. **Cross-Modal Variational Autoencoder (VAE)**
   - Learns a shared latent emotional space by reconstructing audio features from text and visual representations.
   - Improves robustness when one modality is missing or corrupted.

---

### Results

#### Performance comparison on IEMOCAP and MELD: *Baseline results are taken from* [Guo et al., 2024].

| Model | IEMOCAP Happy (F1) | Sadness (F1) | Neutral (F1) | Anger (F1) | Excited (F1) | Frus. (F1) | AVG ACC | AVG F1 | MELD ACC | MELD F1 |
|------|-------------------:|-------------:|-------------:|-----------:|-------------:|-----------:|--------:|-------:|---------:|--------:|
| bc-LSTM | 35.63 | 62.90 | 53.00 | 59.24 | 58.85 | 59.41 | 56.32 | 56.91 | 57.80 | 56.40 |
| CMN | 30.38 | 62.41 | 52.39 | 59.83 | 60.25 | 60.69 | 61.90 | 56.13 | – | – |
| ICON | 32.80 | 74.40 | 60.60 | 68.20 | 68.40 | 66.20 | 64.00 | 63.50 | – | – |
| DialogueRNN | 33.18 | 78.80 | 59.21 | **68.25** | 71.86 | 58.91 | 63.40 | 62.75 | 60.20 | 57.00 |
| DialogueGCN | 42.70 | 84.54 | 63.54 | 64.19 | 63.08 | 66.99 | 65.25 | 64.18 | 59.40 | 58.10 |
| DIMMN | 30.20 | 74.20 | 59.00 | 62.70 | 72.50 | 66.60 | 64.70 | 64.10 | 60.60 | 58.60 |
| MSDFSKC | 37.90 | 77.00 | 56.90 | 61.90 | 66.30 | 59.80 | 61.40 | 61.20 | – | – |
| SACCMA | 38.60 | **86.53** | 64.90 | 64.59 | **74.52** | 62.99 | 67.41 | 67.10 | 62.30 | 59.30 |
| **DMCGES (ours)** | **48.67** | 80.67 | **70.04** | 67.59 | **73.37** | 67.36 | **69.36** | **69.49** | **62.38** | **62.03** |


DMCGES is evaluated on two benchmark datasets:

- **IEMOCAP** (dyadic, longer dialogues, balanced emotions)
- **MELD** (multi-party, short dialogues, strong class imbalance)

The model achieves:

- **IEMOCAP:**  
  - Accuracy: **69.36%**  
  - Weighted F1: **69.49%**  
  - +1.95% F1 and +2.36% accuracy over SACCMA

- **MELD:**  
  - Accuracy: **62.38%**  
  - Weighted F1: **62.03%**  
  - +0.1% F1 and +2.73% accuracy over SACCMA

---

### Ablation Study

| Modality | IEMOCAP ACC | IEMOCAP F1 | MELD ACC | MELD F1 |
|---------|------------:|-----------:|---------:|--------:|
| T | 63.56 | 63.32 | 58.97 | 60.39 |
| V | 41.18 | 40.06 | 43.18 | 40.48 |
| A | 41.12 | 38.59 | 47.74 | 36.42 |
| T + A | 68.19 | 68.03 | 61.92 | 60.34 |
| T + V | 67.45 | 67.39 | 61.61 | 61.45 |
| V + A | 51.17 | 50.51 | 43.75 | 41.01 |
| **T + V + A (ours)** | **69.36** | **69.49** | **62.38** | **62.03** |

An ablation analysis confirms:

- **Text** is the most informative single modality.
- **Multimodal fusion (Text + Audio + Vision)** consistently yields the best performance.
- The full DMCGES configuration outperforms unimodal and bimodal variants on both datasets.

---

### Acknowledgments
Pending

---

### Citation
Pending
