# INHA Dacon LLM-QA

This repository contains code to analyze and answer questions about Korean economy articles. The project experiments with multiple tuning strategies including fine-tuning with LLaMA3-openko, prefix tuning, and QLoRA methods.

[Blog Review](https://hytric.github.io/project/Dacon_LLM/)

## Environment

- **OS:** Linux
- **Python:** 3.10
- **GPU:** A100 or RTX6000
- **CUDA:** 12.1

## Competition Information

- **Dacon Competition Rules:** [Visit here](https://dacon.io/competitions/official/236291/overview/rules)
- **Current Ranking:** 20th (Score: 0.806)

## Models & Approaches

- **Models:**
  - POLAR-14B_qlora
  - POLAR-14B_zero-QLoRA
  - LLaMA3-openko (fine-tuned)

- **Approaches:**
  - Fine-tuning with LLaMA3-openko
  - Prefix tuning
  - QLoRA tuning
  - Other experimental methods


## How to Run

1. **Fine-Tuning (LLaMA3-openko):**
    ```bash
    python LLaMA3-openko.py
     ```

2.	**Prefix Tuning:**
    ```bash
    python Prefix_Polar.py
     ```

2.	**QLoRA Tuning:**
    ```bash
    python QLoRA_Polar.py
    ```
4.	**Data Preprocessing & Analysis:**  
    Refer to the instructions in the data_preprocessing/README.md file.

5. **Base Code:**
   These base code scripts provide essential utilities for model inference and serve as a foundation for further experimentation.
