# Data Preprocessing

This folder contains scripts for preprocessing and analyzing the raw dataset for the project. The scripts include:

- **data_preprocessing.py**  
  - Loads a CSV file (`./dataset/train.csv`)
  - Normalizes the context, question, and answer texts (e.g., removing extra symbols and Han characters)
  - Appends `<|end_of_text|>` to each answer
  - Saves the processed data as a JSON file (`train_base.json`)

- **data_analysis.py**  
  - Loads the CSV dataset and computes various statistics:
    - Word count frequency for answers
    - Counts of how many times an answer appears in its context
    - Sentence splitting and distribution analysis
    - Context length distribution based on token counts
  - Outputs analysis results to the console and saves some results to JSON

- **similarity_DP.py**  
  - Uses SBERT and Jaccard similarity to evaluate the relevance of sentences within the context
  - Splits contexts into sentences and calculates similarity scores between the question and each sentence
  - Filters and selects the top 10 most relevant sentences
  - Saves the refined data as a JSON file (`train_preprocessed4.json`)

## How to Run

1. **Preprocess the Data:**
    ```bash
    python data_preprocessing.py
    ```

2.	**Analyze the Data:**
    ```bash
    python data_analysis.py
    ```

Each script assumes the CSV file is located in the ./dataset/ folder. Adjust paths as needed.