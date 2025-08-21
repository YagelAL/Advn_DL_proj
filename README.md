# Advanced Deep Learning Project: Sentiment Analysis of COVID-19 Tweets

![Project Overview](image.jpg)


## Project Overview
This repository provides a complete workflow for sentiment analysis on COVID-19 tweets using state-of-the-art transformer models. 
The project combines both manual and automated training pipelines, incorporates model compression techniques including distillation, and includes comprehensive exploratory data analysis (EDA). 
The main models utilized are RoBERTa and DeBERTa, with careful hyperparameter tuning and checkpoint management to optimize performance.
The workflow is designed to be modular, reproducible, and adaptable to similar text classification tasks.

## Notebooks

### 1. EDA.ipynb
- **Purpose:** Exploratory Data Analysis of the COVID-19 tweet dataset.
- **Key Steps:**
  - Loads and inspects raw data (`Corona_NLP_train.csv`, etc.)
  - Visualizes sentiment distribution and tweet length
  - Generates word clouds for each sentiment class
  - Cleans and preprocesses text (removes emojis, special characters)
  - Splits data for training and evaluation
- **Libraries Used:** pandas, numpy, matplotlib, seaborn, wordcloud, sklearn

### 2. train_manual.ipynb
- **Purpose:** Manual training and evaluation of sentiment models.
- **Key Steps:**
  - Loads cleaned and split datasets (`train_df.csv`, `eval_df.csv`, `test_df.csv`)
  - Defines label mappings and model configuration
  - Implements custom PyTorch training loop for RoBERTa/DeBERTa
  - Saves and loads model checkpoints
  - Evaluates model on test set and saves predictions to CSV
  - Hyperparameter tuning with Optuna
- **Outputs:**
  - Model checkpoints in `checkpoints/`
  - Test predictions and evaluation results in CSV files
- **Libraries Used:** torch, transformers, pandas, numpy, sklearn, optuna, wandb

### 3. train_HF.ipynb
- **Purpose:** Automated training and evaluation using HuggingFace Trainer API.
- **Key Steps:**
  - Loads datasets and configures HuggingFace models
  - Uses `Trainer` and `TrainingArguments` for training and evaluation
  - Implements callbacks for early stopping, logging, and checkpoint management
  - Hyperparameter optimization with Optuna
  - Evaluates best models and saves predictions
- **Outputs:**
  - Model checkpoints in `roberta_results_HF/` and `deberta_results_HF/`
  - Evaluation results in `model_evaluation_results_HF_*.csv`
- **Libraries Used:** transformers, torch, pandas, numpy, sklearn, optuna, wandb

### 4. sqeeze_models.ipynb
- **Purpose:** Model squeezing, distillation, and quantization for efficient inference.
- **Key Steps:**
  - Loads best checkpoints for RoBERTa and DeBERTa
  - Applies quantization and pruning to reduce model size
  - Compares performance of full and squeezed models
  - Loads and evaluates models on test data
- **Outputs:**
  - Squeezed/quantized model checkpoints
  - Performance metrics for comparison
- **Libraries Used:** torch, transformers, pandas, numpy, sklearn, matplotlib

## Data Files
- `Dataset/Corona_NLP_train.csv`, `Dataset/Corona_NLP_test.csv`: Raw tweet data
- `data/train_df.csv`, `data/eval_df.csv`, `data/test_df.csv`: Cleaned and split datasets

## Results & Outputs
## Experiment Logs

Weights & Biases logs are available in the `wandb/` folder on [Google Drive](YOUR_DRIVE_LINK_HERE).
To view experiment results, visit the public W&B project: [W&B Project Link](YOUR_WANDB_PROJECT_LINK)

**Note:** The W&B logs include several projects, covering different hyperparameter ranges and experiments run from multiple computers over time.


## Requirements
See `requirements.txt` for all required Python packages. Main dependencies:
- torch
- transformers
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- optuna
- wandb
- sentencepiece
- accelerate
- evaluate
- safetensors
- datasets

## How to Download and Use

### 1. Clone the Repository from GitHub
```bash
git clone <your-repo-url>
cd ADL
```

### 2.  Data Files from Google Drive
- Go to the shared Google Drive folder containing the data files.
- Download the following folders/files and place them in the project directory:
  - `Dataset/Corona_NLP_train.csv`
  - `Dataset/Corona_NLP_test.csv`
  also added the processed csv files to the following one:
  - `data/train_df.csv`
  - `data/eval_df.csv`
  - `data/test_df.csv`
  you can use in the notebooks with out the EDA.

### 3. Install Dependencies
the reqiured packages and verison are in the attached requierment file

### 4. Run the Notebooks
- Open the notebooks (`EDA.ipynb`, `train_manual.ipynb`, `train_HF.ipynb`, `sqeeze_models.ipynb`) in Jupyter or VS Code.
- Make sure the data files are in the correct folders as described above.
- Run the cells in order to reproduce the analysis, training, and evaluation.

## Notes
- Data files are not stored in the GitHub repository due to size . download them from the provided Google Drive link.
- If you encounter missing file errors, verify that all required data files are present in the correct locations.


