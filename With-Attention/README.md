# DL-Assignment-3

Name: Anurag Bantu, Roll no: MA24M003

# Telugu Transliteration with Attention (Seq2Seq Model)

This project implements a character-level sequence-to-sequence (seq2seq) model with attention to transliterate Latin script to Telugu script. It supports training, hyperparameter tuning with Weights & Biases (wandb), and visualization of attention mechanisms.

## Project Structure

```
.
├── data/
│   ├── te.translit.sampled.train.tsv
│   ├── te.translit.sampled.dev.tsv
│   ├── te.translit.sampled.test.tsv
│   └── NotoSansTelugu-Regular.ttf
├── predictions_attention/
│   └── test_predictions.tsv

```

## Requirements

Ensure you have the following installed:

- Python ≥ 3.8
- PyTorch
- pandas, numpy, matplotlib, seaborn
- ipywidgets, kaleido
- wandb (for logging experiments)

Install dependencies via:

```bash
pip install torch pandas numpy matplotlib seaborn ipywidgets wandb kaleido
```

## Getting Started

### 1. Prepare the Data

Download the following files and place them in a `data/` directory:

- `te.translit.sampled.train.tsv`
- `te.translit.sampled.dev.tsv`
- `te.translit.sampled.test.tsv`
- `NotoSansTelugu-Regular.ttf` (for attention heatmap visualization)



## Training the Model

### Step 1: Login to WandB

```python
import wandb
wandb.login()
```

### Step 2: Define Hyperparameter Sweep

The sweep configuration (in `sweep_config`) includes variations of:

- Embedding dimensions
- Hidden size
- RNN cell type (`RNN`, `GRU`, `LSTM`)
- Dropout rate
- Learning rate
- Beam size for decoding

### Step 3: Run the Sweep

```python
sweep_id = wandb.sweep(sweep_config, project="transliteration-with-attn-sweep")
wandb.agent(sweep_id, function=train_model, count=2)
```

Each configuration is trained for 5 epochs. Validation accuracy and exact match scores are logged.

## Evaluating the Best Model

Once the best configuration is identified (e.g., via the WandB dashboard), update the `best_config` dictionary and run:

```python
model = best_train_model(config=best_config)
```

This will train the model again with the best configuration and return the trained model for evaluation.

## Testing and Saving Results

### Evaluate on Test Set:

```python
test_char_acc = test_correct / test_total
test_exact_match = exact_match_count / len(test_pairs)
```

### Save Predictions:

Results are written to: `predictions_attention/test_predictions.tsv`

## Visualizing Attention

### Option A: Static Heatmaps

- Grid of attention maps (`plot_attention_grid`)
- Single example (`plot_attention_map`)

### Option B: Interactive Attention Widget

Use the interactive widget to scroll through decoding steps and see character-level attention visually.

```python
interactive_attention_blocks(cache)
```

> Font required: `NotoSansTelugu-Regular.ttf` (ensure it's available in your environment)

## Metrics Tracked

- Character-level Accuracy: Compares predicted and actual characters
- Exact Match Accuracy: Measures whether the whole predicted word matches the true word
- Loss (Train/Val)
- Beam Search Output

## Notes


- Works well even with small embedding and hidden sizes due to character-level granularity.
- Visualization is essential to understand the attention alignment between source and target sequences.

## Acknowledgments

This project uses data sourced from the the dakshina dataset released by Google.
