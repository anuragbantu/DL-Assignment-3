# DL-Assignment-3
Name: Anurag Bantu, Roll no: MA24M003

# Telugu Transliteration (Vanilla Seq2Seq Model)

This project implements a character-level sequence-to-sequence (seq2seq) model to transliterate Latin script words into Telugu script. It uses a basic encoder-decoder architecture with optional hyperparameter tuning via Weights & Biases (wandb).

---

## Project Structure

```
.
├── data/
│   ├── te.translit.sampled.train.tsv
│   ├── te.translit.sampled.dev.tsv
│   ├── te.translit.sampled.test.tsv
├── predictions_vanilla/
│   └── test_predictions.tsv
```

---

## Requirements

Install required libraries via:

```bash
pip install torch pandas numpy matplotlib tabulate wandb
```

---

## Getting Started

### 1. Load Data

Prepare the following files inside a folder named `data/`:

- `te.translit.sampled.train.tsv`
- `te.translit.sampled.dev.tsv`
- `te.translit.sampled.test.tsv`

### 2. Vocabulary Preparation

The code constructs character-level vocabularies from the train set for both source (Latin) and target (Telugu) languages.

---

## Model Architecture

- **Encoder**: RNN/GRU/LSTM-based character embedding + hidden representation
- **Decoder**: RNN/GRU/LSTM-based decoder generating Telugu characters
- **Seq2Seq**: Combines encoder and decoder with teacher forcing

You can select different hyperparameters such as:
- `emb_dim`: Embedding dimension
- `hidden_dim`: Hidden state size
- `cell_type`: RNN cell (`RNN`, `GRU`, `LSTM`)
- `num_layers`: Number of RNN layers
- `dropout`: Dropout for regularization

---

## Training the Model

To launch training with wandb sweep:

```python
sweep_id = wandb.sweep(sweep_config, project="transliteration-sweep")
wandb.agent(sweep_id, function=train_model, count=100)
```

To train a best config directly:

```python
model = best_train_model(config=best_config)
```

---

## Evaluation

### Evaluate on Test Set

After training, the following metrics are computed:

- **Character Accuracy**: Percentage of correct character predictions
- **Exact Match**: Percentage of fully correct predicted words

---

## Predictions

### Save Predictions to File

The test predictions are stored at:

```
predictions_vanilla/test_predictions.tsv
```

### Display Samples in Tabular Form

Use the helper function `display_predictions()` to print predictions with ✓ or ✗ match indicators.

---

## Visualization

- Tabular output shows input, predicted, target, and match status.
- ANSI coloring is used for visual separation between correct and incorrect outputs in console.

---

## Notes

- This model does not include an attention mechanism.
- Works efficiently for character-level transliteration tasks with small datasets.
- Beam search is included to improve prediction quality.

---

## Acknowledgments

Data is sourced from the Dakshina dataset (by Google Research).
