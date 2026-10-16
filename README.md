# DSA4213 Assignment 1: Language Models, Embeddings, and Representations

This repository has the code for my three experiments:

1. **Part I:** I train an interpolated trigram, an RNN, an LSTM and a Transformer language model on WikiText-2 and compare them.
2. **Part II:** I train the LSTM with three different input embeddings (random and trainable, my own word2vec frozen, GloVe frozen).
3. **Part III:** I use the trained models for AG News topic classification and compare the representations.

## Files

| File | What it contains |
|---|---|
| `part1_language_models.ipynb` | Data pipeline, n-gram, RNN, LSTM, Transformer, cost measurements, text generation |
| `part2_embeddings.ipynb` | word2vec training, GloVe loading and coverage, the two frozen-embedding LSTMs, comparison |
| `part3_downstream.ipynb` | AG News preprocessing, feature extraction, classifiers, error analysis |
| `results/` | Saved results: tables (CSV/JSON), plots, generated text, training histories and model checkpoints |
| `requirements.txt` | Python packages |
| `report.pdf` | Report |

The notebooks are saved with their outputs, so all the results, tables and plots can be seen without running anything.

Each notebook can run on its own. Parts II and III repeat the data pipeline and model class cells from Part I without changes, so all three use the same vocabulary and models.

## How I ran it

I ran everything on Google Colab (Python 3.13). Parts I and II used a Tesla T4 GPU. Part III feature extraction used a CPU runtime.

The first cell of each notebook checks where it is running:
- **In Colab**, it mounts Google Drive and saves results to `MyDrive/DSA4213 Assignment/results`.
- **Anywhere else**, it saves results to `./results`, so no Google Drive is needed.

To rerun in Colab, upload the notebooks, choose **Runtime → Change runtime type → T4 GPU**, and use **Runtime → Run all**. Run Part I first, then Part II, then Part III, because the later parts load the checkpoints saved by the earlier ones.

## Running locally in VS Code

1. Install Python 3.10 to 3.12, and the **Python** and **Jupyter** extensions in VS Code.
2. Open this folder in VS Code and create a virtual environment in the terminal:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```
3. Open a notebook, click **Select Kernel** (top right), and choose the `.venv` environment.
4. Click **Run All**. Run the notebooks in order: Part I, then II, then III.

Things to know before running locally:
- An internet connection is needed, because WikiText-2, AG News and GloVe (about 128 MB) are downloaded the first time.
- The notebooks reuse what is already saved in `results/`, so a local run is quick:
  - The model checkpoints are loaded instead of retrained, so no GPU is needed.
  - When the notebooks are not running in Colab, `REUSE_SAVED` is switched on. They then keep the timings from my Colab T4 run instead of timing the models again on a slower machine, and Part III loads the saved features from `part3_features.npz` instead of extracting them again.
  - The test scores, the n-gram, word2vec (about 1 minute), the plots and the generated text are still recomputed. This checks that the saved models load and give the same results.
- To measure everything again from scratch, set `REUSE_SAVED = False` in the first cell. If the checkpoints are deleted, the models are retrained, which is very slow on a CPU (several hours per model).
- Running a notebook rewrites the result files in `results/` (with the same numbers when `REUSE_SAVED` is on). To be safe, make a copy of the `results/` folder before running.
- The `%pip install` line at the top of each notebook installs any missing packages into the selected environment.

## Data

Nothing has to be downloaded by hand, because the notebooks fetch everything.

| Data | Source | Used in | License |
|---|---|---|---|
| WikiText-2 (raw) | Hugging Face `Salesforce/wikitext`, config `wikitext-2-raw-v1` | Parts I, II | CC BY-SA 3.0 |
| GloVe 6B 100d (Wikipedia 2014 + Gigaword 5) | `gensim.downloader.load("glove-wiki-gigaword-100")` | Part II | PDDL |
| AG News | Hugging Face `fancyzhx/ag_news` | Part III | Academic, non-commercial use |

I don't include any data files in this submission.

## Preprocessing

**WikiText-2 (Parts I and II)**
- I lowercase the text and skip empty lines.
- I split on spaces. The raw text already has punctuation separated and hyphens written as `@-@`.
- I add `<eos>` at the end of every line.
- I build the vocabulary from the training split only, keeping words that appear at least 3 times. This gives V = 29,572, and every other word becomes `<unk>`.
- The OOV rates are 2.25% (train), 4.72% (validation) and 5.62% (test).
- For the neural models, I lay out each split as 32 parallel streams of continuous text and cut it into 64-token windows. Each window is processed on its own and **no padding is used**.
- Every model is scored on the same tokenized test split. The n-gram scores every token from the 3rd one onwards. The neural models skip the 6 tokens that don't fill a full row and the first token of each row. That is fewer than 0.02% of the test tokens.

**AG News (Part III)**
- I sample 10,000 training and 2,000 validation examples from the AG News training set with seed 42, with no overlap between them. For testing, I use the full official test set of 7,600 examples.
- To clean the text, I:
  - unescape HTML
  - replace `#39;` with `'` and `#36;` with `$`
  - lowercase everything
  - write hyphens as `@-@`
  - split punctuation into separate tokens, to match WikiText-2
- I encode the texts with the WikiText-2 vocabulary (OOV rate 7.19%) and cut them to 64 tokens (6.8% of texts are cut).

## Settings

### n-gram
- Interpolated trigram: `P = λ1·P_uni + λ2·P_bi + λ3·P_tri`. The unigram uses add-one smoothing.
- I chose the λ values by grid search in steps of 0.05 on the validation set, which gave (0.30, 0.55, 0.15).

### Neural language models (same for all unless stated)

| Setting | Value |
|---|---|
| Embedding size | 100 (to match GloVe 100d) |
| Hidden size / model width | 256 |
| Layers | 2 |
| Dropout | 0.3 (embeddings, between layers, before the output layer) |
| Context length | 64 tokens, windows processed separately |
| Batch size | 32 |
| Optimizer | Adam, learning rate 1e-3 |
| Gradient clipping | 1.0 |
| Epochs | 10, keeping the checkpoint with the lowest validation loss |
| Tied input/output embeddings | No |
| RNN | `nn.RNN` (tanh) |
| LSTM | `nn.LSTM` |
| Transformer | Linear layer from 100 to 256, learned position embeddings, 4 heads, feed-forward size 512, pre-LayerNorm, causal mask |

### Part II embeddings
- **word2vec:** skip-gram, 100-d, window 5, 5 negative samples, `min_count=1`, 5 epochs, `workers=1`. I trained it on the WikiText-2 training split only, after mapping rare words to `<unk>`.
- **GloVe coverage:** 96.2% of the vocabulary and 95.1% of the training tokens.
- **Missing words:** words not found in GloVe get a random vector from N(μ, σ), using the mean and standard deviation of the vectors that were found.
- **Freezing:** in the frozen settings, the whole embedding table has `requires_grad=False`. The notebook checks that the table is unchanged after training.

### Part III classifier
- **Features:** the top-layer hidden states of each frozen LM, combined in one of two ways:
  - **Mean:** the average over the real tokens, with padding left out.
  - **Last:** the hidden state at the last real token.
- **Baseline (no LM):** the average of each text's input embedding vectors, for the learned (A), word2vec (B) and GloVe (C) embedding tables. I read these tables from the saved checkpoints.
- **Classifier:** `StandardScaler` + `LogisticRegression` (`max_iter=3000`), with C chosen from {0.01, 0.1, 1, 10} by validation accuracy.

`train_or_load` loads a finished checkpoint instead of retraining. I trained some models in an earlier Colab session, so those cells show "Loaded saved …". Their epoch-by-epoch training logs are printed from the saved `*_history.json` files.

## Random seeds

- I use `SEED = 42` for Python, NumPy and PyTorch.
- I call `set_seed()` before building each model, so Settings A, B and C in Part II start with the same LSTM and output layer weights.
- word2vec uses `seed=42` and `workers=1`, which makes it repeatable on the same setup. On a different machine or library version the vectors can differ very slightly (I measured a maximum difference of 0.002 between Colab and my laptop), so Part II saves the trained model to `results/word2vec.model` and reuses it in later runs.
- The AG News sample uses `np.random.default_rng(42)`.
- Text generation uses seed `42 + prompt number`, the same for every model.

GPU timings, and the last decimal places of the losses, can change a little between runs.

## Main results

| Model | Test PPL | Params | Train time (T4) |
|---|---|---|---|
| Interpolated trigram | 277.5 | 2.0M n-grams | 3.6 s |
| RNN | 200.5 | 10.8M | 411 s |
| LSTM (Setting A: random, trainable) | 178.2 | 11.5M | 478 s |
| Transformer | 177.7 | 11.7M | 435 s |
| LSTM (Setting B: word2vec, frozen) | 165.2 | 8.5M trainable | 443 s + 62 s for word2vec |
| LSTM (Setting C: GloVe, frozen) | 172.9 | 8.5M trainable | 445 s |

AG News test accuracy:
- Best language-model representation: 82.4% (LSTM-C with GloVe, mean pooling).
- Averaging the GloVe vectors directly, with no LM: 87.2%.

The full tables and discussion are in the report.

## Large files

The model checkpoints (about 44 MB each), the saved word2vec model and `results/part3_features.npz` (115 MB) can be recreated by running the notebooks. They are listed in `.gitignore` together with the local `.venv/` folder, so they aren't pushed to GitHub, which rejects files over 100 MB. If you submit a zip file instead, leave out `.venv/` and these large files.

## References and tool use

- **Libraries:** PyTorch, Hugging Face `datasets`, gensim, scikit-learn, SciPy, pandas, NumPy, matplotlib.
- **Datasets and embeddings:**
  - WikiText-2 (Merity et al., 2016)
  - GloVe (Pennington et al., 2014)
  - AG News (Zhang et al., 2015)
- **AI tools:** I used Claude (Anthropic) through Claude Code for step-by-step guidance on designing the experiments and for help writing the code, notebook notes and this README. I ran, checked and interpreted all of the code and results myself.
