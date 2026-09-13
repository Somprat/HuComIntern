# HuComIntern
ML model for classifying emotions based on texts and videos.

## Run the notebook

Use Python 3.10 or newer. From this repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name hucomintern --display-name "HuComIntern"
python -m jupyterlab model.ipynb
```

On Windows, activate with `.venv\Scripts\activate` instead. Select the
**HuComIntern** kernel in Jupyter or VS Code, then run the cells in order.
If installing from an existing notebook, use `%pip install -r requirements.txt`
and restart the kernel afterward.

## Required data

The repository includes MELD CSV annotations, but does not include video clips.
Obtain and extract the MELD videos from the dataset distribution:
https://github.com/declare-lab/MELD

By default the notebook expects:

```text
datasets/MELD/
  train_sent_emo.csv
  dev_sent_emo.csv
  test_sent_emo.csv
  train/dia0_utt0.mp4
  dev/dia0_utt0.mp4
  test/dia0_utt0.mp4
  ...
```

Edit `data_dir` in the notebook for a different dataset root. Edit `video_dirs`
to point to the actual extracted split folders (which may have
names such as `train_splits`, `dev_splits_complete`, or `output_repeated_splits_test`).
Keep the original `dia<Dialogue_ID>_utt<Utterance_ID>.mp4` filenames.
The notebook checks for missing clips before downloading the models.

The first model load requires internet access and disk space for the DeBERTa
and VideoMAE weights. Training runs for 10 epochs; a CUDA GPU is strongly
recommended. Reduce the DataLoader batch sizes if device memory is insufficient.
The weighted training cell saves `meld_classifier_weighted.pt`. The repository
also includes `meld_classifier_baseline.pt`; both files contain only the trained
classifier head, so the same pretrained encoders are loaded when evaluating.

## Recorded test results

| Checkpoint | Accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: |
| Baseline | 62.84% | 60.14% | 39.61% |
| Square-root class weights | 59.08% | 59.25% | 41.40% |

The baseline is the primary checkpoint for overall accuracy and weighted F1.
The weighted checkpoint trades some overall performance for better balance
across minority classes.
