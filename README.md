# Real-Time Multimodal Emotion-Aware Response Prototype

I implements the **Text + Vision** track of the ML challenge. It
accepts an utterance transcript and a matching MP4 video clip, combines both
modalities, predicts one of the seven MELD emotion categories, and returns a
short response conditioned on the predicted emotional state.

The seven supported emotions are `anger`, `disgust`, `fear`, `joy`, `neutral`,
`sadness`, and `surprise`.

I also implements and optional version of Text + Vision + Audio. The audio model takes in and audio file, combined with other input mentioned above and output one of the seven emotionals states.

## Completed system

The prototype provides an end-to-end local inference path:

1. The user enters the spoken text, the path to its matching video clip and the optional path to the audio file.
2. DeBERTa encodes the text, VideoMAE encodes sampled video frames and WavLM encodes the audio.
3. The three embeddings are concatenated and passed through an MLP classifier.
4. The classifier returns a MELD emotion and softmax confidence score.
5. A lightweight response policy selects a short response associated with the
   predicted emotion.
6. The widget displays the emotion, confidence, response, and inference latency.

The widget can be reused for successive utterances, providing input and output
over time without reloading the models.

## Architecture

- **Text encoder:** `microsoft/deberta-v3-small`
- **Vision encoder:** `MCG-NJU/videomae-base`
- **Audio encoder:** `microsoft/wavlm-base`
- **Fusion:** mean-pooled text, video, and audio embeddings are concatenated
- **Classifier:** two-layer MLP with ReLU and dropout
- **Response generation:** emotion-conditioned response templates from
  `responses.json`

All pretrained encoders is frozen during training. Only the fusion classifier
is trained. This keeps training relatively inexpensive and makes the system
easier to understand, but limits task-specific adaptation of the encoders.

The selected encoders provide a practical balance between representational
capacity and the cost of local inference.

## Parameter count

The parameter limit applies to every learned component in the complete local
inference path, including frozen parameters.

| Component | Parameters |
| --- | ---: |
| DeBERTa-v3-small | 141,304,320 |
| VideoMAE-base | 86,227,200 |
| WavLM | 94,381,936 |
| Fusion classifier | 790,535 |
| **Total** | **322,703,991** |

The complete system contains approximately **228.3 million parameters**, well
below the challenge limit of 6 billion.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name hucomintern --display-name "HuComIntern"
python -m jupyterlab
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Alternatively, open `demo.ipynb` or `model.ipynb`, run the first
`%pip install -r requirements.txt` cell, and restart the notebook kernel if any
packages were installed or changed.

The first model load downloads the pretrained DeBERTa and VideoMAE weights from
Hugging Face. All subsequent inference is performed locally; no remote inference
API is used.

## Run the interactive demo

1. Open `demo.ipynb` from the repository root.
2. Select the environment containing the installed requirements.
3. Run the notebook cells in order.
4. Enter a transcript and the path to its matching local MP4 file.
5. Click **Predict emotion**.
6. Reuse the widget for additional transcript/video pairs as needed.

The included example can be run with:

- **Video:** `test1.mp4`
- **Transcript:** `Oh no!`
- **Demo clip source/redistribution status:** `TBD - fill in before submission`

The precise confidence, selected response, and latency may vary. A successful
prediction returns the following structure:

```json
{
  "emotion": "surprise",
  "confidence": 0.62,
  "response": "That sounds unexpected! What happened next?",
  "latency": 1.28
}
```

Softmax confidence is the model's relative class score and is not a calibrated
probability. Low-confidence predictions are reported as uncertain.

## Definition of real-time

This project defines real-time as **utterance-level interactive inference**. The
system receives a completed transcript and its corresponding short video clip,
then returns an emotion and response quickly enough to support conversational
turn-taking. It is not continuous frame-by-frame streaming.

Model download and initial model loading are treated as startup costs and are
reported separately from warm inference latency.

### Observed latency

Measurements use batch size 1 on the local demo hardware after warm-up.

| Measurement | Result |
| --- | ---: |
| Number of measured runs | 10 |
| Mean inference latency | 1.208 ms |
| Median inference latency | 1.197 ms |

## Hardware and observed resources

### Training and evaluation

- GPU: NVIDIA A40
- Peak GPU memory: 51 GB (for the optional audio model version)
- Training time: ~ 6 hours

### Local interactive demo

- Computer: MacBook Air
- Processor: Apple M4, 10-core CPU
- Memory: 16 GB unified memory
- PyTorch device: MPS
- Downloaded pretrained-model storage: TBD GB
- Classifier checkpoint size: approximately 3 MB

The demo automatically selects CUDA when available, otherwise Apple MPS, and
otherwise CPU. Latency and memory requirements will vary by device.

## MELD data

The repository includes the MELD CSV annotations but not the full video dataset.
Obtain and extract the MELD media from the official dataset distribution:

<https://github.com/declare-lab/MELD>

By default, `model.ipynb` expects:

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

If the extracted split directories use different names, update `video_dirs` in
`model.ipynb`. Preserve the original
`dia<Dialogue_ID>_utt<Utterance_ID>.mp4` filenames.

## Training and evaluation

`model.ipynb` contains dataset loading, model training, checkpoint saving, and
test evaluation. A CUDA GPU is strongly recommended for training. The included
checkpoints contain the trained classifier head; the frozen pretrained encoders
are loaded separately during inference.

Two classifier configurations were evaluated:

| Model | Accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: |
| Baseline cross-entropy | **62.84%** | **60.14%** | 39.61% |
| Square-root class-weighted cross-entropy | 59.08% | 59.25% | **41.40%** |
| Optional Audio + Visual + Text (equally weighted cross entropy) 63.26% | 59.99% | 39.39% | 



### Baseline versus class weighting

The weighted experiment used the square root of inverse-frequency class weights:

```text
anger: 1.134
disgust: 2.295
fear: 2.307
joy: 0.905
neutral: 0.550
sadness: 1.445
surprise: 1.088
```

Weighting improved macro F1 from 39.61% to 41.40% and improved several minority
classes, including anger, disgust, fear, and sadness. However, it reduced overall
accuracy from 62.84% to 59.08% and weighted F1 from 60.14% to 59.25%. The
baseline checkpoint is therefore used for the primary interactive demo, while
the weighted checkpoint is retained to demonstrate the class-balance trade-off.


## Limitations

- Overall test accuracy is 62.84%, so the system is a prototype rather than a
  production-ready emotion detector.
- Fear and disgust have limited training and test support and substantially lower
  per-class recall than the more common emotions.
- The encoders are frozen and are not fine-tuned for MELD emotion recognition.
- VideoMAE was pretrained for general video understanding rather than facial
  emotion recognition.
- The model requires a matching text transcript and does not perform speech
  recognition or use audio.
- Prediction operates after a complete utterance clip is available rather than
  continuously streaming frames.
- Softmax scores are not calibrated confidence estimates.
- Template responses are intentionally limited and may not capture the full
  meaning or context of an utterance.

## Completed and intentionally omitted

### Completed

- Text-and-vision multimodal classification
- Seven-category MELD emotion output
- Structured emotion, confidence, response, and latency output
- Local interactive notebook widget
- Baseline and class-weighted evaluation
- Local classifier checkpoints

### Intentionally omitted

- Audio input and automatic speech recognition
- Three-modality text, audio, and vision fusion
- Continuous streaming inference
- LLM-based response generation
- Reinforcement learning

These features were omitted to keep the prototype focused and understandable
within the challenge timebox.

## External components and AI-assisted development

Important external components include:

- The MELD dataset and annotations
- The pretrained `microsoft/deberta-v3-small` text encoder
- The pretrained `MCG-NJU/videomae-base` video encoder
- PyTorch, Hugging Face Transformers, OpenCV, NumPy, pandas, scikit-learn, and
  Jupyter

Codex was used as an AI-assisted development tool to help draft parts of the
evaluation loop, resolve dependency issues, review the prototype, and assist in
drafting this README. The repository owner remains responsible for the code,
experiments, reported results, and technical decisions, and should be able to
explain and modify the implementation.
