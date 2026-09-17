# Audio experiment results

Recorded output from the completed notebook run; not a new evaluation.

Checkpoint: `meld_classifier_audio.pt`. The notebook currently specifies batch size 2,
10 epochs, learning rate 0.0005, unweighted cross-entropy, and a 200-second audio limit.
These settings describe the saved source; no complete runtime configuration was logged.

```text
Epoch 1/10 | loss: 1.2741 | validation accuracy: 56.59%
Epoch 2/10 | loss: 1.1588 | validation accuracy: 57.49%
Epoch 3/10 | loss: 1.1043 | validation accuracy: 59.30%
Epoch 4/10 | loss: 1.0697 | validation accuracy: 57.85%
Epoch 5/10 | loss: 1.0378 | validation accuracy: 57.85%
Epoch 6/10 | loss: 1.0050 | validation accuracy: 58.57%
Epoch 7/10 | loss: 0.9737 | validation accuracy: 58.21%
Epoch 8/10 | loss: 0.9441 | validation accuracy: 60.02%
Epoch 9/10 | loss: 0.9104 | validation accuracy: 58.21%
Epoch 10/10 | loss: 0.8782 | validation accuracy: 60.20%
Test accuracy: 63.26%
```

```text
Accuracy: 63.26%
Weighted F1: 59.99%
Macro F1: 39.49%

Per-class metrics:
              precision    recall  f1-score   support

       anger     0.5302    0.4319    0.4760       345
     disgust     0.2500    0.0147    0.0278        68
        fear     0.2593    0.1400    0.1818        50
         joy     0.5939    0.4876    0.5355       402
     neutral     0.6859    0.8973    0.7775      1256
     sadness     0.3398    0.1683    0.2251       208
    surprise     0.6126    0.4840    0.5408       281

    accuracy                         0.6326      2610
   macro avg     0.4674    0.3748    0.3949      2610
weighted avg     0.5962    0.6326    0.5999      2610
```
