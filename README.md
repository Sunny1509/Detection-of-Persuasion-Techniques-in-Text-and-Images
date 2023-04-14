# Detection-of-Persuasion-Techniques-in-Text-and-Images

## 1. Task ##

The task is to identify which techniques out of 22 are used in the meme considering both the text and image. The problem lies under the category of multi-class calssification problem.

## 2. Load Data ##

For this task, DataModule is used to prepare the data. It is used for consistent data splits, data preparation and transformation across the model.
A LightningModule organizes  PyTorch code into 6 sections:

  1. Computations (init).
  2. Train Loop (training_step)
  3. Validation Loop (validation_step)
  4. Test Loop (test_step)
  5. Prediction Loop (predict_step)
  6. Optimizers and LR Schedulers (configure_optimizers)
Lightning handles the distributed sampler for you by default. It helps in improving code readibility and performance.



## 3. Model Architecture  ##

A BERT-like architecture known as supervised MultiModal BiTransformers (MMBT) is built on unimodally pretrained text transformers and picture encoders that are then refined in a combined modality by projecting the visual embeddings onto the text token space A transformer model is used in multimodal settings to combine text and image to make predictions.The final activations of a pretrained on image resnet (after the pooling layer) that travels via a linear layer and the embeddings of the tokenized text are supplied into the transformer model (to go from number of features at the end of the resnet to the hidden state dimension of the transformer). The various inputs are integrated, and to inform the model which portion of the input vector pertains to the text and which to the image, a segment embedding is added on top of the positional embeddings. It helps in the classification task.

There are various steps involved in the process:

1. We first train the randomly initialised components
2. Image encoder is trained using pretrained resnet50
3. Fine tuning is performed

## 4. Evaluation Metrics ##

We'll evaluate our performance using the metrics Micro and Macro F1 Scores. The reason for choosing this is that our problem is multi-label classification.

