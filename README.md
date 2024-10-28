# UNSUPERVISED COMPOSABLE REPRESENTATIONS FOR AUDIO

Here we will provide the source code and supplementary material for our ISMIR 2024 submission: *Unsupervised Composable Representations For Audio*

Audio samples: [unsupervised_compositional_representations/](https://ismir-24-sub.github.io/unsupervised_compositional_representations/)

## Abstract

Current generative models are able to generate high-quality artefacts but have been shown to struggle with compositional reasoning, which can be defined as the ability to generate complex structures from simpler elements. In this paper, we focus on the problem of compositional representation learning for music data, specifically targeting the fully-unsupervised setting. We propose a simple and extensible framework that leverages an explicit compositional inductive bias, defined by a flexible auto-encoding objective that can leverage any of the current state-of-art generative models. We demonstrate that our framework, used with diffusion models, naturally addresses the task of unsupervised audio source separation, showing that our model is able to perform high-quality separation. Our findings reveal that our proposal achieves comparable or superior performance with respect to other blind source separation methods and, furthermore, it even surpasses current state-of-art supervised baselines on signal-to-interference ratio metrics. Additionally, by learning an a-posteriori masking diffusion model in the space of composable representations, we achieve a system capable of seamlessly performing unsupervised source separation, unconditional generation, and variation generation. Finally, as our proposal works in the latent space of pre-trained neural audio codecs, it also provides a lower computational cost with respect to other neural baselines.

## Code

### Pre-trained models

Pre-trained models will be made available soon!

### Dataset

You can pre-process your data with `scripts/prepare_dataset.py`. The script is tailored for Slakh2100 so you need to [download it](https://zenodo.org/records/4599666) beforehand.

You can run it from the root directory with

```[bash]
python -m scripts.prepare_dataset -t /path/to/slakh2100 -s [train, validation, test] -o /path/to/output -s [24000, 48000]
```

Then you will need to change the corresponding paths in the `config/slakh/...` gin files.

### Training

Let's assume you want to run the model on Slakh2100 with *drums* and *bass* stems. Inside `config/slakh/db` you will find the gin files for the training of the decomposition and recomposition model.

You will need to adjust the `lmdb_path` in the, e.g., `config/slakh/db/encodec_iadb.gin` with your pre-processed data directory. You can then change your hyper-parameters and launch the training with

```[bash]
python -m scripts.train_iadb -c config/slakh/db/encodec_iadb.gin -v c -e 500 -d cuda:0
```

Similarly, you can train the recomposition model with

```[bash]
python -m scripts.train_iadb_masking -c config/slakh/db/encodec_prior_iadb.gin -v c -e 500 -d cuda:0 -p /path/to/your/pre/trained/decomposition/model (trained with the previous command)
```

### Inference

There's a simple inference notebook for the decomposition model in the `notebooks` directory. More detailed instructions will be provided with the pre-trained models soon!