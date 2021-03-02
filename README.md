# CS 236: Music GAN with Chord-Conditioning

The goal of this project is to train a GAN that can generate music conditioned on chord sequences. Using this architecture, users would have more control over the model output than they would using traditional music generation models. Intuitively, the chord sequence input should also help to guide the generator outputs.

## Getting started

It's easiest to install the dependencies using Anaconda.

```
conda create -f environment.yml
```

The folder `lakh-pianoroll-dataset` is a clone from the ![https://github.com/salu133445/lakh-pianoroll-dataset](Lakh Pianoroll Dataset) project, and contains code for converting midi files to piano roll reprsentations. To do this, run

```
python converter.py path/to/pop909root path/to/dst
```

This has already been done. The results are stored in the `data` directory.


The key files are in the `notebooks` directory. The `data_eventbased.ipynb` notebook converts POP909 to an event-based representation, similar to MIDI. This representation was used during early stages of the project, but not anymore. `training.ipynb` loads the data from `data` and trains a DC-GAN.
