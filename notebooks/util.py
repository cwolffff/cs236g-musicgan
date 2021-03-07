from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pretty_midi as pyd
import torch


@dataclass
class Chord:
    chord: str
    start_time: float
    end_time: float


@dataclass
class Event:
    token: int
    chord: Chord


def get_chord(chords, t):
    """Get the chord that overlaps with time t."""
    for chord in chords:
        if chord.start_time <= t and chord.end_time >= t:
            return chord
    return None


def _merge_1d(arrays, dtype=torch.int64, pad_value=0):
    lengths = [(a != pad_value).sum() for a in arrays]
    padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
    for i, seq in enumerate(arrays):
        end = lengths[i]
        padded[i, :end] = seq[:end]
    return padded


def collate_fn(examples):
    """Create batch tensors from a list of individual examples. Merge examples
    of different length by padding all examples to the maximum length in the batch.

    Args:
        examples (list): List of tuples of the form (notes, chords).

    Returns:
        examples (tuple): Tuple of tensors (notes, chords). All of shape (batch_size, ...).

    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    notes = [ex["notes"] for ex in examples]
    chords = [ex["chords"] for ex in examples]

    # Merge into batch tensors
    notes = torch.stack(notes)
    chords = _merge_1d(chords)

    return notes, chords


class ReprProcessor(ABC):
    """Abstract base class severing as the representation processor.

    It provides the following abstract methods.
    - encode(self, note_seq): encode the note sequence into the representation sequence.
    - decode(self, repr_seq): decode the representation sequence into the note sequence.

    Notes
    -----
    The base representation processor class includes the convertion between the note sequence and the representation sequence.
    In general, we assume the input note sequence has already been quantized.
    In that, the smallest unit of the quantization is actually 1 tick no matter what resolution is.
    If you init "min_step" to be larger than 1, we assume you wish to compress all the base tick.
    e.g. min_step = 2, then the whole ticks will be convertd half.
    If you do this, the representation convertion may not be 100% correct.
    -----
    
    Based largely on https://github.com/music-x-lab/POP909-Dataset/blob/master/data_process/processor.py,
    with minor modifications.

    """

    def __init__(self, min_step: int = 1):
        self.min_step = min_step

    def _compress(self, note_seq=None):
        """Return the compressed note_seq based on the min_step > 1.

        Parameters
        ----------
        note_seq : Note Array.
        ----------

        WARNING: If you do this, the representation convertion may not be 100% correct.

        """
        new_note_seq = [
            Note(
                start=int(d.start / self.min_step),
                end=int(d.end / self.min_step),
                pitch=d.pitch,
                velocity=d.velocity,
            )
            for d in note_seq
        ]
        return new_note_seq

    def _expand(self, note_seq=None):
        """Return the expanded note_seq based on the min_step > 1.

        Parameters
        ----------
        note_seq : Note Array.
        ----------

        WARNING: If you do this, the representation convertion may not be 100% correct.

        """
        new_note_seq = [
            Note(
                start=int(d.start * self.min_step),
                end=int(d.end * self.min_step),
                pitch=d.pitch,
                velocity=d.velocity,
            )
            for d in note_seq
        ]
        return new_note_seq

    @abstractmethod
    def encode(self, note_seq, chords, encode_velocity=False):
        """encode the note sequence into the representation sequence.

        Parameters
        ----------
        note_seq= the input {Note} sequence

        Returns
        ----------
        repr_seq: the representation numpy sequence

        """

    @abstractmethod
    def decode(self, repr_seq=None):
        """decode the representation sequence into the note sequence.

        Parameters
        ----------
        repr_seq: the representation numpy sequence

        Returns
        ----------
        note_seq= the input {Note} sequence

        """


class MidiEventProcessor(ReprProcessor):
    """Midi Event Representation Processor.

    Representation Format:
    -----
    Size: L * D:
        - L for the sequence (event) length
        - D = 1 {
            0-127: note-on event,
            128-255: note-off event,
            256-355(default):
                tick-shift event
                256 for one tick, 355 for 100 ticks
                the maximum number of tick-shift can be specified
            356-388 (default):
                velocity event
                the maximum number of quantized velocity can be specified
            }

    Parameters:
    -----
    min_step(optional):
        minimum quantification step
        decide how many ticks to be the basic unit (default = 1)
    tick_dim(optional):
        tick-shift event dimensions
        the maximum number of tick-shift (default = 100)
    velocity_dim(optional):
        velocity event dimensions
        the maximum number of quantized velocity (default = 32, max = 128)

    e.g.

    [C5 - - - E5 - - / G5 - - / /]
    ->
    [380, 60, 259, 188, 64, 258, 192, 256, 67, 258, 195, 257]

    """

    def __init__(self, **kwargs):
        self.name = "midievent"
        min_step = 1
        if "min_step" in kwargs:
            min_step = kwargs["min_step"]
        super().__init__(min_step)
        self.tick_dim = 100
        self.velocity_dim = 32
        if "tick_dim" in kwargs:
            self.tick_dim = kwargs["tick_dim"]
        if "velocity_dim" in kwargs:
            self.velocity_dim = kwargs["velocity_dim"]
        if self.velocity_dim > 128:
            raise ValueError(
                "velocity_dim cannot be larger than 128", self.velocity_dim
            )
        self.max_vocab = 256 + self.tick_dim + self.velocity_dim
        self.start_index = {
            "note_on": 0,
            "note_off": 128,
            "time_shift": 256,
            "velocity": 256 + self.tick_dim,
        }

    def encode(self, note_seq, chords, encode_velocity=False):
        """Return the note token

        Parameters
        ----------
        note_seq : Note list.
        chords: Chord list.

        Returns
        ----------
        repr_seq: Representation List

        """
        if note_seq is None:
            return []
        if self.min_step > 1:
            note_seq = self._compress(note_seq)
        notes = note_seq
        events = []
        meta_events = []
        for note in notes:
            token_on = {
                "name": "note_on",
                "time": note.start,
                "pitch": note.pitch,
                "vel": note.velocity,
            }
            token_off = {
                "name": "note_off",
                "time": note.end,
                "pitch": note.pitch,
                "vel": None,
            }
            meta_events.extend([token_on, token_off])
        meta_events.sort(key=lambda x: x["pitch"])    
        meta_events.sort(key=lambda x: x["time"])
        time_shift = 0
        cur_vel = 0
        for me in meta_events:
            duration = int((me["time"] - time_shift) * 100)
            while duration >= self.tick_dim:
                e = Event(
                    token=self.start_index["time_shift"] + self.tick_dim - 1,
                    chord=get_chord(chords, me["time"]),
                )
                events.append(e)
                duration -= self.tick_dim
            if duration > 0:
                e = Event(
                    token=self.start_index["time_shift"] + duration - 1,
                    chord=get_chord(chords, me["time"]),
                )
                events.append(e)
            if encode_velocity and me["vel"] is not None:
                if cur_vel != me["vel"]:
                    cur_vel = me["vel"]
                    e = Event(
                        token=self.start_index["velocity"] + int(round(me["vel"] * self.velocity_dim / 128)),
                        chord=get_chord(chords, me["time"]),
                    )
                    events.append(e)
            e = Event(
                token=self.start_index[me["name"]] + me["pitch"],
                chord=get_chord(chords, me["time"])
            )
            events.append(e)
            time_shift = me["time"]
        return events

    def decode(self, repr_seq=None):
        """Return the note seq

        Parameters
        ----------
        repr_seq: Representation Sequence List

        Returns
        ----------
        note_seq : Note List.

        """
        if repr_seq is None:
            return []
        time_shift = 0.0
        cur_vel = 0
        meta_events = []
        note_on_dict = {}
        notes = []
        for e in repr_seq:
            if self.start_index["note_on"] <= e < self.start_index["note_off"]:
                token_on = {
                    "name": "note_on",
                    "time": time_shift,
                    "pitch": e,
                    "vel": cur_vel,
                }
                meta_events.append(token_on)
            if (
                self.start_index["note_off"]
                <= e
                < self.start_index["time_shift"]
            ):
                token_off = {
                    "name": "note_off",
                    "time": time_shift,
                    "pitch": e - self.start_index["note_off"],
                    "vel": cur_vel,
                }
                meta_events.append(token_off)
            if (
                self.start_index["time_shift"]
                <= e
                < self.start_index["velocity"]
            ):
                time_shift += (e - self.start_index["time_shift"] + 1) * 0.01
            if self.start_index["velocity"] <= e < self.max_vocab:
                cur_vel = int(round(
                    (e - self.start_index["velocity"])
                    * 128
                    / self.velocity_dim)
                )
        skip_notes = []
        for me in meta_events:
            if me["name"] == "note_on":
                note_on_dict[me["pitch"]] = me
            elif me["name"] == "note_off":
                try:
                    token_on = note_on_dict[me["pitch"]]
                    token_off = me
                    if token_on["time"] == token_off["time"]:
                        continue
                    notes.append(
                        pyd.Note(
                            velocity=token_on["vel"],
                            pitch=int(token_on["pitch"]),
                            start=token_on["time"],
                            end=token_off["time"],
                        )
                    )
                except:
                    skip_notes.append(me)
        notes.sort(key=lambda x: x.start)
        if self.min_step > 1:
            notes = self._expand(notes)
        return notes

