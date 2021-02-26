import pickle
import torch


class MusicDataset(torch.utils.data.Dataset):
    """
    A dataset of chunks from songs. Each chunk consists of a sequence of
    4 chords along with a sequence of notes played with those chords.

    The chords and the notes are both tokenized already, and converted to
    tensors when __getitem__ is called.
    """

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

        data = pickle.load(open(data_path, "rb"))
        self._chords = [torch.LongTensor(d.chords) for d in data]
        self._events = [torch.LongTensor(d.events) for d in data]

    def __len__(self):
        return len(self._chords)

    def __getitem__(self, idx):
        return (
            self._chords[idx],
            self._events[idx],
        )