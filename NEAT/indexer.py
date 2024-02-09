class Indexer:
    def __init__(self, next_id: int):
        self._next_id = next_id

    def next(self):
        self._next_id += 1
        return self._next_id
