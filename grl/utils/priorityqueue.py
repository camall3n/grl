import heapq

class PriorityQueue:
    """
    Adapted from code by GPT4
    """
    def __init__(self, highest_priority_first=True):
        self._queue = []
        self._index = 0
        self.highest_priority_first = highest_priority_first

    def is_empty(self):
        return not self._queue

    def push(self, item, priority):
        sign = -1 if self.highest_priority_first else 1
        heapq.heappush(self._queue, (sign * priority, self._index, item))
        self._index += 1

    def pop(self):
        if self.is_empty():
            raise Exception('Priority queue is empty')
        return heapq.heappop(self._queue)[-1]
