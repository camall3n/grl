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

    def __bool__(self):
        return not self.is_empty()

    def push(self, item, priority):
        sign = -1 if self.highest_priority_first else 1
        heapq.heappush(self._queue, (sign * priority, self._index, item))
        self._index += 1

    def pop(self, return_priority=False):
        if self.is_empty():
            raise Exception('Priority queue is empty')
        signed_priority, _, item = heapq.heappop(self._queue)
        if return_priority:
            sign = -1 if self.highest_priority_first else 1
            priority = sign * signed_priority
            result = (item, priority)
        else:
            result = item
        return result
