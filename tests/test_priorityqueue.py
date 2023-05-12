from grl.utils.priorityqueue import PriorityQueue

def test_priority_queue():
    pq = PriorityQueue()
    pq.push("item1", 3)
    pq.push("item2", 1)
    pq.push("item3", 2)
    pq.push("item4", 3)

    assert pq.pop() == "item1"
    assert pq.pop() == "item4"
    assert pq.pop() == "item3"
    assert pq.pop() == "item2"
