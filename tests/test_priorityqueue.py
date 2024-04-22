from grl.utils.priorityqueue import PriorityQueue
from grl.utils.discrete_search import SearchNode, generate_hold_mem_fn

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

def test_search_node_pq():
    pq = PriorityQueue()
    HOLD = generate_hold_mem_fn(4, 3, 2)
    nodes = [None] * 4
    nodes[0] = SearchNode(HOLD)
    nodes[1] = nodes[0].get_successors()[0]
    nodes[2] = nodes[0].get_successors()[1]
    nodes[3] = nodes[0].get_successors()[2]

    for i in range(4):
        nodes[i].str = f'item{i}'

    pq.push(nodes[0], 3)
    pq.push(nodes[1], 1)
    pq.push(nodes[2], 2)
    pq.push(nodes[3], 3)

    assert pq.pop().str == "item0"
    assert pq.pop().str == "item3"
    assert pq.pop().str == "item2"
    assert pq.pop().str == "item1"
