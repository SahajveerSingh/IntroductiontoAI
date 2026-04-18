# CUS2 — Iterative Deepening A* (IDA*)

### COS30019 Introduction to AI — Assignment 2A

---

## What is IDA\*?

IDA* (Iterative Deepening A*) is an **informed search algorithm** that combines the memory efficiency of Depth-First Search with the optimality of A*. Instead of storing all nodes in a priority queue like A*, IDA\* performs repeated depth-limited searches using an **f-cost threshold** (f = g + h), gradually increasing the threshold each iteration until the goal is found.

- **g(n)** — actual cost from the origin to the current node
- **h(n)** — heuristic estimate (Euclidean distance to nearest destination)
- **f(n)** — total estimated cost = g(n) + h(n)

---

## Files

| File                                       | Description                                 |
| ------------------------------------------ | ------------------------------------------- |
| `search_cus2.py`                           | Main IDA\* search program                   |
| `PathFinder-test.txt`                      | Sample test case (provided with assignment) |
| `TC01_sample.txt` ... `TC16_dead_ends.txt` | 16 test case files                          |

---

## Requirements

- Python 3.x (no external libraries needed)
- Works on Windows 10/11 via Command Prompt

---

## How to Run

Open a terminal / Command Prompt in the folder containing `search_cus2.py` and your problem file, then run:

```
python search_cus2.py <filename> CUS2
```

**Example:**

```
python search_cus2.py PathFinder-test.txt CUS2
```

---

## Input File Format

The problem file must follow this exact format:

```
Nodes:
1: (4,1)
2: (2,2)
3: (4,4)

Edges:
(2,1): 4
(1,3): 5

Origin:
2

Destinations:
5; 4
```

| Section         | Description                             |
| --------------- | --------------------------------------- |
| `Nodes:`        | Each node ID and its (x, y) coordinates |
| `Edges:`        | Directed edges as `(from, to): cost`    |
| `Origin:`       | The starting node                       |
| `Destinations:` | One or more goal nodes separated by `;` |

---

## Output Format

When a solution is found:

```
filename CUS2
goal number_of_nodes
[path]
```

When no solution exists:

```
filename CUS2
No solution found.
```

**Example output:**

```
PathFinder-test.txt CUS2
4 18
[2, 1, 4]
```

| Line   | Meaning                                              |
| ------ | ---------------------------------------------------- |
| Line 1 | Filename and method name                             |
| Line 2 | Goal node reached, total nodes created during search |
| Line 3 | Path taken from origin to goal as an array           |

---

## How the Algorithm Works

1. **Start** with an initial f-cost threshold = h(origin)
2. **Perform DFS** from the origin, pruning any node whose f-cost exceeds the threshold
3. **Track** the minimum f-cost among all pruned nodes
4. If the **goal is found** → return the path
5. If **no goal found** → raise the threshold to the minimum pruned f-cost and repeat from step 2
6. If the threshold reaches **infinity** → no solution exists

### Tie-Breaking Rules

When multiple neighbours have equal priority, the algorithm expands them in **ascending node ID order** (e.g. node 2 before node 4), matching the assignment specification.

### Cycle Prevention

A `visited` path list tracks nodes on the **current branch** to prevent revisiting the same node in a single path, avoiding infinite loops on cyclic graphs.

---

## Why IDA\* finds the Optimal Path

IDA* is **complete** and **optimal** when the heuristic is admissible (never overestimates). Euclidean distance never overestimates the actual path cost, so IDA* is guaranteed to find the **lowest cost path** to the goal — unlike GBFS which can be misled by the heuristic alone.

---

## IDA\* vs Other Algorithms

| Property              | GBFS            | A\*             | IDA\* (CUS2)             |
| --------------------- | --------------- | --------------- | ------------------------ |
| Uses heuristic h(n)   | ✅              | ✅              | ✅                       |
| Uses actual cost g(n) | ❌              | ✅              | ✅                       |
| Finds optimal path    | ❌              | ✅              | ✅                       |
| Memory usage          | High            | High            | **Very low**             |
| Uses priority queue   | ✅              | ✅              | ❌                       |
| Data structure        | Heap + frontier | Heap + frontier | **Recursion stack only** |

---

## Running All 16 Test Cases

```
python search_cus2.py TC01_sample.txt CUS2
python search_cus2.py TC02_origin_is_goal.txt CUS2
python search_cus2.py TC03_direct_edge.txt CUS2
python search_cus2.py TC04_no_path.txt CUS2
python search_cus2.py TC05_linear_chain.txt CUS2
python search_cus2.py TC06_multi_dest_nearest.txt CUS2
python search_cus2.py TC07_directed_only.txt CUS2
python search_cus2.py TC08_greedy_suboptimal.txt CUS2
python search_cus2.py TC09_tiebreak_nodeid.txt CUS2
python search_cus2.py TC10_large_graph.txt CUS2
python search_cus2.py TC11_cycle.txt CUS2
python search_cus2.py TC12_single_node.txt CUS2
python search_cus2.py TC13_one_reachable_dest.txt CUS2
python search_cus2.py TC14_dense_graph.txt CUS2
python search_cus2.py TC15_forced_detour.txt CUS2
python search_cus2.py TC16_dead_ends.txt CUS2
```

---

## Known Limitations

- IDA* may create significantly more nodes than A* on complex graphs due to re-exploration across iterations (e.g. TC10 creates 51 nodes vs A\*'s fewer). This is the trade-off for using almost no memory.
- Very large graphs with many nodes and edges may be slower than A\* due to repeated DFS iterations.

---

## Authors

Sahajveer Pratap Singh Bhatia
COS30019 — Introduction to AI, Assignment 2A  
Swinburne University of Technology
