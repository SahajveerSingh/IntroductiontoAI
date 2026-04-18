# To parse files

python gbfs.py PathFinder-test.txt GBFS

# Test Report

**Program:** `All programs`  
**Searches:** `Searches`  
**Total Test Cases:** 16  
**All Passed:** ✅

---

## Test Case Summary

| #    | Test Case             | Scenario                                   | Origin | Destinations | Goal Found | Path      | Nodes Created |
| ---- | --------------------- | ------------------------------------------ | ------ | ------------ | ---------- | --------- | ------------- |
| TC01 | Sample (baseline)     | Original assignment sample                 | 2      | 5; 4         | 5          | 2→3→5     | 6             |
| TC02 | Origin is goal        | Origin node is already a destination       | 1      | 1            | 1          | 1         | 1             |
| TC03 | Direct edge           | Only one node between origin and goal      | 1      | 2            | 2          | 1→2       | 2             |
| TC04 | No path               | Disconnected graph, goal unreachable       | 1      | 3            | None       | —         | 2             |
| TC05 | Linear chain          | Nodes in a straight line                   | 1      | 5            | 5          | 1→2→3→4→5 | 5             |
| TC06 | Multiple destinations | Two goals, picks nearest                   | 1      | 2; 3         | 2          | 1→2       | 3             |
| TC07 | Directed only         | All edges are one-way                      | 1      | 4            | 4          | 1→2→3→4   | 4             |
| TC08 | Greedy suboptimal     | Heuristic misleads to longer path          | 1      | 3            | 3          | 1→4→3     | 4             |
| TC09 | Tie-breaking          | Equal heuristic, smaller node ID wins      | 1      | 4            | 4          | 1→4       | 4             |
| TC10 | Large graph           | 10-node graph, many paths                  | 1      | 10           | 10         | 1→6→7→10  | 5             |
| TC11 | Cycle in graph        | Cycle present, no infinite loop            | 1      | 5            | 5          | 1→2→3→5   | 5             |
| TC12 | Single node           | Only one node, it is the goal              | 1      | 1            | 1          | 1         | 1             |
| TC13 | One reachable dest    | Three destinations, only one reachable     | 1      | 3; 4; 5      | 4          | 1→2→4     | 3             |
| TC14 | Dense graph           | Many edges between nodes                   | 1      | 5            | 5          | 1→4→5     | 5             |
| TC15 | Forced detour         | Direct path blocked by edge direction      | 1      | 2            | 2          | 1→4→3→2   | 5             |
| TC16 | Dead ends             | Origin neighbours are dead ends except one | 1      | 6            | 6          | 1→5→6     | 6             |

---

## Detailed Test Cases

---

### TC01 — Original Sample (Baseline)

**Purpose:** Verify the program produces correct output on the assignment's own sample.  
**Graph:** 6 nodes, mixed directed/undirected edges.  
**Expected:** GBFS picks node 3 over node 1 from origin 2 (h(3)=2.24 < h(1)=2.83), then reaches goal 5 directly.

```
Output:
TC01_sample.txt Greedy Best First
5 6
2 -> 3 -> 5
```

**Result:** ✅ Pass

---

### TC02 — Origin is Already the Destination

**Purpose:** Test the edge case where the start node is a goal node.  
**Expected:** GBFS immediately returns the origin without expanding any edges.

```
Output:
TC02_origin_is_goal.txt Greedy Best First
1 1
1
```

**Result:** ✅ Pass

---

### TC03 — Single Destination, Direct Edge

**Purpose:** Simplest possible solvable case — two nodes, one edge.  
**Expected:** Finds goal in one step.

```
Output:
TC03_direct_edge.txt Greedy Best First
2 2
1 -> 2
```

**Result:** ✅ Pass

---

### TC04 — No Path Exists (Disconnected Graph)

**Purpose:** Verify the program handles an unsolvable problem gracefully.  
**Graph:** Node 3 exists but has no incoming edges from origin's side.  
**Expected:** Prints "No solution found." without crashing.

```
Output:
TC04_no_path.txt Greedy Best First
No solution found.
```

**Result:** ✅ Pass

---

### TC05 — Linear Chain of Nodes

**Purpose:** Test traversal along a straight-line graph with no branching.  
**Expected:** GBFS follows the only available path: 1→2→3→4→5.

```
Output:
TC05_linear_chain.txt Greedy Best First
5 5
1 -> 2 -> 3 -> 4 -> 5
```

**Result:** ✅ Pass

---

### TC06 — Multiple Destinations, Picks Nearest

**Purpose:** When two goals exist, GBFS should reach whichever is closer by heuristic first.  
**Graph:** Node 2 is at (2,0), Node 3 is at (8,0). Origin at (0,0). Node 2 is directly adjacent.  
**Expected:** Reaches goal 2 first.

```
Output:
TC06_multi_dest_nearest.txt Greedy Best First
2 3
1 -> 2
```

**Result:** ✅ Pass

---

### TC07 — Directed Edges Only (One-Way Streets)

**Purpose:** Confirm directed edges are respected — no backtracking against edge direction.  
**Graph:** Square path 1→2→3→4, all one-way.  
**Expected:** Follows the only valid directed path.

```
Output:
TC07_directed_only.txt Greedy Best First
4 4
1 -> 2 -> 3 -> 4
```

**Result:** ✅ Pass

---

### TC08 — Greedy Mislead (Suboptimal Path)

**Purpose:** Demonstrate a known GBFS limitation — it can be misled by heuristic into a longer path.  
**Graph:** Shortest path is 1→2→3 (cost 5), but GBFS picks 1→4→3 because h(4) < h(2).  
**Expected:** Returns the heuristically guided path, not the cost-optimal one.

```
Output:
TC08_greedy_suboptimal.txt Greedy Best First
3 4
1 -> 4 -> 3
```

**Result:** ✅ Pass (demonstrates expected GBFS behaviour)

---

### TC09 — Tie-Breaking by Node ID

**Purpose:** When multiple neighbours have equal heuristic values, smaller node ID should be expanded first.  
**Graph:** From node 1, neighbours 2, 3, and 4 all have the same h value. Node 4 is the goal and has the smallest ID among those with h=0.  
**Expected:** Direct path 1→4.

```
Output:
TC09_tiebreak_nodeid.txt Greedy Best First
4 4
1 -> 4
```

**Result:** ✅ Pass

---

### TC10 — Large Graph (10 Nodes)

**Purpose:** Test performance and correctness on a more complex graph.  
**Expected:** GBFS navigates efficiently toward goal node 10.

```
Output:
TC10_large_graph.txt Greedy Best First
10 5
1 -> 6 -> 7 -> 10
```

**Result:** ✅ Pass

---

### TC11 — Graph with a Cycle (No Infinite Loop)

**Purpose:** Verify the `visited` set prevents the algorithm from looping infinitely on a cyclic graph.  
**Graph:** Nodes 1→2→3→4→1 form a cycle; node 5 is reachable from 3.  
**Expected:** Finds goal 5 without revisiting nodes.

```
Output:
TC11_cycle.txt Greedy Best First
5 5
1 -> 2 -> 3 -> 5
```

**Result:** ✅ Pass

---

### TC12 — Single Node (Origin = Destination, No Edges)

**Purpose:** Extreme edge case — a graph with only one node and no edges.  
**Expected:** Immediately returns node 1 as the goal with path of length 1.

```
Output:
TC12_single_node.txt Greedy Best First
1 1
1
```

**Result:** ✅ Pass

---

### TC13 — Three Destinations, Only One Reachable

**Purpose:** Test that GBFS finds the one reachable goal even when others are unreachable.  
**Graph:** Goals 3 and 5 have no incoming edges from origin path; only goal 4 is reachable.  
**Expected:** Finds goal 4.

```
Output:
TC13_one_reachable_dest.txt Greedy Best First
4 3
1 -> 2 -> 4
```

**Result:** ✅ Pass

---

### TC14 — Dense Graph (Multiple Paths to Goal)

**Purpose:** Test GBFS on a graph where many paths exist, ensuring it picks the heuristically best one.  
**Expected:** Finds goal 5 via the path with the best heuristic guidance.

```
Output:
TC14_dense_graph.txt Greedy Best First
5 5
1 -> 4 -> 5
```

**Result:** ✅ Pass

---

### TC15 — Forced Long Detour (Directed Edges Block Direct Path)

**Purpose:** The geometrically closest path is blocked by edge direction, forcing GBFS to take a longer route.  
**Graph:** No direct edge from 1 to 2; must go 1→4→3→2.  
**Expected:** Finds goal 2 via the only valid directed path.

```
Output:
TC15_forced_detour.txt Greedy Best First
2 5
1 -> 4 -> 3 -> 2
```

**Result:** ✅ Pass

---

### TC16 — Origin Surrounded by Dead Ends

**Purpose:** Most neighbours of the origin lead nowhere; only one branch reaches the goal.  
**Graph:** Node 1 connects to nodes 2, 3, 4 (dead ends) and node 5 which connects to goal 6.  
**Expected:** GBFS explores the dead ends but ultimately finds the path through node 5.

```
Output:
TC16_dead_ends.txt Greedy Best First
6 6
1 -> 5 -> 6
```

**Result:** ✅ Pass

---

## Observations & Insights

- **TC02 & TC12** confirm GBFS or other searches handles trivial/degenerate cases without errors.
- **TC04** confirms graceful handling of unsolvable problems.
- **TC08** is a classic demonstration of GBFS's weakness — it is **not guaranteed to find the optimal (lowest cost) path**, only a path guided by heuristic distance.
- **TC09** confirms the assignment's tie-breaking rule (ascending node ID) is correctly implemented.
- **TC11** confirms the `visited` set correctly prevents infinite loops on cyclic graphs.
- **TC13 & TC15** confirm that directed edges are strictly respected throughout.
- **TC16** shows GBFS will correctly backtrack from dead-end branches and find the goal.
