# 4. Testing – Custom Uninformed Search (CUS1: IDDFS)

The custom uninformed search algorithm implemented for this assignment was **Iterative Deepening Depth-First Search (IDDFS)**. Testing was conducted using the same test suite as BFS to ensure consistency and allow direct comparison between algorithms.

All tests were executed using an automated batch script with the following command:

```
python cus1.py <filename> CUS1
```

A total of **15 test cases** were executed, covering a wide range of scenarios including:

- Standard pathfinding problems
- Origin node already being a goal
- Unreachable destinations
- Multiple goals
- Tie-breaking rules
- Cyclic graphs
- Dead ends
- Wide and deep graph structures

All test cases executed successfully and produced expected results.

---

## 4.1 CUS1 Test Case Summary

| # | Test Case | Scenario | Goal Found | Path | Nodes Created | Result |
|---|-----------|----------|------------|------|---------------|--------|
| 1 | TC01_sample.txt | Assignment sample graph | 4 | [2,1,4] | 8 | ✅ Pass |
| 2 | TC02_origin_is_goal.txt | Origin already goal | 1 | [1] | 1 | ✅ Pass |
| 3 | TC04_no_path.txt | Goal unreachable | None | [] | 5 | ✅ Pass |
| 4 | TC06_multi_dest_nearest.txt | Multiple destinations | 2 | [1,2] | 3 | ✅ Pass |
| 5 | TC09_tiebreak_nodeid.txt | Tie-breaking rule | 4 | [1,4] | 5 | ✅ Pass |
| 6 | TC11_cycle.txt | Graph contains cycle | 5 | [1,2,3,5] | 11 | ✅ Pass |
| 7 | TC16_dead_ends.txt | Dead-end branches | 6 | [1,5,6] | 12 | ✅ Pass |
| 8 | TC_BFS_01_fewer_moves_not_lower_cost.txt | Fewer moves vs lower cost | 5 | [1,2,5] | 7 | ✅ Pass |
| 9 | TC_BFS_02_equal_depth_tiebreak.txt | Equal depth tie-breaking | 4 | [1,2,4] | 7 | ✅ Pass |
| 10 | TC_BFS_03_wide_first_level.txt | Wide graph structure | 7 | [1,5,6,7] | 19 | ✅ Pass |
| 11 | TC_BFS_04_deeper_but_cheaper.txt | Deeper but cheaper path exists | 6 | [1,2,6] | 7 | ✅ Pass |
| 12 | TC_BFS_05_multiple_goals_different_depths.txt | Multiple goals at different depths | 3 | [1,2,3] | 7 | ✅ Pass |
| 13 | TC_BFS_06_directed_trap.txt | Directed graph with no valid route | None | [] | 12 | ✅ Pass |
| 14 | TC_BFS_07_cycle_with_escape.txt | Cycle with valid escape path | 5 | [1,2,3,4,5] | 15 | ✅ Pass |
| 15 | TC_BFS_08_large_wide_graph.txt | Large wide graph | 12 | [1,4,9,12] | 26 | ✅ Pass |

---

## 4.2 Selected Test Case Discussion

### TC01 – Assignment Sample Graph

IDDFS successfully found the path `[2,1,4]`, which matches the BFS result. This demonstrates that IDDFS correctly identifies the shallowest goal despite using a depth-first exploration strategy within each iteration.

### TC04 – No Path Exists

The algorithm correctly returned `None` with an empty path, demonstrating proper handling of unreachable destinations.

### TC09 – Tie-Breaking by Node ID

Where multiple nodes were available at the same depth, IDDFS followed the assignment rule of expanding smaller node IDs first, producing the correct path.

### TC11 – Cycle Handling

The graph contained a cycle, but IDDFS avoided infinite looping by preventing revisiting nodes already in the current path, successfully reaching the goal.

### TC_BFS_01 – Fewer Moves vs Lower Cost

IDDFS selected the path `[1,2,5]`, demonstrating that it prioritises shallowest solutions rather than lowest total cost, consistent with uninformed search behaviour.

### TC_BFS_08 – Large Wide Graph

This test highlighted a key limitation of IDDFS. Although it found the correct path, it created a significantly higher number of nodes (26) due to repeated exploration of upper levels across multiple depth iterations.

---

## 4.3 Testing Observations

The IDDFS implementation performed correctly across all test cases and demonstrated the expected characteristics of the algorithm.

**Key observations:**

- IDDFS consistently found the shallowest goal (minimum number of moves).
- The paths returned by IDDFS were often identical to those produced by BFS.
- IDDFS correctly followed ascending node ID tie-breaking rules.
- The algorithm handled cycles and dead ends without entering infinite loops.
- IDDFS correctly identified scenarios where no valid path exists.
- Compared to BFS, IDDFS generally produced a higher number of nodes due to repeated depth-limited searches.
- Despite higher node counts, IDDFS is more memory efficient, as it only stores the current path rather than the entire frontier.

> Overall, the CUS1 implementation of IDDFS was successful and provided a clear demonstration of the trade-off between time and memory efficiency in uninformed search strategies.
