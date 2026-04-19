# 4. Testing – Breadth-First Search (BFS)

**Run command:** `python BFS.py <filename> BFS`

The test suite was designed to cover a range of graph scenarios including:

- Normal route finding problems
- Origin node already being a goal
- Unreachable destinations
- Multiple destination nodes
- Tie-breaking behaviour
- Cyclic graphs
- Dead-end branches
- BFS-specific shortest-depth behaviour

A total of **15 test cases** were executed successfully, with all results matching the expected BFS behaviour.

---

## 4.1 BFS Test Case Summary

| # | Test Case | Scenario | Goal Found | Path | Nodes Created | Result |
|---|-----------|----------|------------|------|---------------|--------|
| 1 | TC01_sample.txt | Assignment sample graph | 4 | [2,1,4] | 10 | Pass |
| 2 | TC02_origin_is_goal.txt | Origin already goal | 1 | [1] | 1 | Pass |
| 3 | TC04_no_path.txt | Goal unreachable | None | [] | 2 | Pass |
| 4 | TC06_multi_dest_nearest.txt | Multiple destinations | 2 | [1,2] | 3 | Pass |
| 5 | TC09_tiebreak_nodeid.txt | Tie-breaking rule | 4 | [1,4] | 6 | Pass |
| 6 | TC11_cycle.txt | Graph contains cycle | 5 | [1,2,3,5] | 5 | Pass |
| 7 | TC16_dead_ends.txt | Dead-end branches | 6 | [1,5,6] | 6 | Pass |
| 8 | TC_BFS_01_fewer_moves_not_lower_cost.txt | Short expensive path vs longer cheap path | 5 | [1,2,5] | 5 | Pass |
| 9 | TC_BFS_02_equal_depth_tiebreak.txt | Equal depth goal paths | 4 | [1,2,4] | 5 | Pass |
| 10 | TC_BFS_03_wide_first_level.txt | Wide first level graph | 7 | [1,5,6,7] | 7 | Pass |
| 11 | TC_BFS_04_deeper_but_cheaper.txt | Deeper path has lower cost | 6 | [1,2,6] | 5 | Pass |
| 12 | TC_BFS_05_multiple_goals_different_depths.txt | Goals at different depths | 3 | [1,2,3] | 5 | Pass |
| 13 | TC_BFS_06_directed_trap.txt | No valid directed route | None | [] | 3 | Pass |
| 14 | TC_BFS_07_cycle_with_escape.txt | Cycle with valid exit path | 5 | [1,2,3,4,5] | 5 | Pass |
| 15 | TC_BFS_08_large_wide_graph.txt | Large wide graph | 12 | [1,4,9,12] | 12 | Pass |

---

## 4.2 Selected Test Case Discussion

### TC01 – Assignment Sample Graph

This test used the sample graph structure provided in the assignment handout. BFS successfully found destination node 4 using the path `[2,1,4]`. This demonstrates that BFS expands nodes level by level and stops when the first valid goal node is reached.

### TC04 – No Path Exists

This graph contained no possible route from the origin to the destination. BFS correctly returned no solution (`None`) and an empty path, demonstrating appropriate failure handling.

### TC09 – Tie-Breaking by Node ID

Where multiple nodes were available at the same search depth, BFS correctly followed the assignment rule of expanding the smaller node number first.

### TC11 – Cycle Detection

This graph contained a cycle. BFS successfully reached the goal without entering an infinite loop, confirming that path checking prevented repeated traversal.

### TC_BFS_01 – Fewer Moves Not Lower Cost

This test showed that BFS prioritises the path with the fewest number of moves rather than the lowest total edge cost. BFS selected `[1,2,5]`, which was shorter in depth but more expensive in cost than the alternative route.

---

## 4.3 Testing Observations

The BFS implementation performed reliably across all tested scenarios. Key observations include:

- BFS always returned the shallowest solution path (minimum number of moves).
- BFS does not consider path cost when selecting nodes.
- BFS correctly followed ascending node ID tie-breaking rules.
- BFS handled cycles and dead-end branches successfully.
- BFS returned no solution when no valid path existed.
- Wider graphs caused larger node creation counts, demonstrating the memory cost of BFS.

Overall, the BFS algorithm was successfully implemented and passed all selected functional tests.

---

## Test Case Results

```
=====================================
Running BFS Test Cases
=====================================

PathFinder-test.txt BFS
4 10
[2,1,4]
-------------------------------------
TC01_sample.txt BFS
4 10
[2,1,4]
-------------------------------------
TC02_origin_is_goal.txt BFS
1 1
[1]
-------------------------------------
TC04_no_path.txt BFS
None 2
[]
-------------------------------------
TC06_multi_dest_nearest.txt BFS
2 3
[1,2]
-------------------------------------
TC09_tiebreak_nodeid.txt BFS
4 6
[1,4]
-------------------------------------
TC11_cycle.txt BFS
5 5
[1,2,3,5]
-------------------------------------
TC16_dead_ends.txt BFS
6 6
[1,5,6]
-------------------------------------
TC_BFS_01_fewer_moves_not_lower_cost.txt BFS
5 5
[1,2,5]
-------------------------------------
TC_BFS_02_equal_depth_tiebreak.txt BFS
4 5
[1,2,4]
-------------------------------------
TC_BFS_03_wide_first_level.txt BFS
7 7
[1,5,6,7]
-------------------------------------
TC_BFS_04_deeper_but_cheaper.txt BFS
6 5
[1,2,6]
-------------------------------------
TC_BFS_05_multiple_goals_different_depths.txt BFS
3 5
[1,2,3]
-------------------------------------
TC_BFS_06_directed_trap.txt BFS
None 3
[]
-------------------------------------
TC_BFS_07_cycle_with_escape.txt BFS
5 5
[1,2,3,4,5]
-------------------------------------
TC_BFS_08_large_wide_graph.txt BFS
12 12
[1,4,9,12]
-------------------------------------
```
