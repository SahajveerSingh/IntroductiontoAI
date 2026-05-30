# Test Report – Integration and GUI

## Student Information

- Name: Mithakshana Ariyarathna
- Role: Member  – Integration and GUI
- Assignment: COS30019 Assignment 2B
- Component: Traffic-Based Route Guidance System (TBRGS)

---

# Introduction

This test report documents the testing conducted for the Integration and GUI component of the Traffic-Based Route Guidance System (TBRGS).

The testing focused on:
- GUI functionality
- DFS route search integration
- travel time calculation
- SCATS route generation
- user input validation
- end-to-end workflow testing

The DFS route search implemented in Assignment 2A was adapted and integrated into the Assignment 2B system using estimated travel time as the edge cost.

---

# Testing Environment

| Component | Details |
|---|---|
| Programming Language | Python 3 |
| IDE | Visual Studio Code |
| GUI Framework | Tkinter |
| Operating System | Windows 11 |
| Version Control | GitHub |
| Route Search Algorithm | Depth First Search (DFS) |

---

# Features Tested

The following features were tested:

- GUI launch and display
- Origin and destination input
- Prediction time input
- DFS route search
- Travel time estimation
- Top-k route recommendation display
- Error handling
- SCATS graph integration
- Configuration file loading
- End-to-end integration

---

# Test Cases

| Test ID | Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| T1 | Launch GUI | Run gui.py | GUI window displayed | GUI displayed successfully | Pass |
| T2 | Valid route search | 2000 → 3977 | Routes generated | Routes displayed correctly | Pass |
| T3 | Missing origin | Empty origin | Error message shown | Error displayed | Pass |
| T4 | Missing destination | Empty destination | Error message shown | Error displayed | Pass |
| T5 | Same origin/destination | 2000 → 2000 | Warning displayed | Warning displayed | Pass |
| T6 | Invalid origin SCATS | 9999 | Invalid site error | Error displayed | Pass |
| T7 | Invalid destination SCATS | 9999 | Invalid site error | Error displayed | Pass |
| T8 | Morning traffic prediction | Time = 08:00 | Higher traffic flow used | Correct output displayed | Pass |
| T9 | Midday traffic prediction | Time = 12:00 | Lower traffic flow used | Correct output displayed | Pass |
| T10 | DFS route search | Valid graph | Routes generated | DFS worked correctly | Pass |
| T11 | Top-k routes | Multiple paths | Routes sorted by travel time | Correct routes displayed | Pass |
| T12 | Travel time calculation | Distance + flow | Correct estimated time | Time calculated correctly | Pass |
| T13 | Config file loading | config.json | Default settings loaded | Config loaded successfully | Pass |
| T14 | Multiple searches | Repeated searches | GUI remains stable | Stable execution | Pass |
| T15 | Full integration workflow | End-to-end search | Successful route recommendation | Successful output generated | Pass |

---

# Integration Testing

Integration testing was conducted to ensure that:
- the GUI correctly communicates with the integration module
- the integration module correctly builds the SCATS graph
- DFS route search correctly processes weighted edges
- travel time calculation correctly converts traffic flow into estimated time
- route results are correctly displayed in the GUI

The system successfully connected all modules and produced route recommendations using estimated travel time.

---

# GUI Testing

The GUI was tested for:
- window display
- button functionality
- user input validation
- route output formatting
- repeated searches
- error handling

The GUI remained stable throughout all testing scenarios.

---

# DFS Route Search Testing

The DFS route search algorithm was tested using:
- multiple SCATS paths
- valid and invalid routes
- repeated searches
- travel time weighted edges

The DFS implementation successfully generated valid routes and returned the top-k shortest travel time routes.

---

# Travel Time Testing

Travel time estimation was tested using:
- low traffic flow
- medium traffic flow
- high traffic flow
- different road distances

The travel time conversion formula produced realistic estimated travel times according to the assignment specification.

The assignment formula used was:

flow = -1.4648375(speed)^2 + 93.75(speed)

Travel time was then calculated using:

time = distance / speed

An additional 30-second delay was added at each controlled intersection.

---

# Known Limitations

The current prediction function is implemented as a placeholder and should later be replaced with the final LSTM/GRU prediction model developed by other team members.

The current SCATS graph is a simplified demonstration network and may be extended using the complete Boroondara dataset.

---

# Conclusion

The Integration and GUI component was successfully tested and integrated into the Traffic-Based Route Guidance System.

The system correctly connected:
- GUI interaction
- DFS route search
- SCATS graph processing
- travel time estimation
- route recommendation output

The testing demonstrated that the system is capable of generating traffic-based route recommendations using estimated travel time and graph search techniques.