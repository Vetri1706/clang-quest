# Original C++ learning curriculum

This catalog contains 24 original C++20 programming challenges: eight beginner, eight intermediate, and eight advanced. Every challenge has a real-world or game-development scenario, explicit input/output rules, constraints, learning objectives, concise concept explanations, three progressive hints, a compilable starter, a complete reference program, two public examples, and three hidden grading cases.

The first four challenges deliberately progress through arithmetic, conditions, loops, and functions. Later challenges remain available for free choice; the data imposes no prerequisite locks. Successful submissions can award the specified XP once per challenge: 50 for beginner, 100 for intermediate, and 150 for advanced, totaling 2,400 available XP.

| Order | Challenge | Level | Track | Main skill |
|---:|---|---|---|---|
| 1 | Café receipt | Beginner | Foundations | Exact integer arithmetic |
| 2 | Stockroom signal | Beginner | Foundations | Strict comparison boundaries |
| 3 | Delivery odometer | Beginner | Foundations | Accumulation loops |
| 4 | Quest reward function | Beginner | Foundations | Reusable functions |
| 5 | Temperature watch | Beginner | Systems | One-pass maximum |
| 6 | Level-up ledger | Beginner | Games | Integer division |
| 7 | Lexical signal counter | Beginner | Compilers | Character scanning |
| 8 | Latency snapshot | Beginner | Systems | Wide sums and rounding |
| 9 | Tile navigator | Intermediate | Games | Manhattan distance |
| 10 | Duplicate reservations | Intermediate | Foundations | Set membership and idempotency |
| 11 | Burst detector | Intermediate | Systems | Sliding windows |
| 12 | Warehouse merge | Intermediate | Foundations | Ordered map aggregation |
| 13 | Incident streak | Intermediate | Systems | Adjacent-state tracking |
| 14 | Packet parity | Intermediate | Systems | Bitwise XOR reduction |
| 15 | Expression scanner | Intermediate | Compilers | Tokenization |
| 16 | Leaderboard cut | Intermediate | Games | Sorting and duplicates |
| 17 | Request cache | Advanced | Systems | LRU with a list and hash map |
| 18 | Rescue route | Advanced | Games | Breadth-first search |
| 19 | Build graph | Advanced | Compilers | Deterministic topological sorting |
| 20 | Arena event scheduler | Advanced | Games | Greedy interval scheduling |
| 21 | SSA name ledger | Advanced | Compilers | Straight-line definition/use renaming |
| 22 | Bytecode stack machine | Advanced | Compilers | Stack interpretation |
| 23 | Network route planner | Advanced | Systems | Dijkstra shortest paths |
| 24 | Frame budget optimizer | Advanced | Games | 0/1 knapsack dynamic programming |

## Integration

`curriculum.json` is the complete source asset. It contains grading secrets: reference solutions and hidden tests. The learner-facing catalog should omit `reference_solution` and every test with `hidden: true`; the local grading service keeps the complete asset. Never trust XP totals supplied by a browser. Grade submissions on the service and persist first-completion rewards there.

Starter functions intentionally return a simple default result. They compile, but do not solve the task. Keep the supplied `main` visible so learners can see the relationship between input, their function, and output. The starter includes only the standard headers required by its complete reference implementation; no external C++ dependencies are needed.

Hints are ordered from a conceptual nudge to a concrete algorithm outline. A mentor can combine the current hint with the learner's compiler diagnostic or failed public case. Hidden tests should remain undisclosed. The story and concepts supply grounded lesson context; they do not certify a language model's programming ability.

The compiler track introduces selected compiler-related algorithms. The SSA exercise handles straight-line assignments only, without control-flow joins or phi nodes, and deliberately makes that boundary explicit. The checksum lesson also explicitly distinguishes XOR parity from cryptographic integrity. These are bounded teaching exercises, not production implementations of LLVM, a network protocol, or a game engine.

## Reproduction and validation

Run `python3 build_content.py` to recreate the JSON. Run `python3 validate_content.py` to compile both programs for every challenge and execute all reference cases. Validation uses `/usr/bin/clang++ -std=c++20 -O0 -Wall -Wextra`, at most two simultaneous compiler processes, 20-second compilation deadlines, and 2-second execution deadlines for these trusted, authored reference programs. This validator is not an untrusted-code runner.

The machine-readable `../../reports/curriculum_validation.json` records the compiler version, the exact catalog SHA-256 digest, and per-challenge results. It verifies schema essentials, unique challenge identifiers, balanced difficulty counts, three hints per challenge, two public and at least three hidden cases, absence of placeholder comments, successful starter/reference compilation, and exact reference output. These finite cases provide concrete checks rather than a proof of correctness for all valid inputs.
