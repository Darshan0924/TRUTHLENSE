# Python Progress Log

_Last updated: 22 October 2026_

## Current checkpoint

- **Most recent independently completed challenge confirmed in the available record:** Challenge **39**, list-element reassignment and printing the complete updated list.
- **Next lesson introduced:** Challenge **40**, `append()`. I have **not yet submitted a verified independent solution** for Challenge 40.
- Challenges in my learning sessions extend from beginner expression/output predictions through control flow, loops, and lists. The exact submissions for every numbered challenge are not available in this archive.

## Confirmed work and learnings

| Challenge or topic | Recorded evidence |
| --- | --- |
| Variable reassignment | Explained `money = 100`, subtract 30 and add 20, resulting in 90. |
| Conditions and indentation | Predicted which statements execute depending on whether an `if` condition is true. |
| For loops / even numbers | Challenge **24D**: submitted a loop with `range(1, 11)`, `i % 2 == 0`, a running total, and a final `print(total)`. |
| While loop accumulation | Challenge **36**: completed a while-loop sum-of-even-numbers exercise; recorded output `Sum: 110`. The exact submitted code is not reproduced because it was not fully recoverable. |
| List indexing | Challenge **38**: correctly identified the result of updating index 2 in a list. |
| List modification | Challenge **39**: submitted `marks = [65, 70, 85, 90]`; then `marks[1] = 95`; then `print(marks)`. Expected output: `[65, 95, 85, 90]`. |

## Important corrections and concepts to reinforce

- Python uses **`!=`** for not-equal; **`!==`** is not Python syntax. This came up during an odd-number filtering exercise.
- List indices start at **0**. So `marks[1]` updates the **second** item.
- A loop running total must be initialized before the loop and updated inside it.
- Conditions involving `and` need *two full comparison expressions*, for example `n % 2 == 0 and n % 3 == 0`. I practiced this, but should keep testing my understanding.
- Indentation determines which statements are inside an `if` or loop.

## Next independent tasks (not marked complete)

- [ ] Challenge 40: use `list.append()` in a program without copying the reference solution.
- [ ] Explain how `append()` differs from replacing `my_list[index]`.
- [ ] Practice a new `and` / `or` condition independently.
- [ ] Continue with list methods and then functions, dictionaries and file handling as they arise in tutoring.
- [ ] Later, add original mini-projects and tests to a dedicated learning repository.

## Authenticity note

`submitted_solutions.py` contains only exercises for which the user's own code is recoverable. `guided_examples.py` is fresh instructional material, not evidence of an independent submission. This archive intentionally does not invent finished solutions, certificates, work experience, or broader ML proficiency.
