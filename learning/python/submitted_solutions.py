"""Recovered beginner Python submissions.

These two solutions are based on code submitted in tutoring sessions.
Whitespace and formatting are normalized; exercise boundaries are added.
Do not treat this as a verbatim transcript of the entire tutoring history.
"""


def challenge_24d_sum_even_numbers():
    """Sum even numbers from 1 through 10 (submitted approach)."""
    total = 0
    for i in range(1, 11):
        if i % 2 == 0:
            total = total + i
    print(total)  # 30


def challenge_39_replace_list_element():
    """Replace the second list value and print the updated list."""
    marks = [65, 70, 85, 90]
    marks[1] = 95
    print(marks)  # [65, 95, 85, 90]


if __name__ == "__main__":
    print("Challenge 24D:")
    challenge_24d_sum_even_numbers()

    print("Challenge 39:")
    challenge_39_replace_list_element()
