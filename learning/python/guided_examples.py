"""Tutor-written Python reference examples.

These are NEW explanations of topics encountered in tutoring, NOT the
learner's independently submitted solutions or completed challenges.

Run:
    python learning/python/guided_examples.py
"""


def reassignment_example():
    money = 100
    money = money - 30
    money = money + 20
    return money  # 90


def conversion_example(text):
    # input() usually returns text; int() converts numeric text.
    return int(text)


def conditional_example(temperature):
    if temperature > 30:
        return "Hot"
    elif temperature < 15:
        return "Cold"
    else:
        return "Comfortable"


def for_loop_example():
    total = 0
    for number in range(1, 11):
        if number % 2 == 0:
            total += number
    return total  # 30


def while_loop_example():
    # Example only: a while-loop running total.
    number = 2
    total = 0
    while number <= 20:
        total += number
        number += 2
    return total  # 110


def boolean_and_example(number):
    # Both comparisons must be true.
    return number % 2 == 0 and number % 3 == 0


def list_indexing_example():
    marks = [65, 70, 85, 90]
    second_mark_before = marks[1]
    marks[1] = 95
    return second_mark_before, marks


if __name__ == "__main__":
    print("Reassignment:", reassignment_example())
    print("Text-to-integer conversion:", conversion_example("25"))
    print("Conditional:", conditional_example(20))
    print("For loop:", for_loop_example())
    print("While loop:", while_loop_example())
    print("Divisible by 2 and 3?", boolean_and_example(12))
    print("List change:", list_indexing_example())
