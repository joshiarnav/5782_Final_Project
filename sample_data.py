import random

#Sampling
def balanced_sampling(count, min_digits=2, D=30, choices=["add", "minus"]):
    questions = []
    answers = []

    for _ in range(count):
        d = random.randint(min_digits, D)
        lower = 10 ** (d - 1)
        upper = 10 ** d - 1

        number1 = random.randint(lower, upper)
        number2 = random.randint(lower, upper)

        choice = random.choice(choices)

        if choice == "add":
            question = f"What is {number1} plus {number2}?"
            answer = str(number1 + number2)
        else:
            question = f"What is {number1} minus {number2}?"
            answer = str(number1 - number2)

        questions.append(question)
        answers.append(answer)

    return questions, answers

def random_sampling(count, D=30, choices=["add", "minus"]):
    questions = []
    answers = []

    for _ in range(count):
        upper = 10 ** D - 1

        number1 = random.randint(0, upper)
        number2 = random.randint(0, upper)

        choice = random.choice(choices)

        if choice == "add":
            question = f"What is {number1} plus {number2}?"
            answer = str(number1 + number2)
        else:
            question = f"What is {number1} minus {number2}?"
            answer = str(number1 - number2)

        questions.append(question)
        answers.append(answer)

    return questions, answers
