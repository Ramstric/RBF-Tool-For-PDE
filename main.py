
# Inserted values
dictionary = {"a": 1, "b": 2, "c": -1}  # Predefined values


def parse_operation(operation: str, dictionary: dict):
    chars = operation.split()
    operators = chars[1::2]

    # Operators
    op = {'+': lambda x, y: x + y,
          '-': lambda x, y: x - y}

    temp = dictionary[chars[0]]

    for char in chars[1:]:
        if char in operators:
            temp_op = char
            continue
        else:
            temp = op[temp_op](temp, dictionary[char])

    return temp


print(parse_operation("a + b - c", dictionary))
