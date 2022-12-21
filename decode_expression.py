def decode_expression(expression):
    # If the expression is a list, it represents a mathematical operation
    if isinstance(expression, list):
        # Decode the operation and the two subexpressions
        operation = expression[0]
        subexpression1 = decode_expression(expression[1])
        subexpression2 = decode_expression(expression[2])

        # Return the infix notation for the operation and the subexpressions
        if operation == '+':
            return f"({subexpression1} + {subexpression2})"
        elif operation == '-':
            return f"({subexpression1} - {subexpression2})"
        elif operation == '*':
            return f"({subexpression1} * {subexpression2})"
        elif operation == '/':
            return f"({subexpression1} / {subexpression2})"
        elif operation == 'sin':
            return f"sin({subexpression1})"
        elif operation == 'cos':
            return f"cos({subexpression1})"
        elif operation == 'tan':
            return f"tan({subexpression1})"
        elif operation == 'exp':
            return f"exp({subexpression1})"
    else:
        # If the expression is not a list, it is a constant value or a variable name
        return str(expression)
///////////////////////////////////////////////
//////////////////////////////////////////////
expression_tree = ['+', ['*', 2, 'x1'], 3]
expression_string = decode_expression(expression_tree)
print(expression_string)  # Output: "(2 * x1 + 3)"
