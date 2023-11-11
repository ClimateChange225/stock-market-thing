import pandas as pd
import yfinance as yf
import numpy as np
from currency_converter import CurrencyConverter
import matplotlib.pyplot as plt


# https://www.varsitytutors.com/hotmath/hotmath_help/topics/quadratic-regression#:~:text=A%20quadratic%20regression%20is%20the,c%20where%20a%E2%89%A00%20
# https://codegolf.stackexchange.com/questions/154482/find-the-local-maxima-and-minima
# https://www.kaggle.com/code/yaserrahmati/finding-the-maxima-and-minima-of-a-function
# https://www.omnicalculator.com/statistics/cubic-regression#:~:text=The%20cubic%20regression%20function%20takes,affects%20the%20value%20of%20y%20.
# https://www.statology.org/cubic-regression-python/
# https://heartbeat.comet.ml/a-comprehensive-guide-to-logarithmic-regression-d619b202fc8


def matrix_multiplication(l1, l2):
    result = [[0 for j in range(len(l2[0]))] for i in range(len(l1))]
    for i in range(len(l1)):
        for j in range(len(l2[0])):
            for k in range(len(l2)):
                result[i][j] += l1[i][k] * l2[k][j]
    return result


def matrix_determinant(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for i in range(n):
            submatrix = [row[:i] + row[i + 1:] for row in matrix[1:]]
            sign = (-1) ** i
            cofactor = matrix[0][i]
            det += sign * cofactor * matrix_determinant(submatrix)
        return det


def matrix_cofactor(matrix):
    n = len(matrix)
    if n == 1:
        return [[1]]
    elif n == 2:
        return [[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]]
    else:
        matrix_cofactor = []
        for i in range(n):
            cofactor_row = []
            for j in range(n):
                minor = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
                cofactor = (-1) ** (i + j) * matrix_determinant(minor)
                cofactor_row.append(cofactor)
            matrix_cofactor.append(cofactor_row)
        return matrix_cofactor


def matrix_transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    transpose = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    return transpose


def matrix_adjoint(matrix):
    return matrix_transpose(matrix_cofactor(matrix))


def matrix_inverse(matrix):
    matrix_array = np.array(matrix)

    try:
        inverse = np.linalg.inv(matrix_array)
        return inverse.tolist()
    except np.linalg.LinAlgError:
        return None


def input_ticker():
    while True:
        tckr = input("Enter stock ticker: ")
        try:
            yf.download(tckr)
        except:
            print("Please enter a valid ticker!\n")
            continue
        break
    l1 = []
    df1 = yf.download(tckr.upper(), period="1mo")
    c = CurrencyConverter()
    INR_Open = [round(c.convert(i, "USD", "INR"), 2) for i in df1["Open"]]
    days = np.arange(1, len(df1)+1)  # Independent variable (days)
    op_price = INR_Open  # Dependent variable (opening price)
    data = pd.DataFrame({'X': days, 'Y': op_price})
    l1 += [f"{tckr.upper()}", data]
    return l1


def view_price_graph(l1):
    for i in range(1, len(l1)+1):
        print(f"{i}. Stock ticker name: {l1[i-1][0]}")
    tckr_no = input("Enter the number corresponding to your ticker: ")
    while True:
        try:
            tckr_no = int(tckr_no)-1
        except:
            print("Please enter a valid ticker number!")
        break
    df = l1[tckr_no][1]
    x = df.X
    y = df.Y

    plt.title(f"{l1[tckr_no][0]}")
    plt.scatter(x, y)
    plt.xlabel("Days")
    plt.ylabel("Prices (In INR)")
    plt.plot(x, y)
    plt.show()


def regression_models(l1):
    for i in range(1, len(l1) + 1):
        print(f"{i}. Stock ticker name: {l1[i - 1][0]}")
    tckr_no = input("Enter the number corresponding to your ticker: ")
    while True:
        try:
            tckr_no = int(tckr_no) - 1
        except:
            print("Please enter a valid ticker number!")
            continue
        break
    df = l1[tckr_no][1]
    n = len(df)
    sigma_x = df.X.sum()
    sigma_x_squared = (df.X ** 2).sum()
    sigma_y = df.Y.sum()
    sigma_xy = (df.X * df.Y).sum()
    sigma_x_pwr4 = (df.X ** 4).sum()
    sigma_x_cubed = (df.X ** 3).sum()
    sigma_x_sqr_y = ((df.X ** 2) * df.Y).sum()

    m1 = [[n, sigma_x], [sigma_x, sigma_x_squared]]
    m2 = [[sigma_y], [sigma_xy]]

    a_b = matrix_multiplication(matrix_inverse(m1), m2)
    intercept = a_b[0][0]
    slope = a_b[1][0]

    linear = {"a": intercept, "b": slope}

    m1 = [[sigma_x_pwr4, sigma_x_cubed, sigma_x_squared],
          [sigma_x_cubed, sigma_x_squared, sigma_x],
          [sigma_x_squared, sigma_x, n]]

    m2 = [[sigma_x_sqr_y], [sigma_xy], [sigma_y]]

    a_b_c = matrix_multiplication(matrix_inverse(m1), m2)

    a = a_b_c[0][0]
    b = a_b_c[1][0]
    c = a_b_c[2][0]

    quadratic = {"a": a, "b": b, "c": c}

    print(f"Linear Regression Equation: y = {linear['a']:.3f} + {linear['b']:.3f}*x")
    print(f"Quadratic Regression Equation: y = {quadratic['a']:.3f}*x^2 + {quadratic['b']:.3f}*x + {quadratic['c']:.3f}")
    print(f"Logarithmic Regression Equation: y = {linear['a']:.3f} + {linear['b']:.3f}*ln(x)")

    l1[tckr_no].append({"linear": linear, "quadratic": quadratic})

    return l1



def mean_squared_error(l1):
    for i in range(1, len(l1)+1):
        print(f"{i}. Stock ticker name: {l1[i-1][0]}")
    tckr_no = input("Enter the number corresponding to your ticker: ")
    while True:
        try:
            tckr_no = int(tckr_no) - 1
        except:
            print("Please enter a valid ticker number!")
            continue
        break

    if len(l1[tckr_no]) == 3:
        df = l1[tckr_no][1]
        actual_val = list(df["Y"])
        n = len(df)

        # Linear Regression
        a = l1[tckr_no][2]["linear"]["a"]
        b = l1[tckr_no][2]["linear"]["b"]
        pred_val = [a + b*i for i in list(df["X"])]
        MSE_linr = 1/n * sum((std_y - pred_y)**2 for std_y, pred_y in zip(actual_val, pred_val))

        # Quadratic Regression
        a = l1[tckr_no][2]["quadratic"]["a"]
        b = l1[tckr_no][2]["quadratic"]["b"]
        c = l1[tckr_no][2]["quadratic"]["c"]
        pred_val = [a*i*i + b*i + c for i in list(df["X"])]
        MSE_quad = 1/n * sum((std_y - pred_y)**2 for std_y, pred_y in zip(actual_val, pred_val))

        # Logarithmic Regression
        a = l1[tckr_no][2]["linear"]["a"]
        b = l1[tckr_no][2]["linear"]["b"]
        pred_val = [a + b*np.log(i) for i in list(df["X"])]
        MSE_ln = 1/n * sum((std_y - pred_y)**2 for std_y, pred_y in zip(actual_val, pred_val))

        return [MSE_linr, MSE_quad, MSE_ln]

    else:
        print("Please select the option for calculating regression first!")


def plot_regression_subplot(l1):
    for i in range(1, len(l1)+1):
        print(f"{i}. Stock ticker name: {l1[i-1][0]}")
    tckr_no = input("Enter the number corresponding to your ticker: ")
    while True:
        try:
            tckr_no = int(tckr_no) - 1
        except:
            print("Please enter a valid ticker number!")
            continue
        break
    if len(l1[tckr_no]) == 3:
        df = l1[tckr_no][1]
        x = df.X
        actual_y = df.Y
        # Linear
        a = l1[tckr_no][2]["linear"]["a"]
        b = l1[tckr_no][2]["linear"]["b"]
        pred_y_linear = [a + b*i for i in list(df["X"])]
        # Quadratic
        a = l1[tckr_no][2]["quadratic"]["a"]
        b = l1[tckr_no][2]["quadratic"]["b"]
        c = l1[tckr_no][2]["quadratic"]["c"]
        pred_y_quad = [a*i*i + b*i + c for i in list(df["X"])]
        # Logarithmic
        a = l1[tckr_no][2]["linear"]["a"]
        b = l1[tckr_no][2]["linear"]["b"]
        pred_y_ln = [a + b*np.log(i) for i in list(df["X"])]

        # Creating SubPlot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Linear Plot
        axes[0].scatter(x, actual_y, color="red", label="Actual Prices")
        axes[0].plot(x, pred_y_linear, color="blue", label="Linear Regression")
        axes[0].set_title("Linear Regression")
        axes[0].set_xlabel("Days")
        axes[0].set_ylabel("Prices (In INR)")
        axes[0].legend()

        # Quadratic Plot
        axes[1].scatter(x, actual_y, color="red", label="Actual Prices")
        axes[1].plot(x, pred_y_quad, color="blue", label="Quadratic Regression")
        axes[1].set_title("Quadratic Regression")
        axes[1].set_xlabel("Days")
        axes[1].set_ylabel("Prices (In INR)")
        axes[1].legend()

        # Logarithmic Plot
        axes[2].scatter(x, actual_y, color="red", label="Actual Prices")
        axes[2].plot(x, pred_y_ln, color="blue", label="Logarithmic Regression")
        axes[2].set_title("Logarithmic Regression")
        axes[2].set_xlabel("Days")
        axes[2].set_ylabel("Prices (In INR)")
        axes[2].legend()

        plt.tight_layout()
        plt.show()
    else:
        print("Please select the option for calculating regression first!")


def main():
    l1 = []
    while True:
        print("Choose an option:")
        print("1. Enter stock ticker")
        print("2. View price graph")
        print("3. Run regression models")
        print("4. Calculate Mean Squared Error")
        print("5. View regression graph")
        print("6. Exit")

        option = input("Enter your choice: ")

        if option == "1":
            l1.append(input_ticker())
        elif option == "2":
            view_price_graph(l1)
        elif option == "3":
            l1 = regression_models(l1)
        elif option == "4":
            mse_values = mean_squared_error(l1)
            if mse_values:
                mse_linr, mse_quad, mse_ln = mse_values
                print(f"Mean Squared Error (Linear Regression): {mse_linr:.3f}")
                print(f"Mean Squared Error (Quadratic Regression): {mse_quad:.3f}")
                print(f"Mean Squared Error (Logarithmic Regression): {mse_ln:.3f}")
                min_mse = min(mse_linr, mse_quad, mse_ln)
                if min_mse == mse_linr:
                    print("Linear Regression has the lowest MSE.")
                elif min_mse == mse_quad:
                    print("Quadratic Regression has the lowest MSE.")
                else:
                    print("Logarithmic Regression has the lowest MSE.")
            else:
                print("Please select the option for calculating regression first!")
        elif option == "5":
            plot_regression_subplot(l1)
        elif option == "6":
            print("Exiting...")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()