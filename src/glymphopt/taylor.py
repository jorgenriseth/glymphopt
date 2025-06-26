import numpy as np


def test_gradient(objective_function, gradient_function, x, p=None):
    """
    Performs a Taylor test to verify the correctness of a gradient implementation.

    This function checks if the error between the function's actual change and
    the first-order Taylor approximation decreases quadratically with the step size.

    Args:
        objective_function (callable): A function that takes a numpy array x
            and returns a scalar value.
        gradient_function (callable): A function that takes a numpy array x
            and returns the gradient as a numpy array.
        x (np.ndarray): The point at which to perform the test.
        p (np.ndarray, optional): The perturbation direction. If None, a random
            direction is chosen. Defaults to None.
    """
    if p is None:
        p = np.abs(np.random.randn(len(x)))

    f_x = objective_function(x)
    grad_x = gradient_function(x)
    grad_p = np.dot(grad_x, p)
    print("Taylor Test for the Gradient:")
    print("-----------------------------")
    print(f"{'h':>8s}{'Error':>20s}{'Error Ratio':>15s}")
    print("-" * 43)

    last_error = None
    for i in range(6):
        h = 0.1 * (1 / 2) ** i
        f_x_hp = objective_function(x + h * p)

        # First-order Taylor error
        error = abs(f_x_hp - f_x - h * grad_p)

        error_ratio = "-"
        if last_error is not None and error > 0:
            error_ratio = f"{last_error / error:.2f}"

        print(f"{h:8.2e}{error:20.6e}{error_ratio:>15s}")
        last_error = error


def test_hessian(
    objective_function, gradient_function, hessian_vector_product, x, p=None
):
    """
    Performs a Taylor test to verify the correctness of a Hessian-vector product.

    This function checks if the error between the function's actual change and
    the second-order Taylor approximation decreases cubically with the step size.

    Args:
        objective_function (callable): A function that takes a numpy array x
            and returns a scalar value.
        gradient_function (callable): A function that takes a numpy array x
            and returns the gradient as a numpy array.
        hessian_vector_product (callable): A function that takes a point x and
            a vector p and returns the Hessian-vector product H(x)p.
        x (np.ndarray): The point at which to perform the test.
        p (np.ndarray, optional): The perturbation direction. If None, a random
            direction is chosen. Defaults to None.
    """
    if p is None:
        p = np.random.randn(len(x))

    f_x = objective_function(x)
    grad_x = gradient_function(x)
    grad_p = np.dot(grad_x, p)
    hvp = hessian_vector_product(x, p)
    p_hvp = np.dot(p, hvp)

    print("\nTaylor Test for the Hessian:")
    print("----------------------------")
    print(f"{'h':>8s}{'Error':>20s}{'Error Ratio':>15s}")
    print("-" * 43)

    last_error = None
    for i in range(5):
        h = 0.1 * (1 / 2) ** i
        f_x_hp = objective_function(x + h * p)

        # Second-order Taylor error
        error = abs(f_x_hp - f_x - h * grad_p - 0.5 * h**2 * p_hvp)

        error_ratio = "-"
        if last_error is not None and error > 0:
            error_ratio = f"{last_error / error:.2f}"

        print(f"{h:8.2e}{error:20.6e}{error_ratio:>15s}")
        last_error = error
