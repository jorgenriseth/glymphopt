import numpy as np


def gradient_finite_differences(F, x, h, **kwargs):
    return np.array(
        [
            (F(x + (h * x[i]) * ei, **kwargs) - F(x - (h * x[i]) * ei, **kwargs))
            / (2 * (h * x[i]))
            for i, ei in enumerate(np.eye(len(x)))
        ]
    )


def hessian_finite_differences(F, x, h, **kwargs):
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise ValueError("Input 'x' must be a 1D NumPy array.")

    n = x.shape[0]
    hessian = np.zeros((n, n))
    fx = F(x, **kwargs)

    # Iterate through each element of the Hessian matrix
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal elements (second partial derivatives)
                x_plus_h_ei = x.copy()
                x_plus_h_ei[i] += h
                f_plus_h_ei = F(x_plus_h_ei, **kwargs)

                x_minus_h_ei = x.copy()
                x_minus_h_ei[i] -= h
                f_minus_h_ei = F(x_minus_h_ei, **kwargs)

                hessian[i, i] = (f_plus_h_ei - 2 * fx + f_minus_h_ei) / (h**2)
            else:
                # Off-diagonal elements (mixed partial derivatives)
                x_plus_h_ei_plus_h_ej = x.copy()
                x_plus_h_ei_plus_h_ej[i] += h
                x_plus_h_ei_plus_h_ej[j] += h
                f_plus_h_ei_plus_h_ej = F(x_plus_h_ei_plus_h_ej, **kwargs)

                x_plus_h_ei_minus_h_ej = x.copy()
                x_plus_h_ei_minus_h_ej[i] += h
                x_plus_h_ei_minus_h_ej[j] -= h
                f_plus_h_ei_minus_h_ej = F(x_plus_h_ei_minus_h_ej, **kwargs)

                x_minus_h_ei_plus_h_ej = x.copy()
                x_minus_h_ei_plus_h_ej[i] -= h
                x_minus_h_ei_plus_h_ej[j] += h
                f_minus_h_ei_plus_h_ej = F(x_minus_h_ei_plus_h_ej, **kwargs)

                x_minus_h_ei_minus_h_ej = x.copy()
                x_minus_h_ei_minus_h_ej[i] -= h
                x_minus_h_ei_minus_h_ej[j] -= h
                f_minus_h_ei_minus_h_ej = F(x_minus_h_ei_minus_h_ej, **kwargs)

                hessian[i, j] = (
                    f_plus_h_ei_plus_h_ej
                    - f_plus_h_ei_minus_h_ej
                    - f_minus_h_ei_plus_h_ej
                    + f_minus_h_ei_minus_h_ej
                ) / (4 * h**2)

                # The Hessian is symmetric
                hessian[j, i] = hessian[i, j]

    return hessian
