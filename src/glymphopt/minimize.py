import itertools
import time

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import OptimizeResult


def adaptive_grid_search(
    func,
    lower_bounds,
    upper_bounds,
    axis_transforms=None,
    n_points_per_dim=5,
    n_iterations=4,
):
    """
    Performs an adaptive grid search with historical expansion, optional axis
    transformations, and returns a complete history of all function evaluations.

    Args:
        func (callable): The function to minimize. Accepts a single NumPy array point.
        lower_bounds (list or np.ndarray): Absolute lower bounds for the search.
        upper_bounds (list or np.ndarray): Absolute upper bounds for the search.
        axis_transforms (list, optional): A list of tuples, one for each dimension.
            Each tuple should be (forward_transform, inverse_transform).
            For a linear axis, use None.
            Example: [(np.log10, lambda x: 10**x), None] for a log-10 scale
            on the first axis and a linear scale on the second. Defaults to None.
        n_points_per_dim (int, optional): Points to sample per dimension. Defaults to 5.
        n_iterations (int, optional): Refinement iterations. Defaults to 4.

    Returns:
        tuple: A tuple containing three items:
            - dict: The best result found, including 'point' and 'value'.
            - list: A history of the best result from each iteration.
            - list: A complete log of every function evaluation, where each
                    entry is a dict {'point': array, 'value': float}.
    """
    # --- Initialization ---
    original_lower_bounds = np.array(lower_bounds, dtype=float)
    original_upper_bounds = np.array(upper_bounds, dtype=float)
    current_lower_bounds = original_lower_bounds.copy()
    current_upper_bounds = original_upper_bounds.copy()

    all_evaluated_points = set()
    n_dims = len(lower_bounds)
    iteration_history = []
    overall_best = {"point": None, "value": float("inf")}

    # NEW: A list to store every single evaluation for the final output
    full_evaluation_history = []

    if axis_transforms and len(axis_transforms) != n_dims:
        raise ValueError(
            "The length of 'axis_transforms' must match the number of dimensions."
        )

    print("--- Starting Adaptive Grid Search (with Full History Logging) ---")

    for i in range(n_iterations):
        print(f"\nIteration {i + 1}/{n_iterations}")
        print(f"  Current Search Bounds:")
        for dim in range(n_dims):
            is_log = " (Log Scale)" if axis_transforms and axis_transforms[dim] else ""
            print(
                f"    Dim {dim + 1}{is_log}: [{current_lower_bounds[dim]:.4g}, {current_upper_bounds[dim]:.4g}]"
            )

        # --- 1. Create Grid ---
        grid_axes = []
        for d in range(n_dims):
            lower_b, upper_b = current_lower_bounds[d], current_upper_bounds[d]
            transform_pair = axis_transforms[d] if axis_transforms else None
            if transform_pair:
                fwd, inv = transform_pair
                final_axis = inv(
                    np.linspace(fwd(lower_b), fwd(upper_b), n_points_per_dim)
                )
            else:
                final_axis = np.linspace(lower_b, upper_b, n_points_per_dim)
            grid_axes.append(final_axis)

        # --- 2. Evaluate Points & Log Full History ---
        points_to_evaluate = np.array(list(itertools.product(*grid_axes)))
        all_evaluated_points.update(map(tuple, points_to_evaluate))

        # NEW: Explicit loop to capture each evaluation for the full history log
        values = []
        for point in points_to_evaluate:
            value = func(point)
            values.append(value)
            full_evaluation_history.append({"point": point, "value": value})
        values = np.array(values)

        min_index = np.argmin(values)
        current_best_point = points_to_evaluate[min_index]
        current_best_value = values[min_index]

        print(f"  Best point in this iteration: {np.round(current_best_point, 4)}")
        print(f"  Minimal value in this iteration: {current_best_value:.4f}")

        iteration_history.append(
            {
                "iteration": i + 1,
                "best_point": current_best_point,
                "best_value": current_best_value,
            }
        )
        if current_best_value < overall_best["value"]:
            overall_best = {"point": current_best_point, "value": current_best_value}

        # --- 3. Determine New Bounds for Next Iteration ---
        if i < n_iterations - 1:
            new_lower_bounds, new_upper_bounds = (
                np.zeros_like(current_lower_bounds),
                np.zeros_like(current_upper_bounds),
            )
            for d in range(n_dims):
                coord_index = np.where(np.isclose(grid_axes[d], current_best_point[d]))[
                    0
                ][0]
                new_lower = (
                    grid_axes[d][coord_index - 1]
                    if coord_index > 0
                    else grid_axes[d][0]
                )
                new_upper = (
                    grid_axes[d][coord_index + 1]
                    if coord_index < n_points_per_dim - 1
                    else grid_axes[d][-1]
                )

                # Expansion Logic (unchanged)
                is_on_lower_edge = coord_index == 0
                is_on_original_lower_bound = np.isclose(
                    current_lower_bounds[d], original_lower_bounds[d]
                )
                if is_on_lower_edge and not is_on_original_lower_bound:
                    print(f"    -> Expanding lower bound for Dim {d + 1}...")
                    candidates = [
                        p[d] for p in all_evaluated_points if p[d] < new_lower
                    ]
                    if candidates:
                        new_lower = max(candidates)
                    else:  # Fallback...
                        # (omitted for brevity, code is identical to previous version)
                        step = (current_upper_bounds[d] - current_lower_bounds[d]) / (
                            n_points_per_dim - 1
                        )
                        new_lower -= step
                    new_lower = max(new_lower, original_lower_bounds[d])

                is_on_upper_edge = coord_index == n_points_per_dim - 1
                is_on_original_upper_bound = np.isclose(
                    current_upper_bounds[d], original_upper_bounds[d]
                )
                if is_on_upper_edge and not is_on_original_upper_bound:
                    print(f"    -> Expanding upper bound for Dim {d + 1}...")
                    candidates = [
                        p[d] for p in all_evaluated_points if p[d] > new_upper
                    ]
                    if candidates:
                        new_upper = min(candidates)
                    else:  # Fallback...
                        # (omitted for brevity, code is identical to previous version)
                        step = (current_upper_bounds[d] - current_lower_bounds[d]) / (
                            n_points_per_dim - 1
                        )
                        new_upper += step
                    new_upper = min(new_upper, original_upper_bounds[d])

                new_lower_bounds[d], new_upper_bounds[d] = new_lower, new_upper

            current_lower_bounds, current_upper_bounds = (
                new_lower_bounds,
                new_upper_bounds,
            )

    print("\n--- Adaptive Grid Search Finished ---")
    print(f"Final Optimal Point: {np.round(overall_best['point'], 6)}")
    print(f"Final Optimal Value: {overall_best['value']:.6f}")

    return overall_best, iteration_history, full_evaluation_history


def projected_newton_solver(
    fun, jac, hessp, bounds, x0, tol=1e-6, max_iter=100, inner_tol=1e-5, verbose=False
):
    """
    Implements a Projected Newton method for box-constrained optimization.

    This method uses an active-set approach to solve problems of the form:
        min f(x)
        subject to: lb <= x <= ub

    Args:
        fun (callable): The objective function to be minimized, fun(x) -> float.
        jac (callable): The gradient (or Jacobian) of the objective function, jac(x) -> ndarray.
        hessp (callable): Function that computes the Hessian-vector product, hessp(x, p) -> ndarray.
        bounds (scipy.optimize.Bounds): An object with `lb` and `ub` attributes.
        x0 (ndarray): The initial guess.
        tol (float, optional): Tolerance for the projected gradient norm to determine convergence. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of outer iterations. Defaults to 100.
        inner_tol (float, optional): Tolerance for the inner Conjugate Gradient solver. Defaults to 1e-5.
        verbose (bool, optional): If True, print progress at each iteration. Defaults to False.

    Returns:
        OptimizeResult: An object containing the optimization results.
    """
    tic = time.time()
    # --- Step 0: Initialization ---
    x = np.clip(x0, bounds.lb, bounds.ub)  # Ensure initial guess is feasible
    lb, ub = bounds.lb, bounds.ub

    # Counters for function/gradient/hessian evaluations
    nfev, njev, nhev = 0, 0, 0

    if verbose:
        print(f"{'Iter':>4} {'x':>30} {'Objective':>13} {'Proj. Grad Norm':>18}")
        print("-" * 70)

    for i in range(max_iter):
        # --- Step 1: Check for Convergence ---
        f_val = fun(x)
        g = jac(x)
        nfev += 1
        njev += 1

        # Calculate the projected gradient. This is the correct stopping criterion for
        # constrained optimization. It is zero if and only if the KKT conditions are met.
        projected_gradient = np.where(
            (x == lb) & (g > 0), 0, np.where((x == ub) & (g < 0), 0, g)
        )

        projected_grad_norm = np.linalg.norm(projected_gradient)

        if verbose:
            x_str = np.array2string(x, formatter={"float_kind": lambda x: f"{x:.3e}"})
            print(f"{i:4d} {x_str:>30} {f_val:13.6e} {projected_grad_norm:18.6e}")

        if projected_grad_norm < tol:
            return OptimizeResult(
                x=x,
                success=True,
                nit=i,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
                fun=f_val,
                jac=g,
                message="Convergence: Projected gradient norm below tolerance.",
                walltime=time.time() - tic,
            )

        # --- Step 2: Identify Active and Inactive Sets ---
        # The active set are variables at a bound where the gradient pushes them
        # further out. These are temporarily "fixed".
        active_mask = ((x == lb) & (g > 0)) | ((x == ub) & (g < 0))
        inactive_mask = ~active_mask

        # If all variables are active, we can't make a Newton step.
        # The algorithm should have already converged if this is an optimal point.
        if np.all(active_mask):
            return OptimizeResult(
                x=x,
                success=False,
                nit=i,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
                fun=f_val,
                jac=g,
                message="Stalled: All variables are active but not converged.",
            )

        # --- Step 3: Compute Search Direction (Subspace Newton Step) ---
        g_inactive = g[inactive_mask]

        # This wrapper creates the "reduced" Hessian-vector product for the free variables.
        # It's the key to using the full `hessp` in a subspace optimization.
        def reduced_hessp(p_inactive):
            nonlocal nhev
            p_full = np.zeros_like(x)
            p_full[inactive_mask] = p_inactive
            # The Hessian-vector product is only evaluated when CG needs it
            nhev += 1
            Hp_full = hessp(x, p_full)
            return Hp_full[inactive_mask]

        n_inactive = np.sum(inactive_mask)
        H_inactive_op = LinearOperator((n_inactive, n_inactive), matvec=reduced_hessp)

        # Solve the reduced Newton system: H_inactive * d_inactive = -g_inactive
        d_inactive, _ = cg(H_inactive_op, -g_inactive, rtol=inner_tol)

        # Reconstruct the full search direction
        d = np.zeros_like(x)
        d[inactive_mask] = d_inactive

        # --- Step 4: Line Search Along the Feasible Arc ---
        # 1. Find the maximum step `alpha_max` before hitting a bound
        with np.errstate(
            divide="ignore"
        ):  # Ignore division by zero if d_inactive is zero
            dist_to_bounds = np.where(
                d_inactive < 0,
                (lb[inactive_mask] - x[inactive_mask]) / d_inactive,
                (ub[inactive_mask] - x[inactive_mask]) / d_inactive,
            )

        # We only care about positive distances that block our path
        dist_to_bounds[dist_to_bounds <= 0] = np.inf
        alpha_max = min(1.0, np.min(dist_to_bounds))  # Newton step is at most 1

        # 2. Perform backtracking line search
        alpha = alpha_max
        c = 1e-4  # Armijo condition constant
        g_dot_d = np.dot(g, d)

        # Fallback for non-descent directions (Hessian not positive definite)
        if g_dot_d >= 0:
            # The Newton direction is not a descent direction.
            # Fall back to a projected gradient descent step.
            print("Falling back to projected gradient.")
            d = -projected_gradient
            g_dot_d = np.dot(g, d)
            alpha = 1.0  # Reset alpha for gradient step

        while fun(x + alpha * d) > f_val + c * alpha * g_dot_d:
            alpha /= 2.0
            nfev += 1
            if alpha < 1e-8:
                print(f"{f_val:10.4e} {c:10.4e} {g_dot_d:10.4e}")
                return OptimizeResult(
                    x=x,
                    success=False,
                    nit=i,
                    nfev=nfev,
                    njev=njev,
                    nhev=nhev,
                    fun=f_val,
                    jac=g,
                    message="Stalled: Line search failed to find a suitable step.",
                    walltime=time.time() - tic,
                )

        # --- Step 5: Update Iterate ---
        x = x + alpha * d
        # A final clip for numerical safety, though theoretically not needed
        # due to the careful `alpha_max` calculation.
        x = np.clip(x, lb, ub)

    results = OptimizeResult(
        x=x,
        success=False,
        nit=max_iter,
        nfev=nfev,
        njev=njev,
        nhev=nhev,
        fun=fun(x),
        jac=jac(x),
        message="Maximum number of iterations reached.",
        walltime=time.time() - tic,
    )
    return results


def solve_trust_region_subproblem_projected_cg(x, g, hessp, bounds, delta):
    """
    Solves the trust-region subproblem with box constraints using a
    Projected Conjugate Gradient (Steihaug-Toint) method.

    This function finds an approximate solution p to:
        min m(p) = g^T p + 0.5 * p^T H p
        s.t. ||p|| <= delta
             lb <= x + p <= ub
    """
    p = np.zeros_like(x)
    r = g
    d = -r

    # Precompute bounds on the step p
    p_lb = bounds.lb - x
    p_ub = bounds.ub - x

    # Failsafe for max iterations
    for j in range(len(x)):
        # Curvature test
        H_d = hessp(x, d)
        d_H_d = np.dot(d, H_d)

        if d_H_d <= 0:
            # Negative curvature detected. Find intersection with trust region and box bounds.
            # This requires solving a quadratic equation for the trust region boundary
            # and linear equations for the box boundaries.
            # For simplicity, we can just step to the trust-region boundary here.
            a = np.dot(d, d)
            b = 2 * np.dot(p, d)
            c = np.dot(p, p) - delta**2
            tau = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            p = p + tau * d
            return p, H_d  # Return p and the last H*d

        alpha = np.dot(r, r) / d_H_d
        p_new = p + alpha * d

        # Check against trust region boundary
        if np.linalg.norm(p_new) > delta:
            a = np.dot(d, d)
            b = 2 * np.dot(p, d)
            c = np.dot(p, p) - delta**2
            tau = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            p = p + tau * d
            return p, None  # H_d is not computed for this final p

        # Check against box bounds
        if np.any(p_new < p_lb) or np.any(p_new > p_ub):
            # Find largest step `beta` along `d` that is feasible
            with np.errstate(divide="ignore"):
                betas = np.where(d > 0, (p_ub - p) / d, (p_lb - p) / d)
            beta = np.min(betas[betas > 0])
            p = p + beta * d
            return p, None

        p = p_new
        r_new = r + alpha * H_d

        if np.linalg.norm(r_new) < 1e-9:
            return p, None  # Converged

        beta = np.dot(r_new, r_new) / np.dot(r, r)
        r = r_new
        d = -r + beta * d

    return p, None


def custom_trust_region_solver(fun, jac, hessp, bounds, x0, tol=1e-6, max_iter=100):
    x = np.clip(x0, bounds.lb, bounds.ub)
    delta = 1.0  # Initial trust radius
    eta1, eta2 = 0.1, 0.75

    # Counters
    nfev, njev, nhev = 0, 0, 0

    for i in range(max_iter):
        g = jac(x)
        njev += 1

        # --- Solve Subproblem ---
        # Note: hessp is called inside the subproblem solver
        p, H_p = solve_trust_region_subproblem_projected_cg(x, g, hessp, bounds, delta)
        nhev += 1  # Approximation

        # --- Calculate rho ---
        f_old = fun(x)
        nfev += 1

        # Predicted reduction
        if H_p is None:
            H_p = hessp(x, p)  # Recompute if needed
        pred = -(np.dot(g, p) + 0.5 * np.dot(p, H_p))

        # If pred is very small, we might be done
        if pred < 1e-12:
            return OptimizeResult(
                x=x,
                success=True,
                nit=i,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
                fun=f_old,
                jac=g,
                message="Predicted reduction is negligible. Converged.",
            )

        f_new = fun(x + p)
        nfev += 1
        ared = f_old - f_new
        rho = ared / pred

        # --- Update Step and Radius ---
        if rho < eta1 / 4.0:  # Very bad step
            delta /= 4.0
        elif rho < eta1:  # Bad step
            delta /= 2.0
        elif (
            rho > eta2 and np.abs(np.linalg.norm(p) - delta) < 1e-6
        ):  # Very good step on boundary
            delta *= 2.0

        if rho > eta1:
            x = np.clip(x + p, bounds.lb, bounds.ub)  # Accept step

        if np.linalg.norm(g) < tol and pred < tol:
            return OptimizeResult(
                x=x,
                success=True,
                nit=i,
                nfev=nfev,
                njev=njev,
                nhev=nhev,
                fun=f_new,
                jac=jac(x),
                message="Convergence criteria met.",
            )

    return OptimizeResult(
        x=x,
        success=False,
        nit=i,
        nfev=nfev,
        njev=njev,
        nhev=nhev,
        fun=fun(x),
        jac=jac(x),
        message="Maximum number of iterations reached.",
    )


def projected_gradient_descent(
    func,
    grad,
    bounds,
    x0,
    tol=1e-6,
    max_iter=1000,
    ls_alpha=0.5,
    ls_beta=0.5,
    ls_max_iter=20,
):
    """
    Solves a bound-constrained minimization problem using projected gradient descent
    with a backtracking line search.

    Args:
        func: The objective function to minimize.
        grad: The gradient of the objective function.
        bounds: A tuple of two numpy arrays, (lower_bounds, upper_bounds).
        x0: The initial guess.
        tol: The tolerance for the stopping criterion.
        max_iter: The maximum number of iterations.
        ls_alpha: The initial step size for the line search.
        ls_beta: The backtracking factor for the line search.
        ls_max_iter: The maximum number of iterations for the line search.

    Returns:
        The optimal solution found.
    """
    x = np.copy(x0)
    lower_bounds, upper_bounds = bounds

    for i in range(max_iter):
        gradient = grad(x)
        alpha = ls_alpha  # Reset step size at each iteration

        # Backtracking line search
        for _ in range(ls_max_iter):
            x_new = x - alpha * gradient
            x_new = np.clip(x_new, lower_bounds, upper_bounds)

            # Check Armijo condition
            fcall = func(x)
            if func(x_new) <= fcall - 0.0001 * alpha * np.dot(gradient, gradient):
                break
            alpha *= ls_beta
        else:
            # Line search failed to find a suitable step size
            print("Line search failed. Using a small fixed step size.")
            alpha = 1e-5
            x_new = x - alpha * gradient
            x_new = np.clip(x_new, lower_bounds, upper_bounds)

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {i + 1} iterations.")
            return x_new

        x = x_new
        print(x, gradient, alpha, np.linalg.norm(gradient))

    print("Maximum number of iterations reached.")
    return x
