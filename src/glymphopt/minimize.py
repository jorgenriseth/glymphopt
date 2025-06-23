import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import OptimizeResult


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
    # --- Step 0: Initialization ---
    x = np.clip(x0, bounds.lb, bounds.ub)  # Ensure initial guess is feasible
    lb, ub = bounds.lb, bounds.ub

    # Counters for function/gradient/hessian evaluations
    nfev, njev, nhev = 0, 0, 0

    if verbose:
        print(f"{'Iter':>4} {'Objective':>13} {'Proj. Grad Norm':>18}")
        print("-" * 40)

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
            print(f"{i:4d} {f_val:13.6e} {projected_grad_norm:18.6e}")

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
            d = -projected_gradient
            g_dot_d = np.dot(g, d)
            alpha = 1.0  # Reset alpha for gradient step

        while fun(x + alpha * d) > f_val + c * alpha * g_dot_d:
            alpha /= 2.0
            nfev += 1
            if alpha < 1e-8:
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
                )

        # --- Step 5: Update Iterate ---
        x = x + alpha * d
        # A final clip for numerical safety, though theoretically not needed
        # due to the careful `alpha_max` calculation.
        x = np.clip(x, lb, ub)

    return OptimizeResult(
        x=x,
        success=False,
        nit=max_iter,
        nfev=nfev,
        njev=njev,
        nhev=nhev,
        fun=fun(x),
        jac=jac(x),
        message="Maximum number of iterations reached.",
    )
