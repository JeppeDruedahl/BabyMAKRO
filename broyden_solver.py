import time
import numpy as np

def check_convergence(y,tol=1e-8,do_print=False,model=None):

    abs_diff = np.max(np.abs(y))
    if do_print: 
        
        print(f'   max. abs. error = {abs_diff:8.2e}')

        if not model is None:

            for target in model.targets:
                errors = model.sol.__dict__[target]
                print(f'    {np.max(np.abs(errors)):8.2e} in {target}')

    return abs_diff < tol

def broyden_solver(f,x0,jac,tol=1e-8,maxiter=100,do_print=False,model=None):
    """ numerical equation system solver using the broyden method """

    # a. initial
    x = x0.ravel()
    y = f(x)

    if do_print: print('initial:')
    converged = check_convergence(y,tol=tol,model=model,do_print=do_print)
    if converged: return x

    # b. iterate
    for it in range(maxiter):

        if do_print: print(f'\n{it = }')

        # i. new x
        t0 = time.time()
        dx = np.linalg.solve(jac,-y)
        x += dx
        t1 = time.time()
        
        if do_print: print(f' solve: {t1-t0 = :.1f} secs')

        # ii. evaluate
        t0 = time.time()
        ynew = f(x)
        t1 = time.time()

        converged = check_convergence(y,tol=tol,model=model,do_print=do_print)
        if converged: return x

        if do_print: print(f' evaluate: {t1-t0 = :.1f} secs')

        # iii. update jac
        t0 = time.time()
        dy = ynew-y
        jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx)**2), dx)
        y = ynew
        t1 = time.time()
        
        if do_print: print(f' update_jac: {t1-t0 = :.1f} secs')

    else:

        raise ValueError(f'no convergence after {maxiter} iterations') 