import time
import numpy as np
from scipy import sparse
import scipy.sparse.linalg

def check_convergence(y,tol=1e-8,do_print=False,model=None):

    abs_diff = np.max(np.abs(y))
    if do_print and abs_diff<tol: 
        
        print(f'\n    max. abs. error = {abs_diff:8.2e}')

        if not model is None:

            for target in model.targets:
                errors = model.sol.__dict__[target]
                print(f'    {np.max(np.abs(errors)):8.2e} in {target}')

    return abs_diff < tol

def broyden_solver(f,x0,jac,tol=1e-8,maxiter=100,do_print=False,model=None):
    """ numerical equation system solver using the broyden method """
    t0 = time.time()
    # a. initial
    x = x0.ravel()
    y = f(x)

   
    if do_print: print('initial:')
    converged = check_convergence(y,tol=tol,model=model,do_print=do_print)
    if converged: return x
    t1 = time.time()
    if do_print: print(f'{t1-t0 = :.5f} secs')

    # b. iterate
    for it in range(maxiter):

        if do_print: print(f'{it = }', end='\r')

        # i. new x
        t0 = time.time()
        dx = np.linalg.solve(jac,-y)
        x += dx
        t1 = time.time()

        # ii. evaluate
        t2 = time.time()
        ynew = f(x)
    
        # iii. update jac
        dy = ynew-y
        jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx)**2), dx)
        y = ynew
        
        converged = check_convergence(y,tol=tol,model=model,do_print=do_print)
        t3 = time.time()

        if do_print and converged: 
            print(f'\nsolve: {t1-t0 = :.5f} secs')
            print(f'evaluate + update: {t3-t2 = :.5f} secs')
        
        if converged: return x

    else:

        raise ValueError(f'no convergence after {maxiter} iterations') 

# @nb.njit(nogil=True)
def sparse_solver(f,x0,jac,tol=1e-8,maxiter=100,do_print=False,model=None):
    """ numerical equation system solver using sparse matrices """

    # a. initial
    x = x0.ravel()
    y = f(x)
    jacobian = sparse.csr_matrix(jac)

    if do_print: print('initial:')
    converged = check_convergence(y,tol=tol,model=model,do_print=do_print)
    if converged: return x

    dx = scipy.sparse.linalg.lgmres(jacobian,-y, x0=x, tol=tol, maxiter=maxiter)
    x += dx
    converged = check_convergence(y,tol=tol,model=model,do_print=do_print)

    # # b. iterate
    # for it in range(maxiter):

    #     if do_print: print(f'\n{it = }')

    #     # i. new x
    #     t0 = time.time()
    #     dx = scipy.sparse.linalg.minres(jacobian, -y)
    #     x += dx
    #     t1 = time.time()
        
    #     if do_print: print(f' solve: {t1-t0 = :.1f} secs')

    #     # ii. evaluate
    #     t0 = time.time()
    #     ynew = f(x)
    #     t1 = time.time()

    #     converged = check_convergence(y,tol=tol,model=model,do_print=do_print)
    #     if converged: return x

    #     if do_print: print(f' evaluate: {t1-t0 = :.1f} secs')

    #     # iii. update jac
    #     t0 = time.time()
    #     dy = ynew-y
    #     jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx)**2), dx)
    #     jacobian = sparse.csr_matrix(jac)
    #     y = ynew
    #     t1 = time.time()
        
    #     if do_print: print(f' update_jac: {t1-t0 = :.1f} secs')

    # else:

    #     raise ValueError(f'no convergence after {maxiter} iterations') 
