from multiprocessing.sharedctypes import Value
import time
import numpy as np

from EconModel import EconModelClass, jit
from consav import elapsed

# local
import blocks
import steady_state
from broyden_solver import broyden_solver

class BabyMAKROModelClass(EconModelClass):    

    # This is the BabyMAKROModelClass
    # It builds on the EconModelClass -> read the documentation

    # in .settings() you must specify some variable lists
    # in .setup() you choose parameters
    # in .allocate() all variables are automatically allocated

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','sol']

        # b. blocks
        self.blocks = [
            'household_search',
            'labor_agency',
            'production_firm',
            'bargaining',
            'repacking_firms_prices',
            'foreign_economy',
            'capital_agency',
            'households_consumption',
            'repacking_firms_components',
            'goods_market_clearing',
        ]
        
        # c. variable lists
        
        # exogenous variables
        self.exo = [
            'chi',
            'P_F',
            'P_M_C',
            'P_M_I',
            'P_M_X',
        ]
        
        # unknowns
        self.unknowns = [
            'Bq',
            'K',
            'L',
            'r_K',
            'w',
        ]

        # targets
        self.targets = [
            'bargaining_cond',
            'Bq_match',
            'FOC_capital_agency',
            'FOC_K_ell',
            'mkt_clearing',
        ]

        # all non-household variables
        self.varlist = [
            'B',
            'bargaining_cond',
            'Bq_match',
            'Bq',
            'C_M',
            'C_Y',
            'C',
            'chi',
            'curlyM',
            'delta_L',
            'ell',
            'FOC_C',
            'FOC_capital_agency',
            'FOC_K_ell',
            'I_M',
            'I_Y',
            'I',
            'iota',
            'K',
            'L_ubar',
            'L',
            'm_s',
            'm_v',
            'M',
            'mkt_clearing',            
            'MPL',
            'P_C',
            'P_F',
            'P_I',
            'P_M_C',
            'P_M_I',
            'P_M_X',
            'P_X',
            'P_Y',
            'pi_hh',
            'r_ell',
            'r_K',
            'S',
            'v',
            'w_ast',
            'w',
            'X_M',
            'X_Y',
            'X',
            'Y',
        ]

        # all household variables
        self.varlist_hh = [
            'B_a',
            'C_a',
            'FOC_C',
            'L_a',
            'L_ubar_a',
            'S_a',
        ]

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.T = 500 # number of time-periods
        
        # a. households
        par.A = 80 # life-span
        par.A_R = 60 # work-life-span
        par.beta = 0.95 # discount factor
        par.sigma = 2.0 # CRRA coefficient
        par.mu_B = 2.5 # weight on bequest motive
        par.r_hh = 0.04 # nominal return rate
        par.delta_L_a = 0.05*np.ones(par.A_R) # separation probabilities
        par.w_U = 0.25 # outside option in bargaining

        # b. production firm
        par.r_firm = 0.04 # internal rate of return
        par.delta_K = 0.10 # discount factor
        par.mu_K = 1/3 # weigth on capital
        par.sigma_Y = 1.01 # substitution

        # c. labor agency
        par.kappa_L = 0.025

        # d. capital agency
        par.Psi_0 = 0.5 # adjustment costs

        # e. repacking
        par.mu_M_C = 0.30 # weight on imports in C
        par.sigma_C = 1.5 # substitution
        par.mu_M_I = 0.35 # weight on imports in I
        par.sigma_I = 1.5 # substitution
        par.mu_M_X = 0.40 # weight on imports in X
        par.sigma_X = 1.5 # substitution

        # f. foreign
        par.sigma_F = 1.5 # substitution in export demand

        # g. matching
        par.sigma_m = 1.5 # curvature

        # h. bargaining
        par.gamma_w = 0.75 # wage persistence
        par.phi = np.nan # bargaining power of firms (determined when finding steady state)

    def allocate(self):
        """ allocate model """

        par = self.par
        ini = self.ini
        ss = self.ss
        sol = self.sol

        # a. non-household variables
        for varname in self.varlist:
            setattr(ini,varname,np.nan)
            setattr(ss,varname,np.nan)
            setattr(sol,varname,np.zeros(par.T))

        for varname in self.exo: assert varname in self.varlist, varname

        # b. household variables
        for varname in self.varlist_hh:
            setattr(ini,varname,np.zeros(par.A))
            setattr(ss,varname,np.zeros(par.A))
            setattr(sol,varname,np.zeros((par.A,par.T)))            

        for varname in self.unknowns: assert varname in self.varlist+self.varlist_hh, varname
        for varname in self.targets: assert varname in self.varlist+self.varlist_hh, varname

    ################
    # steady state #
    ################
    
    def find_ss(self,m_s,do_print=False):
        """ find steady state """

        steady_state.find_ss(self.par,self.ss,m_s,do_print=do_print)

    #################
    # set functions #
    #################

    # functions for setting and getting variables
    
    def set_ss(self,varlist):
        """ set variables in varlist to steady state """

        par = self.par
        sol = self.sol
        ss = self.ss

        for varname in varlist:

            ssvalue = ss.__dict__[varname]

            if varname in self.varlist:
                sol.__dict__[varname] = np.repeat(ssvalue,par.T)
            elif varname in self.varlist_hh:
                sol.__dict__[varname] = np.zeros((par.A,par.T))
                for t in range(par.T):
                    sol.__dict__[varname][:,t] = ssvalue
            else:
                raise ValueError(f'unknown variable name, {varname}')

    def set_exo_ss(self):
        """ set exogenous variables to steady state """

        self.set_ss(self.exo)

    def set_unknowns_ss(self):
        """ set unknowns to steady state """

        self.set_ss(self.unknowns)

    def set_unknowns(self,x):
        """ set unknowns """

        sol = self.sol

        i = 0
        for unknown in self.unknowns:
            n = sol.__dict__[unknown].size
            sol.__dict__[unknown].ravel()[:] = x[i:i+n]
            i += n
    
    def get_errors(self,do_print=False):
        """ get errors in target equations """

        sol = self.sol

        errors = np.array([])
        for target in self.targets:

            errors_ = sol.__dict__[target]
            errors = np.hstack([errors,errors_.ravel()])

            if do_print: print(f'{target:20s}: abs. max = {np.abs(errors_).max():8.2e}')

        return errors

    ############
    # evaluate #
    ############

    def evaluate_blocks(self,ini=None,do_print=False,py=False):
        """ evaluate all blocks """

        # a. initial conditions
        if ini is None: # initial conditions are from steady state
            for varname in self.varlist: self.ini.__dict__[varname] = self.ss.__dict__[varname] 
            for varname in self.varlist_hh: self.ini.__dict__[varname] = self.ss.__dict__[varname].copy() 
        else: # initial conditions are user determined
            for varname in self.varlist: self.ini.__dict__[varname] = ini.__dict__[varname] 
            for varname in self.varlist_hh: self.ini.__dict__[varname] = ini.__dict__[varname].copy() 

        # b. evaluate
        with jit(self) as model: # use jit for faster evaluation

            for block in self.blocks:
                
                if not hasattr(blocks,block): raise ValueError(f'{block} is not a valid block')
                func = getattr(blocks,block)

                if py: # python version for debugging
                    func.py_func(model.par,model.ini,model.ss,model.sol)
                else:
                    func(model.par,model.ini,model.ss,model.sol)

                if do_print: print(f'{block} evaluated')
    
    ########
    # IRFs #
    ########
    
    def calc_jac(self,do_print=False,dx=1e-4):
        """ calculate Jacobian arround steady state """

        t0 = time.time()

        sol = self.sol

        # a. baseline
        self.set_exo_ss()
        self.set_unknowns_ss()
        self.evaluate_blocks()

        base = self.get_errors()

        x_ss = np.array([])
        for unknown in self.unknowns:
            x_ss = np.hstack([x_ss,sol.__dict__[unknown].ravel()])

        # b. allocate
        jac = self.jac = np.zeros((x_ss.size,x_ss.size))

        # c. calculate
        for i in range(x_ss.size):
            
            x = x_ss.copy()
            x[i] += dx

            self.set_unknowns(x)
            self.evaluate_blocks()
            alt = self.get_errors()
            jac[:,i] = (alt-base)/dx

        if do_print: print(f'Jacobian calculated in {elapsed(t0)} secs')

    def find_IRF(self,ini=None):
        """ find IRF """

        sol = self.sol

        # a. set initial guess
        self.set_unknowns_ss()

        x0 = np.array([])
        for unknown in self.unknowns:
            x0 = np.hstack([x0,sol.__dict__[unknown].ravel()])

        # b. objective
        def obj(x):
            
            # i. set unknowns from x
            self.set_unknowns(x)

            # ii. evaluate
            self.evaluate_blocks(ini=ini)

            # iii. get and return errors
            return self.get_errors()

        # c. solver
        broyden_solver(obj,x0,self.jac,tol=1e-10,maxiter=100,do_print=True,model=self)
