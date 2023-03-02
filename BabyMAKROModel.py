import time
import numpy as np

from EconModel import EconModelClass, jit
from consav import elapsed

import matplotlib.pyplot as plt   
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

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
            'repacking_firms_prices',
            'wage_determination',
            'search_and_match',
            'labor_agency',
            'production_firm',
            'phillips_curve',
            'foreign_economy',
            'capital_agency',
            'government',
            'household_consumption',
            'repacking_firms_components',
            'goods_market_clearing',
            'real_productivity',
        ]
        
        # c. variable lists
        
        # exogenous variables
        self.exo = [
            'Gamma',
            'G',
            'chi',
            'P_M_C',
            'P_M_G',
            'P_M_I',
            'P_M_X',
            'P_F',
            'r_hh',
        ]
        
        # unknowns
        self.unknowns = [
            'Aq',
            'A_R_death',
            'K',
            'L',
            'r_K',
            'P_Y',
        ]

        # targets
        self.targets = [
            'A_R_ini_error',
            'Aq_diff',
            'FOC_capital_agency',
            'FOC_K_ell',
            'mkt_clearing',
            'PC',
        ]

        # all non-household variables
        self.varlist = [
            'A_R_death',
            'A_R_ini_error',
            'A',
            'Aq_diff',
            'Aq',
            'B',
            'C_M',
            'C_Y',
            'C',
            'C_HtM',
            'C_R',
            'chi',
            'curlyM',
            'delta_L',
            'ell',
            'FOC_C',
            'FOC_capital_agency',
            'FOC_K_ell',
            'G_M',
            'G_Y',
            'G',
            'Gamma',
            'I_M',
            'I_Y',
            'I',
            'inc',
            'iota',
            'K',
            'L_ubar',
            'L',
            'm_s',
            'm_v',
            'M',
            'mkt_clearing',            
            'N',
            'PC',
            'P_C',
            'P_F',
            'P_G',
            'P_I',
            'P_M_C',
            'P_M_G',
            'P_M_I',
            'P_M_X',
            'P_X',
            'P_Y',
            'P_Y_0',
            'pi_hh',
            'r_ell',
            'r_K',
            'r_hh',
            'real_Aq',
            'real_inc',
            'real_r_K',
            'real_r_ell',
            'real_W',
            'real_r_hh',
            'S',
            'tau',
            'U',
            'v',
            'W',
            'X_M',
            'X_Y',
            'X',
            'Y',
        ]

        # all household variables
        self.varlist_hh = [
            'A_a',
            'A_HtM_a',
            'A_R_a',
            'C_a',
            'C_HtM_a',
            'C_R_a',
            'inc_a',
            'L_a',
            'L_ubar_a',
            'S_a',
            'U_a',
        ]

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.T = 400 # number of time-periods
        
        # a. households
        par.age_ini = 25 # initial age in model
        par.life_span = 75 # maximum life-span
        par.work_life_span = 45 # work-life-span
        par.zeta = 4.0 # mortality parameter (-> inf then everybody dies in last period)
        par.Lambda = 0.30 # share of hands-to-mouth households

        par.beta = 0.95 # discount factor
        par.sigma = 2.0 # CRRA coefficient
        par.mu_Aq = 100 # weight on bequest motive

        par.r_hh = 0.04 # nominal return rate
        par.W_U = 0.80 # unemployment benefits (rel. to ss.W)
        par.W_R = 0.50 # retirement benefits (rel. to ss.W)

        par.delta_L_a_fac = 0.10 # age-specific separation rate (common)

        # b. production firm and phillips curve
        par.r_firm = 0.04 # internal (nominal) rate of return
        par.delta_K = 0.10 # depreciation rate
        par.mu_K = 1/3 # weigth on capital
        par.sigma_Y = 1.01 # substitution
        par.theta = 0.1 # mark-up
        par.eta = 0.1 # PC-slope

        # c. labor agency
        par.kappa_L = 0.05 # cost of vancies in labor units

        # d. capital agency
        par.Psi_0 = 5.0 # adjustment costs

        # e. government
        par.r_b = 0.04 # nominal rate of return on government debt
        par.epsilon_B = 0.20 # adjustment speed  
        par.G_share_ss = 0.30 # share of government spending in Y

        # f. repacking
        par.mu_M_C = 0.30 # weight on imports in C
        par.sigma_C = 1.5 # substitution
        par.mu_M_G = 0.10 # weight on imports in G
        par.sigma_G = 1.5 # substitution
        par.mu_M_I = 0.35 # weight on imports in I
        par.sigma_I = 1.5 # substitution
        par.mu_M_X = 0.40 # weight on imports in X
        par.sigma_X = 1.5 # substitution

        # g. foreign
        par.sigma_F = 1.5 # substitution in export demand
        par.gamma_X = 0.50 # export persistence

        # h. matching
        par.sigma_m = np.nan # curvature in matching function, determined in ss

        # i. steady state
        par.W_ss = 1.0 # wage
        par.pi_hh_ss = 0.00 # inflation
        par.m_s_ss = 0.75 # job-finding rate
        par.m_v_ss = 0.75 # job-filling rate
        par.B_ss = 0.0 # government debt

    def mortality(self):
        """ calculate mortality by age """

        par = self.par

        par.zeta_a = np.zeros(par.life_span)
        par.zeta_a[-1] = 1.0 # everybody dies in last period

        for a in range(par.life_span-1):
            if a < par.work_life_span: # no death before retirement
                par.zeta_a[a] = 0.0
            else:
                par.zeta_a[a] = ((a+1-par.work_life_span)/(par.life_span-par.work_life_span))**par.zeta
    
    def demographic_structure(self):
        """ calculate demographic structure """
        
        par = self.par

        par.N_a = np.zeros(par.life_span)
        par.N_a[0] = 1.0 # normalization
             
        for a in range(1,par.life_span):
            par.N_a[a] = (1-par.zeta_a[a-1])*par.N_a[a-1]
                    
        par.N = np.sum(par.N_a)
        par.N_work = np.sum(par.N_a[:par.work_life_span])

    def job_separation_rate(self):
        """ calcualte job-sepration rate by age"""
    
        par = self.par
        
        par.delta_L_a = par.delta_L_a_fac*np.ones(par.work_life_span)

    def allocate(self):
        """ allocate model """

        par = self.par
        ini = self.ini
        ss = self.ss
        sol = self.sol

        # a. demographics
        
        # mortality
        self.mortality()

        # demographic structure
        self.demographic_structure()    

        # job-separation rate
        self.job_separation_rate()    

        # b. non-household variables
        for varname in self.varlist:
            setattr(ini,varname,np.nan)
            setattr(ss,varname,np.nan)
            setattr(sol,varname,np.zeros(par.T))

        for varname in self.exo: assert varname in self.varlist, varname

        # c. household variables
        for varname in self.varlist_hh:
            setattr(ini,varname,np.zeros(par.life_span))
            setattr(ss,varname,np.zeros(par.life_span))
            setattr(sol,varname,np.zeros((par.life_span,par.T)))            

        for varname in self.unknowns: assert varname in self.varlist+self.varlist_hh, varname
        for varname in self.targets: assert varname in self.varlist+self.varlist_hh, varname

    ################
    # steady state #
    ################
    
    def find_ss(self,do_print=False):
        """ find steady state """

        steady_state.find_ss(self,do_print=do_print)

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
                sol.__dict__[varname] = np.zeros((par.life_span,par.T))
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

    def evaluate_block(self,block,py=False):

        with jit(self) as model: # use jit for faster evaluation

            if not hasattr(blocks,block): raise ValueError(f'{block} is not a valid block')
            func = getattr(blocks,block)

            if py: # python version for debugging
                func.py_func(model.par,model.ini,model.ss,model.sol)
            else:
                func(model.par,model.ini,model.ss,model.sol)

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
        for block in self.blocks:

            self.evaluate_block(block,py=py)
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

        if do_print: print(f'Jacobian calculated in {elapsed(t0)}')

    def find_IRF(self,ini=None,do_print=True):
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
        broyden_solver(obj,x0,self.jac,tol=1e-10,maxiter=100,do_print=do_print,model=self)

    #################
    # basic figures #
    #################

    def plot_IRF(self,varlist,ncol=3,T_IRF=60, abs = None,Y_share = None):
        """ plot IRFs """

        if abs is None:
            abs = []
        if Y_share is None:
            Y_share = []
                
        ss = self.ss
        sol = self.sol

        nrow = len(varlist)//ncol
        if len(varlist) > nrow*ncol: nrow+=1

        fig = plt.figure(figsize=(ncol*6,nrow*6/1.5))
        for i,varname in enumerate(varlist):

            ax = fig.add_subplot(nrow,ncol,1+i)

            path = sol.__dict__[varname]
            ssvalue = ss.__dict__[varname]

            if varname in abs:
                ax.axhline(ssvalue,color='black')
                ax.plot(path[:T_IRF],'-o',markersize=3)
            elif varname in Y_share:
                ax.plot(path[:T_IRF]/sol.Y[:T_IRF],'-o',markersize=3)   
                ax.set_ylabel('share of Y')         
            elif np.isclose(ssvalue,0.0):
                ax.plot(path[:T_IRF]-ssvalue,'-o',markersize=3)
                ax.set_ylabel('diff.to ss')
            else:
                ax.plot((path[:T_IRF]/ssvalue-1)*100,'-o',markersize=3)
                ax.set_ylabel('% diff.to ss')

            ax.set_title(varname)

        fig.tight_layout(pad=1.0)
    
    def plot_IRF_hh(self,varlist,t0_list=None,ncol=2):
        """ plot IRFs for household variables """

        par = self.par
        ss = self.ss
        sol = self.sol

        if t0_list is None: t0_list = [-par.life_span+1,0,par.life_span]

        nrow = len(varlist)//ncol
        if len(varlist) > nrow*ncol: nrow+=1

        fig = plt.figure(figsize=(ncol*6,nrow*6/1.5))

        for i,varname in enumerate(varlist):

            ax = fig.add_subplot(nrow,ncol,1+i)

            for t0 in t0_list:

                t_beg = np.fmax(t0,0)
                t_end = t0 + par.life_span-1

                y = np.zeros(t_end-t_beg)
                for j,t in enumerate(range(t_beg,t_end)):
                    a = t-t0
                    y[j] = sol.__dict__[varname][a,t]-ss.__dict__[varname][a]
                    
                ax.plot(par.age_ini+np.arange(t_beg-t0,t_end-t0),y,label=f'$t_0$ = {t0}')
                ax.set_xlabel('age')
                ax.set_ylabel('diff to ss')
                ax.set_title(varname)

            if i == 0:
                ax.legend(frameon=True)

        fig.tight_layout(pad=1.0)

    ################
    # multi-models #
    ################

    def multi_model(self,parameter,parvalues):
        """ create multiple models with different parameters """
        
        par = self.par
        models = []  

        for parvalue in parvalues:

            model_ = self.copy()
            setattr(model_.par,parameter,parvalue)
            model_.find_ss()
            model_.calc_jac(do_print=True)

            models.append(model_)

        return models

    def plot_IRF_models(self,models,parameter,varlist,ncol=3,T_IRF=50,abs = None,Y_share = None):
        """ plot IRFs """

        if abs is None:
            abs = []
        if Y_share is None:
            Y_share = []

        nrow = len(varlist)//ncol
        if len(varlist) > nrow*ncol: nrow+=1 

        fig = plt.figure(figsize=(ncol*6,nrow*6/1.5))

        for i,varname in enumerate(varlist):
            
            ax = fig.add_subplot(nrow,ncol,1+i)
            for model_ in models:

                par = model_.par
                ss = model_.ss
                sol = model_.sol

                parvalue = par.__dict__[parameter]
                ssvalue = ss.__dict__[varname]
                path = sol.__dict__[varname]

                if varname in abs:
                    ax.axhline(ssvalue,color='black',linewidth=0.75)
                    ax.plot(path[:T_IRF],'-o',markersize=2)
                elif varname in Y_share:
                    ax.plot(path[:T_IRF]/sol.Y[:T_IRF],'-o',markersize=2, label=f'{parameter} = {parvalue}',linewidth=0.75)   
                    ax.set_ylabel('share of Y')         
                elif np.isclose(ssvalue,0.0):
                    ax.plot(path[:T_IRF]-ssvalue,'-o',markersize=2, label=f'{parameter} = {parvalue}',linewidth=0.75)
                    ax.set_ylabel('diff.to ss')
                else:
                    ax.plot((path[:T_IRF]/ssvalue-1)*100,'-o',markersize=2, label=f'{parameter} = {parvalue}',linewidth=0.75)
                    ax.set_ylabel('% diff.to ss')
                
                handles, labels = ax.get_legend_handles_labels()
            
            ax.set_title(varname)
            fig.legend(handles,labels,loc='upper right',frameon=True)

        fig.tight_layout(pad=1.0)

    ######################
    # multi-shock-models #
    ######################

    def multi_shock_model(self,Tshock,persistence,shock1_values,shock2_values,shock1_size=0.01,shock2_size=0.005, shock1 = None, shock2 = None):
        """ create multiple models with different shocks """
        
        if shock1 is None:
            shock1 = []
            shock1.extend(shock1_values)
        if shock2 is None:
            shock2 = []
            shock2.extend(shock2_values)
        
        shocks = [shock1,shock2]
        shock_size = [shock1_size,shock2_size]
        
        ss = self.ss
        sol = self.sol
        
        self.find_ss()
        self.calc_jac(do_print=True)
        jac = self.jac
        self.set_exo_ss()
        
        for i,shock_index in enumerate(shocks):
            for shock_ in shock_index:
                ss_shock_ = getattr(ss, shock_)
                sol_shock_ = getattr(sol, shock_)
                var_shock_ = shock_size[i]*ss_shock_
                sol_shock_[:Tshock] = ss_shock_ + var_shock_*persistence    

        self.find_IRF()   
        
        models = [self]  

        for i, shock_index in enumerate(shocks):
            model_ = self.copy()
            ss = model_.ss
            sol = model_.sol
            model_.find_ss()
            model_.jac = jac.copy()
            model_.set_exo_ss()

            for shock_ in shock_index:
                ss_shock_ = getattr(ss, shock_)
                sol_shock_ = getattr(sol, shock_)
                var_shock_ = shock_size[i]*ss_shock_
                sol_shock_[:Tshock] = ss_shock_ + var_shock_*persistence

            model_.find_IRF()
            models.append(model_)

        return models    

    def multi_shock_IRF(self,models,shocks,varlist,ncol=3,T_IRF=50,abs = None,Y_share = None):
        """ plot IRFs """

        if abs is None:
            abs = []
        if Y_share is None:
            Y_share = []

        nrow = len(varlist)//ncol
        if len(varlist) > nrow*ncol: nrow+=1 

        fig = plt.figure(figsize=(ncol*6,nrow*6/1.5))
        for i,varname in enumerate(varlist):
            ss = []
            sol = []
            path = []
            ssvalue = []
            
            ax = fig.add_subplot(nrow,ncol,1+i)
            for j, model in enumerate(models):
                ss.append(model.ss)
                sol.append(model.sol)
                path.append(sol[j].__dict__[varname])
                ssvalue.append(ss[j].__dict__[varname])

                if varname in abs:
                    ax.axhline(ssvalue[j],color='black', label=f'Shock to {shocks[j]}',linewidth=0.75)
                    ax.plot(path[j][:T_IRF],'-o',markersize=2)
                elif varname in Y_share:
                    ax.plot(path[j][:T_IRF]/sol[j].Y[:T_IRF],'-o',markersize=2, label=f'Shock to {shocks[j]}',linewidth=0.75)   
                    ax.set_ylabel('share of Y')         
                elif np.isclose(ssvalue[j],0.0):
                    ax.plot(path[j][:T_IRF]-ssvalue[j],'-o',markersize=2, label=f'Shock to {shocks[j]}',linewidth=0.75)
                    ax.set_ylabel('diff.to ss')
                else:
                    ax.plot((path[j][:T_IRF]/ssvalue[j]-1)*100,'-o',markersize=2, label=f'Shock to {shocks[j]}',linewidth=0.75)
                    ax.set_ylabel('% diff.to ss')
                handles, labels = ax.get_legend_handles_labels()
            ax.set_title(varname)
            fig.legend(handles, labels, loc='upper right', frameon = True)

        fig.tight_layout(pad=1.0)        