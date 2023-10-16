import numpy as np
from scipy import optimize

# local
import blocks
class Fonttype:
    HEADER = '\033[1m' + '\033[94m'
    END = '\033[0m'

def find_household_consumption_ss(model):
    """ find household behavior in steady state given """
    
    par = model.par
    ss = model.ss

    result = optimize.root_scalar(household_consumption_ss,
        bracket=[0.0001,1000],method='brentq',args=(par,ss,))
    
    household_consumption_ss(result.root,par,ss)
    ss.A_R_death = ss.A_R_a[-1]

    return result

def household_consumption_ss(A_R_death,par,ss):
    """ find household behavior in steady state given A_death """

    # a. income
    ss.inc_a[:] = (1-ss.tau)*ss.W*ss.LH_a/par.N_a + (1-ss.tau)*par.W_U*ss.W*ss.U_a/par.N_a + ss.Aq/par.N
    ss.inc_a[par.work_life_span:] += (1-ss.tau)*par.W_R*ss.W

    # a. HtM
    ss.C_HtM_a[:] = ss.inc_a/ss.P_C
    ss.A_HtM_a[:] = np.zeros(par.life_span)

    # b. Ricardian
    A_R_ini_error = np.nan
    ss.A_R_a[-1] = A_R_death

    for i in range(par.life_span):

        a = par.life_span-1-i

        # i. consumption
        RHS = par.zeta_a[a]*par.mu_Aq*(ss.A_R_a[a]/ss.P_C)**(-par.sigma)
        
        if a < par.life_span-1: 
            RHS += (1-par.zeta_a[a])*par.beta*(1+ss.real_r_hh)*ss.C_R_a[a+1]**(-par.sigma)

        ss.C_R_a[a] = RHS**(-1/par.sigma)

        # ii. assets
        A_R_lag = (ss.A_R_a[a] + ss.P_C*ss.C_R_a[a] - ss.inc_a[a])/(1+ss.r_hh)         
        if a > 0:
            ss.A_R_a[a-1] = A_R_lag
        else:
            A_R_ini_error = A_R_lag-0.0

    # c. aggregate
    ss.C_a = par.Lambda*ss.C_HtM_a+(1-par.Lambda)*ss.C_R_a 
    ss.A_a = par.Lambda*ss.A_HtM_a+(1-par.Lambda)*ss.A_R_a 

    ss.inc = np.sum(par.N_a*ss.inc_a)
    ss.C = np.sum(par.N_a*ss.C_a)
    ss.C_HtM = np.sum(par.N_a*ss.C_HtM_a)
    ss.C_R = np.sum(par.N_a*ss.C_R_a)
    ss.A = np.sum(par.N_a*ss.A_a)

    return A_R_ini_error

def find_Aq_ss(Aq,model):
    """ find Aq in steady state """

    par = model.par
    ss = model.ss

    # a. initial guess
    ss.Aq = Aq

    # b. iterate
    it = 0
    while True:

        old_Aq = ss.Aq

        # i. solve     
        find_household_consumption_ss(model)

        # ii. update
        ss.Aq = (1+ss.r_hh)*np.sum(par.zeta_a*par.N_a*ss.A_a)

        # iii. converged?
        if np.abs(ss.Aq-old_Aq) < 1e-12: 
            find_household_consumption_ss(model)
            Aq = (1+ss.r_hh)*np.sum(par.zeta_a*par.N_a*ss.A_a)
            ss.Aq_diff = Aq - old_Aq
            break

        it += 1
        if it > 100: raise ValueError(f'search for ss.Aq did not converge')

def household_search_ss(par,ss):
    """ find labor supply in steady state """

    for a in range(par.life_span):
        
        if a == 0:
            ss.S_a[a] = 1.0
            ss.L_ubar_a[a] = 0.0
            ss.x_a[a] = 0

        elif a >= par.work_life_span:
            ss.S_a[a] = 0.0
            ss.L_ubar_a[a] = 0.0
            ss.x_a[a] = 0
          
        else:
            ss.S_a[a] = (1-par.zeta_a[a])*((par.N_a[a-1]-ss.L_a[a-1]) + par.delta_L_a[a]*ss.L_a[a-1])          
            ss.L_ubar_a[a] = (1-par.zeta_a[a])*(1-par.delta_L_a[a])*ss.L_a[a-1]
            ss.x_a[a] = ss.x_a[a-1]+ss.L_a[a-1]/par.N_a[a-1]
        
        ss.H_a[a] = 1 + par.rho_1*ss.x_a[a] - par.rho_2*ss.x_a[a]**2


        ss.L_a[a] = ss.L_ubar_a[a] + ss.m_s*ss.S_a[a]
        ss.LH_a[a] = ss.L_a[a]*ss.H_a[a]
        
        if a >= par.work_life_span:
            ss.U_a[a] = 0.0
        else:
            ss.U_a[a] = par.N_a[a]-ss.L_a[a]

    ss.S = np.sum(par.N_a*ss.S_a)
    ss.L_ubar = np.sum(par.N_a*ss.L_ubar_a)
    ss.L = np.sum(par.N_a*ss.L_a)
    ss.LH = np.sum(par.N_a*ss.LH_a)
    ss.U = np.sum(par.N_a*ss.U_a)
    ss.H = ss.LH/ss.L

def find_ss(model,do_print=True):

    par = model.par
    ss = model.ss

    # Demographics:
    model.mortality()
    model.demographic_structure()
    model.job_separation_rate()

    # a. price noramlizations
    ss.P_Y = 1.0                                                
    ss.P_F = 1.0
    ss.P_M_C = 1.0
    ss.P_M_G = 1.0
    ss.P_M_I = 1.0
    ss.P_M_X = 1.0
    
    # b. fixed variables
    ss.W = par.W_ss
    ss.pi_hh = par.pi_hh_ss
    ss.m_s = par.m_s_ss
    ss.m_v = par.m_v_ss
    ss.B = par.B_ss
    ss.r_hh = par.r_hh

    # c. pricing in repacking firms
    ss.P_C = blocks.CES_P(ss.P_M_C,ss.P_Y,par.mu_M_C,par.sigma_C,Gamma=1)
    ss.P_G = blocks.CES_P(ss.P_M_G,ss.P_Y,par.mu_M_G,par.sigma_G,Gamma=1)
    ss.P_I = blocks.CES_P(ss.P_M_I,ss.P_Y,par.mu_M_I,par.sigma_I,Gamma=1)
    ss.P_X = blocks.CES_P(ss.P_M_X,ss.P_Y,par.mu_M_X,par.sigma_X,Gamma=1)

    # d. labor supply, search and matching
    household_search_ss(par,ss)

    ss.delta_L = (ss.L-ss.L_ubar)/ss.L
    ss.curlyM = ss.delta_L*ss.L
    
    ss.v = ss.curlyM/ss.m_v

    error_M = lambda sigma_m: ss.curlyM - ss.S*ss.v/(ss.S**(1/sigma_m)+ss.v**(1/sigma_m))**(sigma_m)
    
    result = optimize.root_scalar(error_M,
        bracket=[0.01,1],method='brentq')

    par.sigma_m = result.root

    assert result.converged

    if do_print:
        print(Fonttype.HEADER + 'Labor supply, search and matching:' + Fonttype.END)
        print(f'{ss.S/par.N_work = :.2f}, {ss.L/par.N_work = :.2f}, {ss.U/par.N_work = :.2f}')
        print(f'{ss.delta_L = :.2f}, {ss.m_s = :.2f}, {ss.m_v = :.2f}, {ss.v = :.2f}')
        print(f'{par.sigma_m = :.2f}')

    # e. capital agency FOC
    ss.r_K = (par.r_firm + par.delta_K)*ss.P_I
    ss.real_r_K = ss.r_K/ss.P_Y

    if do_print: 
        print(Fonttype.HEADER + 'Capital agency FOC:' + Fonttype.END)
        print(f'{ss.r_K = :.2f}')
    
    # f. labor agency FOC
    ss.r_ell = ss.W / (1-par.kappa_L/(ss.m_v*ss.H) + (1-ss.delta_L)/(1+par.r_firm)*par.kappa_L/(ss.m_v*ss.H))
    ss.real_r_ell = ss.r_ell/ss.P_Y
    ss.ell = ss.LH - par.kappa_L*ss.v

    if do_print: 
        print(Fonttype.HEADER + 'Labor agency FOC:' + Fonttype.END)
        print(f'{ss.r_ell = :.2f}, {(ss.LH-ss.ell)/par.N_work*100 = :.2f}')
    
    # g. production firm & phillips-curve
    ss.P_Y_0 = ss.P_Y/(1+par.theta)
    P_Y_0_NoGamma = blocks.CES_P(ss.r_K,ss.r_ell,par.mu_K,par.sigma_Y,Gamma=1.0)
    ss.Gamma = P_Y_0_NoGamma/ss.P_Y_0
    P_Y_0 = blocks.CES_P(ss.r_K,ss.r_ell,par.mu_K,par.sigma_Y,Gamma=ss.Gamma)
    assert np.isclose(P_Y_0,ss.P_Y_0)

    ss.K = par.mu_K/(1-par.mu_K)*(ss.r_ell/ss.r_K)**par.sigma_Y*ss.ell
    ss.Y = blocks.CES_Y(ss.K,ss.ell,par.mu_K,par.sigma_Y,Gamma=ss.Gamma)

    if do_print: 
        print(Fonttype.HEADER + 'Production firm:' + Fonttype.END)
        print(f'{ss.P_Y_0 = :.2f}, {ss.Gamma = :.2f}, {ss.Y = :.2f}, {ss.K = :.2f}')

    # h. capital accumulation
    ss.iota = ss.I = par.delta_K*ss.K
    
    if do_print: 
        print(Fonttype.HEADER + 'Capital accumulation:' + Fonttype.END)
        print(f'{ss.iota = :.2f}, {ss.I = :.2f}')

    # i. government
    ss.G = par.G_share_ss*ss.Y
    ss.tau = (par.r_b*ss.B+ss.P_G*ss.G+par.W_U*ss.U*ss.W+par.W_R*ss.W*(par.N-par.N_work))/(ss.W*ss.LH+par.W_U*ss.U*ss.W+par.W_R*ss.W*(par.N-par.N_work))     

    if do_print: 
        print(Fonttype.HEADER + 'Government:' + Fonttype.END)
        print(f'{ss.B = :.2f}, {ss.G = :.2f}, {ss.tau = :.2f}')

    # j. household behavior
    ss.real_W = ss.W/ss.P_C
    ss.real_r_hh = (1+ss.r_hh)/(1+ss.pi_hh)-1
    find_Aq_ss(0.0,model)
    ss.real_inc = ss.inc/ss.P_C
    ss.real_Aq = ss.Aq/ss.P_C

    if do_print:
        print(Fonttype.HEADER + 'Households:' + Fonttype.END)
        print(f'{ss.Aq/par.N = :.2f}, {ss.real_W = :.2f}, {ss.C = :.2f}, {ss.A = :.2f}, {ss.r_hh = :.2f}')

    # k. CES demand in packing firms
    ss.C_M = blocks.CES_demand(ss.P_M_C,ss.P_C,par.mu_M_C,ss.C,par.sigma_C,Gamma=1)
    ss.C_Y = blocks.CES_demand(ss.P_Y,ss.P_C,1-par.mu_M_C,ss.C,par.sigma_C,Gamma=1)

    ss.G_M = blocks.CES_demand(ss.P_M_G,ss.P_G,par.mu_M_G,ss.G,par.sigma_G,Gamma=1)
    ss.G_Y = blocks.CES_demand(ss.P_Y,ss.P_G,1-par.mu_M_G,ss.G,par.sigma_G,Gamma=1)

    ss.I_M = blocks.CES_demand(ss.P_M_I,ss.P_I,par.mu_M_I,ss.I,par.sigma_I,Gamma=1)
    ss.I_Y = blocks.CES_demand(ss.P_Y,ss.P_I,1-par.mu_M_I,ss.I,par.sigma_I,Gamma=1)

    # l. market clearing
    ss.X_Y = ss.Y - (ss.C_Y + ss.G_Y + ss.I_Y) 
    ss.X = ss.chi = ss.X_Y/(1-par.mu_M_X)
    ss.X_M = blocks.CES_demand(ss.P_M_X,ss.P_X,par.mu_M_X,ss.X,par.sigma_X,Gamma=1)
    
    ss.M = ss.C_M + ss.G_M + ss.I_M + ss.X_M

    if do_print: 

        print(Fonttype.HEADER + 'Market clearing:' + Fonttype.END)
        print(f'{ss.C/ss.Y = :.2f}, {ss.G/ss.Y = :.2f}, {ss.I/ss.Y = :.2f}, {ss.X/ss.Y = :.2f}, {ss.M/ss.Y = :.2f}')

    # m. ratios
    ss.C_ratio = ss.C/ss.Y
    ss.G_ratio = ss.G/ss.Y
    ss.I_ratio = ss.I/ss.Y
    ss.K_ratio = ss.K/ss.Y
    ss.L_ratio = ss.L/par.N
    ss.M_ratio = ss.M/ss.Y
    ss.X_ratio = ss.X/ss.Y