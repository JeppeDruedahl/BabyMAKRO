from os import P_OVERLAY
import numpy as np
import numba as nb

#######################
# auxiliary functions #
#######################

@nb.njit
def lag(ssvalue,pathvalue):
    return np.hstack((np.array([ssvalue]),pathvalue[:-1]))

@nb.njit
def lead(pathvalue,ssvalue):
    return np.hstack((pathvalue[1:],np.array([ssvalue])))

@nb.njit
def CES_Y(Xi,Xj,mui,sigma,Gamma=1.0):

    muj = 1-mui
    inv_sigma = 1/sigma
    pow_sigma = (sigma-1)/sigma
    inv_pow_sigma = sigma/(sigma-1)

    part_i = mui**inv_sigma*Xi**pow_sigma
    part_j = muj**inv_sigma*Xj**pow_sigma

    return Gamma*(part_i+part_j)**inv_pow_sigma

@nb.njit
def CES_demand(Pi,P,mui,X,sigma,Gamma=1.0):

    return mui*(Pi/P)**(-sigma)*X*Gamma**(sigma-1)
    
@nb.njit
def CES_P(Pi,Pj,mui,sigma,Gamma=1.0):

    muj = 1-mui
    
    part_i = mui*Pi**(1-sigma)
    part_j = muj*Pj**(1-sigma)

    return 1/Gamma*(part_i+part_j)**(1/(1-sigma))

@nb.njit
def adj_cost(iota,K_lag,Psi_0,delta_K):

    return 0.5*Psi_0*(iota/K_lag-delta_K)**2*K_lag

@nb.njit
def adj_cost_iota(iota,K_lag,Psi_0,delta_K):

    return Psi_0*(iota/K_lag-delta_K)

@nb.njit
def adj_cost_K(iota,K_lag,Psi_0,delta_K):

    return 0.5*Psi_0*(iota/K_lag-delta_K)**2 - Psi_0*(iota/K_lag-delta_K)*iota/K_lag

##########
# blocks #
##########

@nb.njit
def repacking_firms_prices(par,ini,ss,sol):

    # inputs
    P_M_C = sol.P_M_C
    P_M_G = sol.P_M_G 
    P_M_I = sol.P_M_I
    P_M_X = sol.P_M_X
    P_Y = sol.P_Y

    # outputs
    P_C = sol.P_C
    P_G = sol.P_G
    P_I = sol.P_I
    P_X = sol.P_X

    P_C[:] = CES_P(P_M_C,P_Y,par.mu_M_C,par.sigma_C, Gamma=1)
    P_G[:] = CES_P(P_M_G,P_Y,par.mu_M_G,par.sigma_G, Gamma=1)
    P_I[:] = CES_P(P_M_I,P_Y,par.mu_M_I,par.sigma_I, Gamma=1)
    P_X[:] = CES_P(P_M_X,P_Y,par.mu_M_X,par.sigma_X, Gamma=1)

@nb.njit
def wage_determination(par,ini,ss,sol):

    # inputs
    P_C = sol.P_C

    # outputs
    W = sol.W

    real_wage_ss = ss.W/ss.P_C
    W[:] = real_wage_ss*P_C

@nb.njit
def search_and_match(par,ini,ss,sol):

    # inputs
    L = sol.L

    # outputs
    curlyM = sol.curlyM
    delta_L = sol.delta_L
    L_a = sol.L_a
    L_ubar = sol.L_ubar
    L_ubar_a = sol.L_ubar_a
    m_s = sol.m_s
    m_v = sol.m_v
    S = sol.S
    S_a = sol.S_a
    U = sol.U
    U_a = sol.U_a
    v = sol.v

    # evaluations
    for t in range(par.T):
        
        # a. lagged employment
        if t == 0:
            L_lag = ini.L
        else:
            L_lag = L[t-1]

        # b. searchers and employed before matching
        for a in range(par.life_span):

            if a == 0:
            
                S_a[a,t] = 1.0
                L_ubar_a[a,t] = 0.0

            elif a >= par.work_life_span:
            
                S_a[a,t] = 0.0
                L_ubar_a[a,t] = 0.0

            else:

                if t == 0:
                    L_a_lag = ini.L_a[a-1]
                else:
                    L_a_lag = L_a[a-1,t-1]

                S_a[a,t] = (1-par.zeta_a[a])*((par.N_a[a-1]-L_a_lag) + par.delta_L_a[a]*L_a_lag)
                L_ubar_a[a,t] = (1-par.zeta_a[a])*((1-par.delta_L_a[a])*L_a_lag)

        S[t] = np.sum(par.N_a*S_a[:,t])
        L_ubar[t] = np.sum(par.N_a*L_ubar_a[:,t])

        # c. aggregate separation rate
        delta_L[t] = (L_lag-L_ubar[t])/L_lag

        # d. matching
        curlyM[t] = L[t]-L_ubar[t]
        m_s[t] = curlyM[t]/S[t]
        v[t] = (curlyM[t]**(1/par.sigma_m)/(1-m_s[t]**(1/par.sigma_m)))**par.sigma_m
        m_v[t] = curlyM[t]/v[t]

        # e. emplolyment and unemployment
        for a in range(par.life_span):

            L_a[a,t] = L_ubar_a[a,t] + m_s[t]*S_a[a,t]

            if a < par.work_life_span:
                U_a[a,t] = par.N_a[a]-L_a[a,t]            
            else:
                U_a[a,t] = 0.0

        U[t] = np.sum(par.N_a*U_a[:,t])

@nb.njit
def labor_agency(par,ini,ss,sol):

    # inputs
    delta_L = sol.delta_L
    L = sol.L
    m_v = sol.m_v
    v = sol.v
    W = sol.W

    # outputs
    ell = sol.ell
    r_ell = sol.r_ell
    
    # evaluations
    ell[:] = L-par.kappa_L*v

    for k in range(par.T):

        t = par.T-1-k

        if k == 0:
            r_ell_plus = ss.r_ell
            delta_L_plus = ss.delta_L
            m_v_plus = ss.m_v
        else:
            r_ell_plus = r_ell[t+1]
            delta_L_plus = delta_L[t+1]
            m_v_plus = m_v[t+1]
        
        fac = 1/(1-par.kappa_L/m_v[t])
        term = r_ell_plus*(1-delta_L_plus)/(1+par.r_firm)*par.kappa_L/m_v_plus

        r_ell[t] = fac*(W[t]-term)

@nb.njit
def production_firm(par,ini,ss,sol):

    # inputs
    ell = sol.ell
    Gamma = sol.Gamma
    K = sol.K
    r_K = sol.r_K
    r_ell = sol.r_ell

    # outputs
    P_Y_0 = sol.P_Y_0
    Y = sol.Y

    # targets
    FOC_K_ell = sol.FOC_K_ell

    # evaluations
    K_lag = lag(ini.K,K)

    Y[:] = CES_Y(K_lag,ell,par.mu_K,par.sigma_Y,Gamma=Gamma)
    P_Y_0[:] = CES_P(r_K,r_ell,par.mu_K,par.sigma_Y,Gamma=Gamma)

    FOC_K_ell[:] = K_lag/ell - par.mu_K/(1-par.mu_K)*(r_ell/r_K)**par.sigma_Y

@nb.njit
def phillips_curve(par,ini,ss,sol):

    # inputs
    P_Y_0 = sol.P_Y_0
    Y = sol.Y

    # outputs
    P_Y = sol.P_Y

    # targets
    PC = sol.PC

    # evaluations
    P_Y_lag = lag(ini.P_Y,P_Y)
    P_Y_lag_lag = lag(ini.P_Y,P_Y_lag)
    P_Y_plus = lead(P_Y,ss.P_Y)
    Y_plus = lead(Y,ss.Y)
 
    LHS = P_Y

    RHS_0 = (1+par.theta)*P_Y_0

    fac_lag = P_Y/P_Y_lag/(P_Y_lag/P_Y_lag_lag)
    RHS_1 = -par.eta*(fac_lag-1)*fac_lag*P_Y

    fac = P_Y_plus/P_Y/(P_Y/P_Y_lag)
    RHS_2 = 2/(1+par.r_firm)*par.eta*Y_plus/Y*(fac-1)*fac*P_Y_plus

    PC[:] = LHS - RHS_0 - RHS_1 - RHS_2

@nb.njit
def foreign_economy(par,ini,ss,sol):

    # inputs
    P_F = sol.P_F
    chi = sol.chi
    P_X = sol.P_X

    # outputs
    X = sol.X
    
    # evaluations
    for t in range(par.T):

        if t == 0:
            X_lag = ini.X
        else:
            X_lag = X[t-1]

        X[:] = par.gamma_X*X_lag + (1-par.gamma_X)*chi*(P_X/P_F)**(-par.sigma_F)
        
@nb.njit
def capital_agency(par,ini,ss,sol):

    # inputs
    K = sol.K
    P_I = sol.P_I
    r_K = sol.r_K

    # outputs
    I = sol.I
    iota = sol.iota

    # targets
    FOC_capital_agency = sol.FOC_capital_agency

    # evaluations
    K_lag = lag(ini.K,K)
    P_I_plus = lead(P_I,ss.P_I)
    r_K_plus = lead(r_K,ss.r_K)

    iota[:] = K - (1-par.delta_K)*K_lag
    I[:] = iota + adj_cost(iota,K_lag,par.Psi_0,par.delta_K)

    iota_plus = lead(iota,ss.iota)

    term_a = -P_I*(1+adj_cost_iota(iota,K_lag,par.Psi_0,par.delta_K))
    term_b = (1-par.delta_K)*P_I_plus*(1+adj_cost_iota(iota_plus,K,par.Psi_0,par.delta_K))
    term_c = -P_I_plus*adj_cost_K(iota_plus,K,par.Psi_0,par.delta_K)
    
    FOC_capital_agency[:] = term_a + 1/(1+par.r_firm)*(r_K_plus + term_b + term_c)

@nb.njit
def government(par,ini,ss,sol):

    # inputs
    G = sol.G
    L = sol.L
    P_G = sol.P_G
    U = sol.U
    W = sol.W

    # outputs
    B = sol.B
    tau = sol.tau

    # evaluations
    for t in range(par.T):

        if t == 0:
            B_lag = ini.B
        else:
            B_lag = B[t-1]
        
        expenditure = par.r_b*B_lag + P_G[t]*G[t] + par.W_U*ss.W*U[t] + par.W_R*ss.W*(par.N-par.N_work)
        taxbase =  W[t]*L[t] + par.W_U*ss.W*U[t] + par.W_R*ss.W*(par.N-par.N_work)

        B_tilde = B_lag + expenditure - ss.tau*taxbase
        tau[t] = ss.tau + par.epsilon_B*(B_tilde-ss.B)/taxbase

        B[t] = B_lag + expenditure - tau[t]*taxbase

@nb.njit
def household_consumption(par,ini,ss,sol):    

    # inputs
    A_R_death = sol.A_R_death
    Aq = sol.Aq
    L_a = sol.L_a
    P_C = sol.P_C
    P_Y = sol.P_Y
    tau = sol.tau
    U_a = sol.U_a
    W = sol.W

    # outputs
    A = sol.A
    A_a = sol.A_a
    A_HtM_a = sol.A_HtM_a
    A_R_a = sol.A_R_a
    C = sol.C
    C_a = sol.C_a
    C_HtM = sol.C_HtM
    C_HtM_a = sol.C_HtM_a
    C_R = sol.C_R
    C_R_a = sol.C_R_a
    inc = sol.inc
    inc_a = sol.inc_a
    pi_hh = sol.pi_hh
    r_hh = sol.r_hh       
    real_W = sol.real_W
    real_r_hh = sol.real_r_hh

    # targets
    A_R_ini_error = sol.A_R_ini_error
    Aq_diff = sol.Aq_diff

    # evaluations
    P_C_lag = lag(ini.P_C,P_C)
    pi_hh[:] = P_C/P_C_lag-1
    pi_hh_plus = lead(pi_hh,ss.pi_hh)

    real_W[:] = W/P_C
    real_r_hh[:] = (1+r_hh)/(1+pi_hh_plus)-1

    # a. income
    for t in range(par.T):
        inc_a[:,t] = (1-tau[t])*W[t]*L_a[:,t]/par.N_a + (1-tau[t])*par.W_U*ss.W*U_a[:,t]/par.N_a + Aq[t]/par.N
        inc_a[par.work_life_span:,t] += (1-tau[t])*par.W_R*ss.W
        inc[t] = np.sum(par.N_a*inc_a[:,t])

    # b. HtM
    for t in range(par.T):
        C_HtM_a[:,t] = inc_a[:,t]/P_C[t]
        A_HtM_a[:,t] = 0.0 

    # c. Ricardian
    for t0 in range(-par.life_span+1,par.T): # birthcohort

        for i in range(par.life_span):
            
            a = par.life_span - 1 - i
            t = t0 + a

            if t < 0: continue
            if t > par.T-1: continue

            # i. now and plus
            if a == par.life_span-1:

                A_R_a_now =  A_R_a[a,t] = A_R_death[t]
                C_R_a_plus = np.nan
            
            else:

                if t == par.T-1:
                    A_R_a_now = A_R_a[a,t] = ss.A_R_a[a]
                    C_R_a_plus = ss.C_R_a[a+1]
                else:
                    A_R_a_now = A_R_a[a,t]
                    C_R_a_plus = C_R_a[a+1,t+1]

            # ii. consumption
            RHS = par.zeta_a[a]*par.mu_Aq*(A_R_a_now/P_C[t])**(-par.sigma)
            
            if a < par.life_span-1: 
                RHS += (1-par.zeta_a[a])*par.beta*(1+real_r_hh[t])*C_R_a_plus**(-par.sigma) 

            C_R_a[a,t] = RHS**(-1/par.sigma)

            # iii. lagged assets
            A_R_a_lag = (A_R_a_now + P_C[t]*C_R_a[a,t] - inc_a[a,t])/(1+r_hh[t])

            if a > 0 and t > 0:
                A_R_a[a-1,t-1] = A_R_a_lag
            elif t0 < 0:
                A_R_ini_error[t0-(-par.life_span+1)] = A_R_a_lag-ini.A_R_a[a-1]
            elif t0 <= par.T-1-par.life_span+1:
                A_R_ini_error[t0-(-par.life_span+1)] = A_R_a_lag-0.0

    # d. aggregate
    for t in range(par.T):  
        
        # life-cycle
        C_a[:,t] = par.Lambda*C_HtM_a[:,t]+(1-par.Lambda)*C_R_a[:,t] 
        A_a[:,t] = par.Lambda*A_HtM_a[:,t]+(1-par.Lambda)*A_R_a[:,t] 

        # time period
        C[t] = np.sum(par.N_a*C_a[:,t])
        C_HtM[t] = np.sum(par.N_a*C_HtM_a[:,t])
        C_R[t] = np.sum(par.N_a*C_R_a[:,t])
        A[t] = np.sum(par.N_a*A_a[:,t])

        # bequest
        if t == 0:
            Aq_implied = (1+r_hh[t])*np.sum(par.zeta_a*par.N_a*ss.A_a)
        else:
            Aq_implied = (1+r_hh[t])*np.sum(par.zeta_a*par.N_a*A_a[:,t-1])

        Aq_diff[t] = Aq_implied-Aq[t]                                  

@nb.njit
def repacking_firms_components(par,ini,ss,sol):

    # inputs
    C = sol.C
    G = sol.G
    I = sol.I
    P_C = sol.P_C
    P_G = sol.P_G
    P_I = sol.P_I
    P_M_C = sol.P_M_C
    P_M_G = sol.P_M_G
    P_M_I = sol.P_M_I
    P_M_X = sol.P_M_X
    P_X = sol.P_X
    P_Y = sol.P_Y
    X = sol.X  

    # outputs
    C_M = sol.C_M
    C_Y = sol.C_Y
    G_M = sol.G_M 
    G_Y = sol.G_Y
    I_M = sol.I_M
    I_Y = sol.I_Y
    X_M = sol.X_M
    X_Y = sol.X_Y

    # evaluations
    C_M[:] = CES_demand(P_M_C,P_C,par.mu_M_C,C,par.sigma_C,Gamma=1) 
    G_M[:] = CES_demand(P_M_G,P_G,par.mu_M_G,G,par.sigma_G,Gamma=1)
    I_M[:] = CES_demand(P_M_I,P_I,par.mu_M_I,I,par.sigma_I,Gamma=1)
    X_M[:] = CES_demand(P_M_X,P_X,par.mu_M_X,X,par.sigma_X,Gamma=1)

    C_Y[:] = CES_demand(P_Y,P_C,1-par.mu_M_C,C,par.sigma_C,Gamma=1)
    G_Y[:] = CES_demand(P_Y,P_G,1-par.mu_M_G,G,par.sigma_G,Gamma=1)
    I_Y[:] = CES_demand(P_Y,P_I,1-par.mu_M_I,I,par.sigma_I,Gamma=1)
    X_Y[:] = CES_demand(P_Y,P_X,1-par.mu_M_X,X,par.sigma_X,Gamma=1)
 
@nb.njit
def goods_market_clearing(par,ini,ss,sol):

    # inputs
    C_M = sol.C_M
    C_Y = sol.C_Y
    G_M = sol.G_M
    G_Y = sol.G_Y
    I_M = sol.I_M
    I_Y = sol.I_Y
    X_M = sol.X_M
    X_Y = sol.X_Y
    Y = sol.Y

    # outputs
    M = sol.M
    
    # targets
    mkt_clearing = sol.mkt_clearing

    # evalautions
    M[:] = C_M + G_M + I_M + X_M 
    
    mkt_clearing[:] = Y - (C_Y + G_Y + I_Y + X_Y)

@nb.njit
def real_productivity(par,ini,ss,sol):

    # inputs
    r_K = sol.r_K
    r_ell = sol.r_ell
    P_Y = sol.P_Y
    P_C = sol.P_C
    inc = sol.inc
    Aq = sol.Aq

    # outputs
    real_r_K = sol.real_r_K
    real_r_ell = sol.real_r_ell
    real_inc = sol.real_inc
    real_Aq = sol.real_Aq

    #evaluations
    real_r_K[:] = r_K/P_Y
    real_r_ell[:] = r_ell/P_Y
    real_inc[:] = inc/P_C
    real_Aq[:] = Aq/P_C
    
