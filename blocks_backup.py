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
def CES_Y(Xi,Xj,mui,sigma):

    muj = 1-mui
    inv_sigma = 1/sigma
    pow_sigma = (sigma-1)/sigma
    inv_pow_sigma = sigma/(sigma-1)

    part_i = mui**inv_sigma*Xi**pow_sigma
    part_j = muj**inv_sigma*Xj**pow_sigma

    return (part_i+part_j)**inv_pow_sigma

@nb.njit
def CES_demand(mui,Pi,P,X,sigma):

    return mui*(Pi/P)**(-sigma)*X
    
@nb.njit
def CES_P(Pi,Pj,mui,sigma):

    muj = 1-mui
    
    part_i = mui*Pi**(1-sigma)
    part_j = muj*Pj**(1-sigma)

    return (part_i+part_j)**(1/(1-sigma))

@nb.njit
def adj_cost(iota,K_lag,Psi_0,delta_K):

    return 0.5*Psi_0*(iota/K_lag-delta_K)**2*K_lag

@nb.njit
def adj_cost_iota(iota,K_lag,Psi_0,delta_K):

    return Psi_0*(iota/K_lag-delta_K)

@nb.njit
def adj_cost_K(iota,K_lag,Psi_0,delta_K):

    return 0.5*Psi_0*(iota/K_lag-delta_K)**2 + Psi_0*(iota/K_lag-delta_K)*iota/K_lag

##########
# blocks #
##########

@nb.njit
def household_search(par,ini,ss,sol):

    # inputs
    L = sol.L

    # outputs
    S_a = sol.S_a
    S = sol.S
    L_ubar_a = sol.L_ubar_a
    L_ubar = sol.L_ubar
    delta_L = sol.delta_L
    curlyM = sol.curlyM
    m_s = sol.m_s
    v = sol.v
    m_v = sol.m_v
    L_a = sol.L_a

    # evaluations
    for t in range(par.T):
        
        # a. lagged employment
        if t == 0:
            L_lag = ini.L
        else:
            L_lag = L[t-1]

        # b. searchers and employed before matching
        for a in range(par.A):

            if a == 0:
            
                S_a[a,t] = 1.0
                L_ubar_a[a,t] = 0.0

            elif a >= par.A_R:
            
                S_a[a,t] = 0.0
                L_ubar_a[a,t] = 0.0

            else:

                if t == 0:
                    L_a_lag = ini.L_a[a-1]
                else:
                    L_a_lag = L_a[a-1,t-1]

                S_a[a,t] = (1-L_a_lag) + par.delta_L_a[a]*L_a_lag
                L_ubar_a[a,t] = (1-par.delta_L_a[a])*L_a_lag

        S[t] = np.sum(S_a[:,t])
        L_ubar[t] = np.sum(L_ubar_a[:,t])

        # c. aggregate separation rate
        delta_L[t] = (L_lag-L_ubar[t])/L_lag

        # d. matching
        curlyM[t] = L[t]-L_ubar[t]
        m_s[t] = curlyM[t]/S[t]
        v[t] = (m_s[t]**(1/par.sigma_m)*S[t]**(1/par.sigma_m)/(1-m_s[t]**(1/par.sigma_m)))**par.sigma_m
        m_v[t] = curlyM[t]/v[t]

        # e. emplolyment
        for a in range(par.A):
            L_a[a,t] = L_ubar_a[a,t] + m_s[t]*S_a[a,t]

@nb.njit
def labor_agency(par,ini,ss,sol):

    # inputs
    w = sol.w
    m_v = sol.m_v
    delta_L = sol.delta_L
    L = sol.L
    v = sol.v

    # outputs
    r_ell = sol.r_ell
    ell = sol.ell
    
    # evaluations
    m_v_plus = lead(m_v,ss.m_v)

    for k in range(par.T):

        t = par.T-1-k

        if k==0:
            r_ell_plus = ss.r_ell
        else:
            r_ell_plus = r_ell[t+1]
        
        r_ell[t] = 1/(1-par.kappa_L/m_v[t])*(w[t] - ((1-delta_L[t])/(1+par.r_firm)*r_ell_plus*par.kappa_L/m_v_plus[t])) 

    ell[:] = L-par.kappa_L*v

@nb.njit
def production_firm(par,ini,ss,sol):

    # inputs
    K = sol.K
    ell = sol.ell
    r_K = sol.r_K
    r_ell = sol.r_ell

    # outputs
    Y = sol.Y
    P_Y = sol.P_Y

    # targets
    FOC_K_ell = sol.FOC_K_ell

    # evaluations
    K_lag = lag(ini.K,K)

    Y[:] = CES_Y(K_lag,ell,par.mu_K,par.sigma_Y)
    P_Y[:] = CES_P(r_K,r_ell,par.mu_K,par.sigma_Y)

    FOC_K_ell[:] = K_lag/ell-par.mu_K/(1-par.mu_K)*(r_ell/r_K)**par.sigma_Y

@nb.njit
def bargaining(par,ini,ss,sol):

    # inputs
    w = sol.w
    Y = sol.Y
    ell = sol.ell

    # outputs
    MPL = sol.MPL
    w_ast = sol.w_ast

    # targets
    bargaining_cond = sol.bargaining_cond

    # evaluations
    w_lag = lag(ini.w,w)

    MPL[:] = ((1-par.mu_K)*Y/ell)**(1/par.sigma_Y)
    w_ast[:] = par.phi*MPL + (1-par.phi)*par.w_U

    bargaining_cond[:] = w - (par.gamma_w*w_lag + (1-par.gamma_w)*w_ast)
    
@nb.njit
def repacking_firms_prices(par,ini,ss,sol):

    # inputs
    P_Y = sol.P_Y
    P_M_C = sol.P_M_C
    # P_M_G = sol.P_M_G 
    P_M_I = sol.P_M_I
    P_M_X = sol.P_M_X

    # outputs
    P_C = sol.P_C
    # P_G = sol.P_G
    P_I = sol.P_I
    P_X = sol.P_X

    P_C[:] = CES_P(P_M_C,P_Y,par.mu_M_C,par.sigma_C)
    # P_G[:] = CES_P(P_M_G,P_Y,par.mu_M_G,par.sigma_G)
    P_I[:] = CES_P(P_M_I,P_Y,par.mu_M_I,par.sigma_I)
    P_X[:] = CES_P(P_M_X,P_Y,par.mu_M_X,par.sigma_X)

@nb.njit
def foreign_economy(par,ini,ss,sol):

    # inputs
    P_F = sol.P_F
    chi = sol.chi
    P_Y = sol.P_Y

    # outputs
    X = sol.X
    
    # evaluations
    X[:] = chi*(P_Y/P_F)**(-par.sigma_F)

@nb.njit
def capital_agency(par,ini,ss,sol):

    # inputs
    r_K = sol.r_K
    P_I = sol.P_I
    K = sol.K

    # outputs
    iota = sol.iota
    I = sol.I

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
    tau_tilde = sol.tau_tilde
    P_G = sol.P_G
    G = sol.G
    w = sol.w
    L = sol.L

    # outputs
    tau = sol.tau
    tau_bar = sol.tau_bar
    B_G = sol.B_G

    # evaluations
    tau_tilde = 0

    for t in range(par.T):

        if t == 0:
            B_G_lag = 0.0
        else:
            B_G_lag = B_G[t-1]
        
        tau_bar[t] = ss.tau*(B_G_lag/ss.B_G)**par.epsilon_B #problem for negative værdier af B_G_lag
        omega = 3*((t-par.t_b)/par.delta_B)**2 - 2*((t-par.t_b)/par.delta_B)**3

        if t < par.t_b:
            tau[t]=tau_tilde
        
        elif t > par.t_b + par.delta_B:
            tau[t]=tau_bar[t]
        
        else:
            tau[t]=(1-omega)*tau_tilde+omega*tau_bar[t]
        
        B_G[t] = (1+par.r_b)*B_G_lag + P_G[t]*G[t] - tau[t]*w[t]*L[t]


@nb.njit
def households_consumption(par,ini,ss,sol):    

    # inputs
    L_a = sol.L_a
    P_C = sol.P_C
    w = sol.w
    Bq = sol.Bq
    tau = sol.tau

    # outputs
    pi_hh = sol.pi_hh
    C_HTM = sol.C_HTM
    C_R = sol.C_R
    C_a = sol.C_a
    B_a = sol.B_a
    C = sol.C
    B = sol.B

    # evaluations
    P_C_lag = lag(ini.P_C,P_C)

    pi_hh = P_C/P_C_lag-1
    pi_hh_plus = lead(pi_hh,ss.pi_hh)

    # C_HTM = w*L_a+(1-par.Lambda)*Bq/par.A #(1-tau)*

    # targets
    Bq_match = sol.Bq_match

    # find consumption backwards
    for i in range(par.A):
        
        a = par.A-1-i 

        for t in range(par.T): 

            # RHS
            if i == 0:
                
                RHS = par.mu_B*Bq[t]**(-par.sigma)

            else:

                if t == par.T-1:
                    C_R_plus = ss.C_R[a+1]
                else:
                    C_R_plus = C_R[a+1,t+1]

                RHS = par.beta*(1+par.r_hh)/(1+pi_hh_plus[t])*C_R_plus**(-par.sigma)    

            # invert
            C_R[a,t] = RHS**(-1/par.sigma)
            C_a[a,t] = (1-par.Lambda)*C_R[a,t] #par.Lambda*C_HTM[a,t]+

    # find savings forward (and aggregates)
    for t in range(par.T):

        for a in range(par.A):

            if a == 0:
                B_a_lag = 0.0
            elif t==0:
                B_a_lag = ini.B_a[a-1]
            else:
                B_a_lag = B_a[a-1,t-1]
                
            B_a[a,t] = (1+par.r_hh)*B_a_lag + w[t]*L_a[a,t] + (1-par.Lambda)*Bq[t]/par.A - P_C[t]*C_a[a,t] #(1-tau[t])*
        

    # aggregate
    C[:] = np.sum(C_a,axis=0)
    B[:] = np.sum(B_a,axis=0)  


    # matching Bq
    Bq_match[:] = Bq - B_a[-1,:]

@nb.njit
def repacking_firms_components(par,ini,ss,sol):

    # inputs
    P_Y = sol.P_Y

    P_M_C = sol.P_M_C
    P_C = sol.P_C
    C = sol.C

    # P_M_G = sol.P_M_G
    # P_G = sol.P_G
    # G = sol.G

    P_M_I = sol.P_M_I
    P_I = sol.P_I
    I = sol.I

    P_M_X = sol.P_M_X
    P_X = sol.P_X
    X = sol.X    

    # outputs
    C_M = sol.C_M
    # G_M = sol.G_M 
    I_M = sol.I_M
    X_M = sol.X_M

    C_Y = sol.C_Y
    # G_Y = sol.G_Y
    I_Y = sol.I_Y
    X_Y = sol.X_Y

    # evaluations
    C_M[:] = CES_demand(par.mu_M_C,P_M_C,P_C,C,par.sigma_C)
    # G_M[:] = CES_demand(par.mu_M_G,P_M_G,P_G,G,par.sigma_G)
    I_M[:] = CES_demand(par.mu_M_I,P_M_I,P_I,I,par.sigma_I)
    X_M[:] = CES_demand(par.mu_M_X,P_M_X,P_X,X,par.sigma_X)

    C_Y[:] = CES_demand(1-par.mu_M_C,P_Y,P_C,C,par.sigma_C)
    # G_Y[:] = CES_demand(1-par.mu_M_G,P_Y,P_G,G,par.sigma_G)
    I_Y[:] = CES_demand(1-par.mu_M_I,P_Y,P_I,I,par.sigma_I)
    X_Y[:] = CES_demand(1-par.mu_M_X,P_Y,P_X,X,par.sigma_X)
    
@nb.njit
def goods_market_clearing(par,ini,ss,sol):

    # inputs
    Y = sol.Y
    
    C_M = sol.C_M
    # G_M = sol.G_M
    I_M = sol.I_M
    X_M = sol.X_M

    C_Y = sol.C_Y
    # G_Y = sol.G_Y
    I_Y = sol.I_Y
    X_Y = sol.X_Y

    # outputs
    M = sol.M
    
    # targets
    mkt_clearing = sol.mkt_clearing

    # evalautions
    M[:] = C_M + I_M + X_M #+ G_M
    
    mkt_clearing[:] = Y - (C_Y + I_Y + X_Y) #+ G_Y