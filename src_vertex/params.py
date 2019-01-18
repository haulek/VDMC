from scipy import pi

p = {'rs'     : 4,   # rs of the electron gas
     'beta'   : 50,  # inverse temperature
     'lmbda'  : 0.1, # inverse screening length
     'dmu'    : -0.0015690226138, # chemical potentila shift from mu=-kF^2
     #'lmax'   : 10,  # expansion for the angle
     'lmax'   : 4,  # expansion for the angle
     'lmax_t' : 20,  # expansion order for time variable
     'lmax_k' : 15,  # expansion order for momentum variable k
     'lmax_q' : 17,  # expansion order for external momentum variable q
     'Nq'     :120,  # number of external momentum points in computing the ladders
     'Nk'     :120,  # number of internal momentum points in computing the ladders
     'ntail'  : 30,  # number of Matsubara points in the tail
     'SaveAll': True,# If we want to save original vertex in k,w representation, not jut Legnedre form
     }
p['nom'] = int(1.5*p['beta'])

kF = (9*pi/4.)**(1./3.) /p['rs']    
p['cutoffq'] = 3.0*kF
p['cutoffk'] = p['cutoffq'] + 1.0*kF + 3.0/p['beta']
p['Nt']  = max(100,int(2*p['beta']))    # Number of imaginary time points
