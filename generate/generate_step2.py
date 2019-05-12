#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
import copy as cp
from scipy import *
import itertools
import sys
import loops as lp
import os
import equivalent as eq
from collections import Counter

def GiveInteractionPermutations_new(V0r):
    """ Gives interaction propagators, which are equivalent to V0. For example, if V0=[(2,3),(4,5)] then
        equivalent set of interactions is
        [(2, 3), (5, 4)],
        [(3, 2), (4, 5)],
        [(3, 2), (5, 4)],
        [(4, 5), (2, 3)],
        [(4, 5), (3, 2)],
        [(5, 4), (2, 3)],
        [(5, 4), (3, 2)]
    """ 
    n = len(V0r)
    if n==0: return []
    V0f = list(itertools.chain.from_iterable(V0r))  # just a single list from list of tuples
    n_min = min(V0f)
    n_max = max(V0f)
    Vn = [0]*(2*n)                                  # empty array of correct size
    pm = itertools.permutations(range(n_min,n_max+1))    # all permutations of numbers 2,3,4,5....2*n-1
    V0perm=[]
    for p in pm:
        for i in range(0,2*n): Vn[i] = p[V0f[i]-n_min]
        # since interaction has no direction, pair (i,k) is equivalent to pair sorted(i,k)
        # the order in which propagators are stored does not matter, hence we can reorder them
        l = sorted([tuple(sorted([Vn[2*i],Vn[2*i+1]])) for i in range(len(V0r))])
        if l==V0r: V0perm.append(p)
    V0perm = V0perm[1:]
    return (V0perm)


def Transform(Gp, prm):
    "Input is a diagram Gp, and permutation of interactions prm"
    Gn = list(Gp[:]) # create the list, so we can change
    for j in range(len(prm)):   # If we exchange vertices according to permutation V0perm, the electron propagators has to change 
        Gn[prm[j]] = Gp[j+2]    # in the following way:
    Gm=Gn[:]
    for j in range(len(Gn)):  # because (0,1) interaction line is the measuring line
        if Gn[j]>=2: Gm[j] = prm[Gn[j]-2]
    return array(Gm) # finally we have the correspondingly permuted Gp


def InverseOneLoop(Gp):
    cycles = lp.to_cycles(Gp)
    forms=[]
    for cc in cycles:
        Gnew = list(Gp[:])
        for c in cc:
            Gnew[Gp[c]] = c
        forms.append( Gnew )
    return forms
    
def CheckHowSimilar(Gall,Inverse,V0perm):
    ip = 0
    Gnew = [ array(Gall[ip]) ]
    considered = set([ip])
    if Inverse[ip]!=ip:
        iq = Inverse[ip]
        Gnew.append( array(Gall[iq]) )
        considered.add(iq)
        
    all = set(range(len(Gall)))
    print
    print 'all=', all, 'considered=', considered
    while len(considered) < len(all):
        ip = min(all-considered) # not yet considered
        Gps = [ list(Gall[ip][:]) ]
        considered.add(ip)
        if Inverse[ip]!=ip:
            iq = Inverse[ip]
            Gps.append( list(Gall[iq][:]) )
            considered.add(iq)
            
        #Gp = Gall[ip]   # Considering this diagram, we will change this diagram until we like it
        print 'I am considering Gp=', Gps, 'while the list of so-far considered diags is', considered
        #Gns = list(Gp[:]) # convert from tuple to list, so that we can change
        forms={}
        for i in range(len(V0perm)): # We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
            if i==0:
                Gns = Gps[:]
            else:
                Gns = [Transform( Gp, V0perm[i] ).tolist() for Gp in Gps]
            print 'at i=',i,'we generated ', Gns
            how_many_equals = zeros(len(Gnew),dtype=int)
            for j in range(len(Gnew)):  # over all other diagrams
                how_many_equal = [ len(filter(lambda x: x==True, Gnew[j]==Gn)) for Gn in Gns]
                how_many_equals[j] = sum(how_many_equal)
                print 'current=', Gns, j, 'previous=', Gnew[j], 'how many equal=', how_many_equal
            print 'How many equals=', how_many_equals
            max_how_many_equals = max(how_many_equals)
            if forms.has_key(max_how_many_equals):
                forms[max_how_many_equals].append( [i,Gns] )
            else:
                forms[max_how_many_equals] = [ [i,Gns] ]
        max_equals = max(forms.keys())  # maximum number of equal Green's function propagators possible
        print 'max_equals=', max_equals
        frms = forms[max_equals]        # the many forms that have the maximum number of equal propagators
        print 'what forms=', forms
        print 'best forms=', frms
        index_min = argmin([x[0] for x in frms]) # which one to choose from. The one with the lowest index....
        print 'index_min=', index_min, 'and its value=', frms[index_min][1]
        Gnew += [array(x) for x in frms[index_min][1]] # now storing it
        print 'Gnew is now', Gnew

    print 'Finally, considered=', considered
    print 'len(considered)=', len(considered), 'len(all)=', len(all)
    
    return [tuple(g) for g in Gnew]

def FoundEqual(_Gn_, Gall):
    iequal=-1
    for iq in range(len(Gall)):
        if _Gn_ == Gall[iq]:
            iequal = iq
            break
    return iequal

def CompareWithInverse(Gall,V0perm):
    Gnew = cp.deepcopy(Gall)
    Inverse=zeros(len(Gnew),dtype=int)
    for ip in range(len(Gnew)):
        Ginv = tuple(InverseOneLoop(Gnew[ip])[-1])   # We first inverse one fermionic loop
        print ip, 'Considering', Gnew[ip], ' with inverse Ginv=', Ginv
        
        iequal = FoundEqual(Ginv, Gnew)   # Finding if this transformed diagram form exists in the rest of the list
        print 'iequal=', iequal, 'Searching if ', Ginv, 'is in current', Gnew
        if iequal>=0:
            print 'This diagram is in a form which is compatible with its inverse'
            Inverse[ip] = iequal
        else:
            for i in range(1,len(V0perm)): # We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
                _Gn_ = tuple(Transform( Ginv, V0perm[i] )) # transforming Ginv into equivalent form
                print 'transformed[',i,']=', _Gn_
                iequal = FoundEqual(_Gn_, Gall)   # Finding if this transformed diagram form exists in the rest of the list
                if iequal>0:
                    print 'iequal=', iequal
                    Gbetter = tuple(Transform(Gall[ip], V0perm[i]))
                    Inverse[ip] = iequal
                    if ip!=iequal:   # If the diagram is inverse to itself, we do not need to change its form.
                        print 'Better form is', Gbetter
                        Gnew[ip] = Gbetter
                    else:
                        print 'Diagram is inverse to inself, hence leaving it in its current form'
                    print 'Gnew=', Gnew
                    break
    print 'Inverse=', Inverse
    print 'Gnew=', Gnew
    return Gnew, Inverse


def Transform2Phi(Gp):
    Gn = Gp[:]
    Gn[0]=0; Gn[1]=1
    for i in range(2,len(Gp)):
        if Gp[i]==0 and Gp[Gp[i]]!=1 or Gp[i]==1 and Gp[Gp[i]]!=0:
            Gn[i] = Gp[Gp[i]]
        if Gp[i]==0 and Gp[Gp[i]]==1 or Gp[i]==1 and Gp[Gp[i]]==0:
            Gn[i] = Gp[Gp[Gp[i]]]
    return Gn    

def FindFunctionals(Gall,V0perm):
        
    def CheckPreviousPhis(prmGn, prm, Phis):
        for j in range(len(Phis)):    # Now check with all previous Phis if this is equivalent to any of them
            #print 'Is prmGn=', prmGn, ' equivalent to Phis[',j,']=', Phis[j]
            if prmGn == Phis[j]: # Checking if such Phi has already occured. If yes, we know that this diagram is part of the same group
                Gnew[ip] = tuple(Transform(Gp, prm))
                indx[ip] = j
                break
        return indx[ip]>=0
    
    indx = -ones(len(Gall), dtype=int)
    Phis = [ Transform2Phi(list(Gall[0])) ]
    indx[0] = 0
    Gnew = cp.deepcopy(Gall)

    for ip in range(1,len(Gall)):
        print 'indx=', indx, 'Phis=', Phis
        Gp = list(Gall[ip])
        Gn = Transform2Phi(Gp)
        print ip, 'Considering', Gp, ' with Phi=', Gn
        
        prmGns=[Gn] # non-equivalent permutations of Phi, which were already tried
        Found = CheckPreviousPhis(Gn, V0perm[0], Phis)
        
        if not Found:
            for iq,prm in enumerate(V0perm[1:]):    # over all possible permutations
                #print 'Gn=', Gn, 'prm=', prm, 'going to transform'
                prmGn = Transform(Gn, prm).tolist() # new permutation of Phi. But is it different?
                is_new=True                         # Is this permutation that gives different Phi?
                for j in range(len(prmGns)):        # over all previous forms of this diagram
                    if prmGn == prmGns[j]:
                        is_new=False                # it is not new.
                        break
                #print 'prmGn=', prmGn, 'is new=', is_new
                if is_new:
                    prmGns.append(prmGn)
                    Found = CheckPreviousPhis(prmGn, prm, Phis)
                    if Found: break
                    
                    
        if indx[ip]<0:  # no permutation can find equivalent Phi. Hence Phi of this diagram is different from all previous diagrams
            indx[ip] = len(Phis)
            Phis.append( Gn )
            Gnew[ip] = tuple(Gp)
        print '    ', ip, 'equivalent to index=', indx[ip], 'transformed Phi=', Phis[indx[ip]], 'which gives diag=', Gnew[ip]
    #print 'Finished with indx=', indx
    #print 'and Gnew=', Gnew
    return Gnew, indx

def WhichContainBubbles(Gall):
    containsB = zeros(len(Gall), dtype=int)
    for ip in range(len(Gall)):
        Gp = Gall[ip]
        for i in range(len(Gp)):
            if Gp[Gp[i]]==i: 
                containsB[ip]=1
                break
    return containsB


def EliminateBubble(i, j, Gp, Vtype, fellow, V0, Vind):
    N = len(Gp)
    participant = sorted([fellow[i], fellow[j], i, j])
    prm = range(len(Gp))            # create an index to resort diagram vertices
    prm[fellow[i]] = participant[0] # connecting around the bubble, using the smallest index involved
    prm[fellow[j]] = participant[1] # now we connect around the bubble
    # now we need to shift all other vertices so as to eliminate these two indices, inherited from the bubble
    rest = range(participant[2]+1,len(Gp)) # the rest of the diagram after the bubble
    rest.remove(participant[3])         # of course the last index should appear as the last integer
    #print 'rest=', rest
    for k in rest:
        if k < participant[3]: prm[k]=k-1
        else: prm[k] = k-2
    prm[i] = N-2
    prm[j] = N-1

    Vtype[Vind[participant[0]]] =  Vtype[Vind[i]] + Vtype[Vind[j]] + 1  # this is now counter term of higher order than previous interactions
    del Vtype[ Vind[participant[3]] ] # this interaction is removed
    
    #print 'found bubble', i, j, fellow[i], fellow[j], 'in Gp=', Gp, 'prm=', prm
    Gnew = Transform(Gp, prm[2:])
    
    #print '   Gnew=', Gnew[:-2], 'Vtype=', Vtype
    return (Gnew[:-2], Vtype )
    
def CreateCounterTerms_Of_All_Levels(Gp, V0, fellow, Vind, Vperm, Viiperm, indx=0):
    Vtype=zeros(len(V0),dtype=int).tolist()
    #CounterTerms=[]
    # First pass through diagram finds all bubbles and eliminates one by one
    nord = len(Gp)/2
    Gall={} # next level must be of lower order, hence it does not need to check for equivalence back
    Vtyp={} # next level must be of lower order, hence it does not need to check for equivalence back
    for n in range(1,nord):
        Gall[n]=[]
        Vtyp[n]=[]
    # If there are more bubbles, we need to go through recursively
    #for Gp,Vtype,ii in CounterTerms:
    Gall[nord] = [Gp]
    Vtyp[nord] = [zeros(len(V0),dtype=int).tolist()]
    for corder in range(nord,1,-1): # first going over those of order nord, followed by nord-1, followed by nord-2,...
        for ni in range(len(Gall[corder])):
            Gp = Gall[corder][ni]
            Vtype = Vtyp[corder][ni]
            for i in range(len(Gp)):
                if Gp[Gp[i]]==i and i<Gp[i]: # found bubble
                    cVtype = Vtype[:]
                    (Gnew, cVtype) = EliminateBubble(i, Gp[i], Gp, cVtype, fellow, V0, Vind)
                    iorder = len(Gnew)/2
                    if iorder != corder-1:
                        print 'ERROR: After elimination of bubble we should have one order smaller diagram!'
                    ic = -1
                    if len(Gall[iorder])>0:
                        (ic,ii) = eq.FindEquivalent2(Gnew, cVtype, Gall[iorder], Vtyp[iorder], Vperm[iorder], Viiperm[iorder])  # finding if the exchanged diagram has an equivalent in the rest of the list. Returns index to it.
                    if ic < 0: # this is new type of diagram
                        Gall[iorder].append( list(Gnew) )
                        Vtyp[iorder].append( list(cVtype) )
                    else:
                        if tuple(Gnew) != tuple(Gall[iorder][ic]):
                            print 'WARNING : Found equivalent counter term of ', Gnew, ', which is ', Gall[iorder][ic]
                            print '        : their Vtyp=', cVtype, 'and', Vtyp[iorder][ic]
                       
                    #AlreadyPresent=False
                    #for Gold,Vold,jj in CounterTerms:
                    #    if len(Gold)==len(Gnew) and array_equal(Gold,Gnew) and (Vold==cVtype):
                    #        AlreadyPresent=True
                    #if not AlreadyPresent:
                    #    CounterTerms.append( (Gnew, cVtype,ii) )
                        
    #return CounterTerms
    return Gall, Vtyp

def CreateSingleCounterTerm(Gp, V0, fellow, Vind, Vperm, Viiperm, indx=0):
    """This routine works only when Bubbles are computed analytically and added to the counter terms. 
    In this case we need to eliminate all bubbles recursively, and replace them with the single counter term of the highest possible order.
    """
    Vtype=zeros(len(V0),dtype=int).tolist()
    nord = len(Gp)/2
    Gall={} # next level must be of lower order, hence it does not need to check for equivalence back
    Vtyp={} # next level must be of lower order, hence it does not need to check for equivalence back
    for n in range(1,nord):
        Gall[n]=[]
        Vtyp[n]=[]
    Gall[nord] = [Gp]
    Vtyp[nord] = [zeros(len(V0),dtype=int).tolist()]
    
    # First pass through diagram finds all bubbles and eliminates one by one
    while (True):
        # Repeat until all bubbles are eliminated
        FoundBubble = False
        for i in range(len(Gp)):  # going over the diagram
            if Gp[Gp[i]]==i and i<Gp[i]:  # is there any bubble?
                cVtype = Vtype[:]         # copy current Vtype, which shows which interactions are counter terms, and which are not
                #print 'Eliminating bubble at i=', i, 'Gp[i]=', Gp[i]
                (Gnew, cVtype) = EliminateBubble(i, Gp[i], Gp, cVtype, fellow, V0, Vind) # eliminate bubble
                #print 'Gnew=', Gnew, 'cVtype=', cVtype
                Gp, Vtype = Gnew, cVtype   # and then take the diagram with bubble eliminated, and use it again to search bubbles
                FoundBubble = True         # the bubble was found, hence we need to check again for possible more bubbles
                break                      # go back to the beginning and check if there is another bubble.
        if not FoundBubble:                # no more bubbles?
            iorder = len(Gnew)/2
            for vi in range(1,len(cVtype)):           # over all interactions
                for vj in range(1,len(cVtype)):       # over all interactions
                    if vi==vj or cVtype[vi]==0 or cVtype[vj]==0: continue  # we need both interactions to be counter-terms
                    if Gnew[2*vi+1] == 2*vj+1 or Gnew[2*vj+1] == 2*vi+1 : # counter terms vi and vj have the dynamic vertex (2i+1) nearby
                        print 'starting with Gnew=', Gnew, 'vi=', vi, 'vj=', vj
                        Gn = Gnew[:]
                        Success = False
                        for (v1,v2) in [ (vi,vj), (vj,vi) ]:  # try both possibilities if necessary
                            # exchanging vertices 2*v1 and 2*v1+1 (first iteration v1=vi, and next v1=vj)
                            Gn[2*v1], Gn[2*v1+1] = Gn[2*v1+1], Gn[2*v1]
                            for j in range(len(Gnew)):
                                if Gn[j]==2*v1 :
                                    Gn[j]=2*v1+1
                                elif Gn[j]==2*v1+1:
                                    Gn[j] = 2*v1
                            
                            if not ( Gn[2*v1+1] == 2*v2+1 or Gn[2*v2+1] == 2*v1+1):
                                Success = True  # managed to avoid counter terms to tuch
                                Gnew = Gn
                                break
                        if not Success:
                            print 'WARNING : Could not exchange the indices so that the vertex-type II is avoided. You need to code vertex type II.'
                        print 'Finished with Gnew=', Gnew, 'vi=', vi, 'vj=', vj
            #CounterTerms.append( (Gnew, cVtype,indx) ) # all bubbles were eliminated, hence store the result
            Gall[iorder].append( list(Gnew) )
            Vtyp[iorder].append( list(cVtype) )
            break
    return Gall, Vtyp #CounterTerms

def CreateAllCounterTerms(Gall, indx, V0, Vperm, Viiperm, AllCounters=True):
    # note that those diagrams with the same indx could be simplified using common information. But somehow it is simpler to redo
    # the calculation for each diagram, but is less efficient. If this part of the code becomes critical, you can optimize it.
    nord = len(Gall[0])/2
    fellow = zeros(2*nord, dtype=int)
    Vind   = zeros(2*nord, dtype=int)
    for i in range(len(V0)):
        fellow[V0[i][0]] = V0[i][1]
        fellow[V0[i][1]] = V0[i][0]
        Vind[V0[i][0]]=i
        Vind[V0[i][1]]=i
        
    Gall_corder={}
    Vtyp_corder={}
    Indx_corder={}
    for n in range(2,nord):
        Gall_corder[n]=[]
        Vtyp_corder[n]=[]
        Indx_corder[n]=[]
        
    #CounterTerms=[]
    for ip in range(len(Gall)):
        Gp = Gall[ip]
        if (AllCounters):
            #current = CreateCounterTerms_Of_All_Levels(Gp, V0, fellow, Vind, Vperm, Viiperm, indx[ip])
            c_G, c_Vtyp = CreateCounterTerms_Of_All_Levels(Gp, V0, fellow, Vind, Vperm, Viiperm, indx[ip])
            for n in range(2,nord):
                Gall_corder[n] += c_G[n]
                Vtyp_corder[n] += c_Vtyp[n]
                Indx_corder[n] += [indx[ip] for j in range(len(c_G[n]))]
        else:
            #CounterTerms += CreateSingleCounterTerm(Gp, V0, fellow, Vind, Vperm, Viiperm, indx[ip])
            c_G, c_Vtyp = CreateSingleCounterTerm(Gp, V0, fellow, Vind, Vperm, Viiperm, indx[ip])
            for n in range(2,nord):
                Gall_corder[n] += c_G[n]
                Vtyp_corder[n] += c_Vtyp[n]
                Indx_corder[n] += [indx[ip] for j in range(len(c_G[n]))]

        #if ip==86: sys.exit(0)
            
    #return CounterTerms
    return Gall_corder, Vtyp_corder, Indx_corder



def FindEquivalent(Gp, istart, Gall, V0perm):
    Gn = list(Gp[:]) # convert from tuple to list, so that we can change
    for i,p in enumerate(V0perm): # We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
        for j in range(len(p)):   # If we exchange vertices according to permutation V0perm, the electron propagators has to change 
            Gn[p[j]] = Gp[j+2]    # in the following way:
        Gm=Gn[:]
        
        for j in range(len(Gn)):  # because (0,1) interaction line is the measuring line
            if Gn[j]>=2: Gm[j] = p[Gn[j]-2]
        
        _Gn_ = tuple(Gm)          # finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else

        #print i, 'permuting it into', _Gn_
        #print 'Gp=', Gp, 'p=', p, 'Gn=', _Gn_, 'interm=', Gn
        for iq in range(istart,len(Gall)):  # over all other diagrams
            if _Gn_==Gall[iq]:    # is this permuted diagram somewhere in the remaining of the list?
                #print 'ipermutation=', i
                return iq
    return -1
def FindEquivalent2(Gp, cVtyp, Gall, Vtyp, V0perm, Viiperm):
    Gn = list(Gp[:]) # convert from tuple to list, so that we can change
    nord = len(Gp)/2
    #print 'Finding equivalent of Gp=', Gp, 'Vp=', cVtyp
    for i,p in enumerate(V0perm): # We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
        for j in range(len(p)):   # If we exchange vertices according to permutation V0perm, the electron propagators has to change 
            Gn[p[j]] = Gp[j+2]    # in the following way:
        Gm=Gn[:]
        
        for j in range(len(Gn)):  # because (0,1) interaction line is the measuring line
            if Gn[j]>=2: Gm[j] = p[Gn[j]-2]

        _Gn_ = tuple(Gm)          # finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else

        xVtyp = [0]*(nord-1)
        for j in range(1,nord):
            #print j, cVtyp[j], Viiperm[i][j-1]
            xVtyp[Viiperm[i][j-1]-1] = cVtyp[j]
        xVtyp = tuple(xVtyp)
        
        #xVtyp = tuple([cVtyp[Viiperm[i][j]] for j in range(nord-1)]) # permutaed cVtyp
        #print i, 'p=', p, 'Viiperm=', Viiperm[i], 'cVtyp=', cVtyp[1:], 'xVtyp=', xVtyp, 'Gm=', _Gn_, 'xxVtyp=', xxVtyp
        
        #print i, 'permuting it into', _Gn_
        #print 'Gp=', Gp, 'p=', p, 'Gn=', _Gn_, 'interm=', Gn
        for iq in range(0,len(Gall)):  # over all other diagrams
            if _Gn_==Gall[iq] and xVtyp == tuple(Vtyp[iq][1:]):    # is this permuted diagram somewhere in the remaining of the list?
                #print 'permuted Vtype=', xVtyp, ' and equivalent=', Vtyp[iq][1:]
                #if xVtyp != tuple(Vtyp[iq][1:]):
                #    print 'ERROR Vtypes not the same!'
                return iq
    return -1

def combine_exchange(ex1, ex2):
    ex = ex1 + ex2
    e0 = set(ex)
    for i in e0:
        if ex.count(i) == 2: # if entry appears twice, we remove all occurences
            while i in ex: ex.remove(i)
    return sorted(ex)
        
def Find_Hugenholtz(Glast, Vtyp, indx, V0perm, Hugenholtz_All=False, Viiperm=None):
    Vtype_all=[]
    G_Hugen=[]
    Factor=[]
    FIndx=[]
    Signs_all=[]
    debug = False
    Gall = cp.deepcopy(Glast)
    Indx = cp.copy(indx)
    norder = len(Glast[0])/2
    NotCounter=False
    if sum(Vtyp)==0:
        NotCounter=True

    print 'NotCounter=', NotCounter
    while (Gall):
        ip=0
        Gc = Gall[ip] # starting by checking the first remaining diagram
        if debug :
            print '********************************'
            print 'considering Gc=', Gc, 'Vtype=', Vtyp[ip], 'len(Gall)=', len(Gall)
        fellows=[ ([], Gall[ip] ) ] # this one is the first in the fellow list
        fellows_index0 = [ip]   # and needs to be removed, once we found his group
        fellows_iiperm = [range(1,norder)]
        ncases = 2**(norder-1) # there are 2^(norder-1) cases of interaction exchange. Let's enumerate both using binary representation of integer.
        for i in range(1,ncases):
            Gp = copy(Gc)                              # the diagram for which we try to find all Hugenholtz equivalents
            Vexchanged=[ j+1 for j in range(norder-1) if i & (1<<j) ] # which interactions need to be exchanged for this case, which is one out of 2^(n-1)
            #Vexchanged=[ norder-j-1 for j in range(norder-1) if i & (1<<j) ]
            if debug:
                print 'Vexchanged=', Vexchanged
                print i, ': ',
            for ii in Vexchanged:
                j1, j2 = 2*ii, 2*ii+1  # these are the vertices that need to be exchanged
                if debug:
                    print ' Exchanging (', j1,',', j2, ') -> (', j2,',',j1,')  ', 'with Vtyp=', Vtyp[ip],
                i1, i2 = Gc.index(j1), Gc.index(j2)        # vertices to exchange
                Gp[i1], Gp[i2] = Gp[i2], Gp[i1]            # with Hugenholtz exchange becomes Gp
                
            if NotCounter:
                #ic = FindEquivalent(Gp, 0, Gall, V0perm)
                ic = eq.FindEquivalent(Gp, 0, Gall, V0perm)  # finding if the exchanged diagram has an equivalent in the rest of the list. Returns index to it.
                viiperm = range(1,norder)
            else:
                #ic = FindEquivalent2(Gp, Vtyp[ip], Gall, Vtyp, V0perm, Viiperm)
                ic,ii = eq.FindEquivalent2(Gp, Vtyp[ip], Gall, Vtyp, V0perm, Viiperm)  # finding if the exchanged diagram has an equivalent in the rest of the list. Returns index to it.
                viiperm = Viiperm[ii]
                
            if debug:
                print
                print 'And got', Gp,
            if ic>=0:
                fellows.append( (Vexchanged, Gall[ic] ) ) # this is now new fellow in the group
                fellows_index0.append( ic )           # and needs to be removed from the list of all remaining
                fellows_iiperm.append( viiperm )
                if debug:
                    print 'which is equivalent to', ic, Gall[ic]
            else:
                if debug:
                    ic_all=[]
                    ic=-1
                    while True:
                        ic = eq.FindEquivalent(Gp, ic+1, Gall, V0perm)
                        if ic<0: break
                        ic_all.append(ic)
                        
                    print 'and cannot find equivalent but have',
                    for ic in ic_all:
                        Gall[ic], Vtyp[ic], ' , ',
                    print

        if debug: print 'fellows_index0=', fellows_index0
        Hugen_equal_V={} # This is used only when Hugenholtz_All
        cVty = Vtyp[ip]
        if False: # This simplistic remove does not work in general, but gives the idea what we want to accomplist, namely remove all duplications.
            flrm=[]
            for i in range(len(fellows_index0)):
                if fellows_index0[i] in fellows_index0[:i]:
                    flrm.append(i)
        else:   # we want to keep the minimal number of Hugenholtz-exchanges so as to get all diagrams in the group of fellows.
            # We want to remove all duplications.
            nrd = len(fellows_index0).bit_length() - 1   # we will achieve : len(fellows_index0) == 2^nrd
            fellows_index1 = [fellows_index0[1<<j] for j in range(nrd)] # We just select terms like 0001,0010,0100,1000, which are sufficient to find possible repetition
            if debug: print 'fellows_index1=', fellows_index1
            rmf=set()
            for i in range(1,nrd):   # did this index i already appear before?
                V_e_i = fellows[1<<i][0][0]  # which interaction was exchanged to get fellow[i]
                for j in range(i):   # Check if two independent exchanges give the same diagram.
                    if fellows_index1[j]==fellows_index1[i]: # If yes, remove all exchanges associated with the second exchange.
                        V_e_j = fellows[1<<j][0][0] # which interaction exchanged to get fellow[j]
                        if cVty[V_e_j] > cVty[V_e_i]:  # level of counter-term for fellow[i] and fellow[j]
                            rmf.update( filter(lambda x: x&(1<<j), range((1<<nrd))) ) # remove the term with higher counterterm
                            if Hugen_equal_V.has_key(V_e_i):
                                Hugen_equal_V[V_e_i].append(V_e_j)
                            else:
                                Hugen_equal_V[V_e_i] = [V_e_j]  # these two interaction lines could be used in Hugen. Any of them is fine
                            #print 'i=', V_e_i, 'j=', V_e_j, 'eliminate', V_e_j, 'H_e_V=', Hugen_equal_V
                        else:
                            rmf.update( filter(lambda x: x&(1<<i), range((1<<nrd))) )
                            if Hugen_equal_V.has_key(V_e_j):
                                Hugen_equal_V[V_e_j].append(V_e_i)
                            else:
                                Hugen_equal_V[V_e_j] = [V_e_i]  # these two interaction lines could be used in Hugen. Any of them is fine
                            #print 'i=', V_e_i, 'j=', V_e_j, 'eliminate=', V_e_i, 'H_e_V=', Hugen_equal_V
            if debug: print 'fellows_index1=', fellows_index1, 'HeV=', Hugen_equal_V
            flrm = list(rmf)
            
        if debug:
            print 'flrm=', flrm
            print 'Fellows start', fellows
        
        fellows_considered = fellows_index0[:]
        # Now we remove the repetition, and keep only the essential group of diagrams, which are all distinct
        for ic in sorted( flrm, reverse=True ):
            del fellows[ic]
            del fellows_considered[ic]
            del fellows_iiperm[ic]
        if debug:
            print 'Fellows end  ', fellows

        how_many_appear = 1
        fellows_considered2 = set(fellows_considered)
        if len(fellows_considered) != len(fellows_considered2):
            fw = {}
            for i in range(len(fellows_considered)):
                fc = fellows_considered[i]
                if fw.has_key(fc):
                    fw[fc].append(i)
                else:
                    fw[fc]=[i]
            klengths = [len(fw[k]) for k in fw.keys()]
            if klengths[1:] != klengths[:-1]: # check that all klengths are equal
                print 'ERROR : We still have repetitions and not all terms appear equal number of times: ', klengths
                print '      : Here fw=', fw
                print '      : fellows_considered=', '('+str(len(fellows_considered))+')', fellows_considered
                print '      : while fellows_considered2=', '('+str(len(fellows_considered2))+')', fellows_considered2
                sys.exit(1)
                
            how_many_appear = klengths[0] # all should be equal anyway
            
            flrm = [fw[k][1:] for k in fw.keys()]  # mark to remove all but the first index
            flrm = reduce(lambda x,y: x+y,flrm)    # flatten it
            
            for ic in sorted( flrm, reverse=True ):
                del fellows[ic]
                del fellows_considered[ic]
                del fellows_iiperm[ic]

            print 'fellows_considered=', fellows_considered
            print 'WARNING : Fellows end-end  ', fellows
            
        
        fellows_unique = [Gall[ic] for ic in fellows_considered]  # some fellows repeat. We will remove that
        fellows_Vtype  = [Vtyp[ic] for ic in fellows_considered]
        fellows_indx   = [Indx[ic] for ic in fellows_considered]
        fellows_Vs     = [f[0] for f in fellows]
        
        Vs_involved = set(reduce(lambda x,y: x+y,fellows_Vs))
        if len(fellows_Vs)*how_many_appear != 2**len(Vs_involved):
            print 'WARNING The number of diagrams combined for Gc=', Gc, 'is not 2^n, hence I can not combine them yet : Vs=', fellows_Vs, ' Vs_involved=', Vs_involved
            print 'fellows are ', fellows_unique, 'with Vtypes', fellows_Vtype
            # Now saving all withouth grouping
            nord = len(fellows_unique[0])/2
            for i0,i in enumerate(fellows_considered):
                G_Hugen.append( Gall[i] )
                Vtype_all.append( Vtyp[i] )
                FIndx.append( Indx[i] )
                loops = len(lp.to_cycles(Gall[i]))
                Signs_all.append( ( (-2)**(loops)*(-1)**nord,) )
                Factor.append( 1 )
            # and removing all
            which_to_remove = sorted( fellows_considered, reverse=True ) # you have to start deleing from the back
            # actually removing the group
            for i in which_to_remove:
                del Gall[i]
                del Vtyp[i]
                del Indx[i]
        else:
            
            flps = [len(lp.to_cycles(f)) for f in fellows_unique] # how many fermionic loops
            
            imin = min(flps)
            a_which = []
            for i in range(len(flps)):
                if flps[i]==imin:
                    a_which.append(i)
            
            i_which = argmin(flps)
            print 'possible diags to choose from with minimal loops =', [fellows_indx[i] for i in a_which], 'choosen', fellows_indx[i_which]

            if (Hugenholtz_All):
                extra=[]
                for vi in Vs_involved:
                    if Hugen_equal_V.has_key(vi):
                        extra += Hugen_equal_V[vi]
                print 'Vs_involved=', Vs_involved, extra, 'Hugen_equal_V=', Hugen_equal_V
                Vs_involved.update(extra)  ##### IMPORTANT CHANGE
            
            fellow_chosen = fellows_unique[i_which]
            cVtyp = fellows_Vtype[i_which]
            iiperm = fellows_iiperm[i_which]
            for j in Vs_involved:
                j_choosen = iiperm[j-1] # careful, since the two are equivalent after transformation involving iiperm, the interactions which are involved in Hugenholts have to be properly permuted too.
                cVtyp[j_choosen] += 10
            
            print 'Vtype=', cVtyp, 'fellow_chosen=', fellow_chosen, 'from all fellows=', fellows_unique, ' with loops=', flps, 'how_many_appear=', how_many_appear, 'iiperm=', iiperm
            print 'choosen with loops=', flps[i_which]
            
            # BUG-Jan 2019 : This generates the order, which is incompatible with C++ sampling. We need the order corresponding to the proper binary tree
            Exchanges = [ i for i,v in enumerate(cVtyp) if v>=10][::-1]
            print 'Exchanges=', Exchanges
            #norder = len(Exchanges)+1
            #ncases = 2**(norder-1) # there are 2^(norder-1) cases of interaction exchange. Let's enumerate both using binary representation of integer.
            Gc = fellow_chosen
            loops=[flps[i_which]]
            _how_many_appear_=[i_which]  # this is used only if Hugenholtz_All
            for i in range(1,2**len(Exchanges)):
                Gp = copy(Gc)                              # the diagram for which we try to find all Hugenholtz equivalents
                Vexchanged=[ Exchanges[j] for j in range(len(Exchanges)) if i & (1<<j) ] # which interactions need to be exchanged for this case, which is one out of 2^(n-1)
                print '   x: Vexchanged=', Vexchanged
                print '   x:', i, ': ',
                for ii in Vexchanged:
                    j1, j2 = 2*ii, 2*ii+1  # these are the vertices that need to be exchanged
                    print 'Exchanging (', j1,',', j2, ') -> (', j2,',',j1,')  ',# 'with Vtyp=', cVtyp,
                    i1, i2 = Gc.index(j1), Gc.index(j2)        # vertices to exchange
                    Gp[i1], Gp[i2] = Gp[i2], Gp[i1]            # with Hugenholtz exchange becomes Gp
                ic = eq.FindEquivalent(Gp, 0, fellows_unique, V0perm)  # finding if the exchanged diagram has an equivalent in the rest of the list. Returns index to it.
                loops.append(flps[ic])
                _how_many_appear_.append(ic)
                print '\n   x:  and got G=', Gp, ' equvalent ic=', ic, 'with loops=', flps[ic]

            if Hugenholtz_All:
                how_many_appear2=[]
                for i in range(len(_how_many_appear_)):
                    how_many_appear2.append( _how_many_appear_.count( _how_many_appear_[i] ) )
            else:
                how_many_appear2=ones(len(loops))
                
            print '_how_many_appear_=', _how_many_appear_, len(fellows_unique), 'how_many_appear2=', how_many_appear2
            nord = len(fellow_chosen)/2
            Signs_all.append( [(-2)**l*(-1)**nord / float(how_many_appear2[il]) for il,l in enumerate(loops)] )
            print 'Signs_all=', Signs_all[-1]
            Vtype_all.append( cVtyp )
            G_Hugen.append( fellow_chosen )
            FIndx.append( fellows_indx[i_which] )
            Factor.append( how_many_appear )
            # actually removing the group
            if how_many_appear != 1:
                print 'WARNING how_many_appear=', how_many_appear, ', which can occurs at order n>=6.'
                
            for i in sorted(fellows_considered, reverse=True ): # you have to start deleing from the back
                del Gall[i]
                del Vtyp[i]
                del Indx[i]
            
    return (G_Hugen, Vtype_all, FIndx, Signs_all, Factor)

def Find_Hugenholtz_old(Glast, Vtyp, indx, V0perm, Viiperm=None):
    Vtype_all=[]
    G_Hugen=[]
    Factor=[]
    FIndx=[]
    Signs_all=[]
    debug = False
    Gall = cp.deepcopy(Glast)
    Indx = cp.copy(indx)
    norder = len(Glast[0])/2
    NotCounter=False
    if sum(Vtyp)==0:
        NotCounter=True

    print 'NotCounter=', NotCounter
    while (Gall):
        ip=0
        Gc = Gall[ip] # starting by checking the first remaining diagram
        if debug :
            print '********************************'
            print 'considering Gc=', Gc, 'Vtype=', Vtyp[ip], 'len(Gall)=', len(Gall)
        fellows=[ ([], Gall[ip] ) ] # this one is the first in the fellow list
        fellows_index0 = [ip]   # and needs to be removed, once we found his group
        fellows_iiperm = [range(1,norder)]
        ncases = 2**(norder-1) # there are 2^(norder-1) cases of interaction exchange. Let's enumerate both using binary representation of integer.
        for i in range(1,ncases):
            Gp = copy(Gc)                              # the diagram for which we try to find all Hugenholtz equivalents
            Vexchanged=[ j+1 for j in range(norder-1) if i & (1<<j) ] # which interactions need to be exchanged for this case, which is one out of 2^(n-1)
            #Vexchanged=[ norder-j-1 for j in range(norder-1) if i & (1<<j) ]
            if debug:
                print 'Vexchanged=', Vexchanged
                print i, ': ',
            for ii in Vexchanged:
                j1, j2 = 2*ii, 2*ii+1  # these are the vertices that need to be exchanged
                if debug:
                    print ' Exchanging (', j1,',', j2, ') -> (', j2,',',j1,')  ', 'with Vtyp=', Vtyp[ip],
                i1, i2 = Gc.index(j1), Gc.index(j2)        # vertices to exchange
                Gp[i1], Gp[i2] = Gp[i2], Gp[i1]            # with Hugenholtz exchange becomes Gp
                
            if NotCounter:
                #ic = FindEquivalent(Gp, 0, Gall, V0perm)
                ic = eq.FindEquivalent(Gp, 0, Gall, V0perm)  # finding if the exchanged diagram has an equivalent in the rest of the list. Returns index to it.
                viiperm = range(1,norder)
            else:
                #ic = FindEquivalent2(Gp, Vtyp[ip], Gall, Vtyp, V0perm, Viiperm)
                ic,ii = eq.FindEquivalent2(Gp, Vtyp[ip], Gall, Vtyp, V0perm, Viiperm)  # finding if the exchanged diagram has an equivalent in the rest of the list. Returns index to it.
                viiperm = Viiperm[ii]
                
            if debug:
                print
                print 'And got', Gp,
            if ic>=0:
                fellows.append( (Vexchanged, Gall[ic] ) ) # this is now new fellow in the group
                fellows_index0.append( ic )           # and needs to be removed from the list of all remaining
                fellows_iiperm.append( viiperm )
                if debug:
                    print 'which is equivalent to', ic, Gall[ic]
            else:
                if debug:
                    ic_all=[]
                    ic=-1
                    while True:
                        ic = eq.FindEquivalent(Gp, ic+1, Gall, V0perm)
                        if ic<0: break
                        ic_all.append(ic)
                        
                    print 'and cannot find equivalent but have',
                    for ic in ic_all:
                        Gall[ic], Vtyp[ic], ' , ',
                    print

        if debug: print 'fellows_index0=', fellows_index0
        
        cVty = Vtyp[ip]
        if False: # This simplistic remove does not work in general, but gives the idea what we want to accomplist, namely remove all duplications.
            flrm=[]
            for i in range(len(fellows_index0)):
                if fellows_index0[i] in fellows_index0[:i]:
                    flrm.append(i)
        else:   # we want to keep the minimal number of Hugenholtz-exchanges so as to get all diagrams in the group of fellows.
            # We want to remove all duplications.
            nrd = len(fellows_index0).bit_length() - 1   # we will achieve : len(fellows_index0) == 2^nrd
            fellows_index1 = [fellows_index0[1<<j] for j in range(nrd)] # We just select terms like 0001,0010,0100,1000, which are sufficient to find possible repetition
            if debug: print 'fellows_index1=', fellows_index1
            rmf=set()
            for i in range(1,nrd):   # did this index i already appear before?
                V_e_i = fellows[1<<i][0][0]  # which interaction was exchanged to get fellow[i]
                for j in range(i):   # Check if two independent exchanges give the same diagram.
                    if fellows_index1[j]==fellows_index1[i]: # If yes, remove all exchanges associated with the second exchange.
                        V_e_j = fellows[1<<j][0][0] # which interaction exchanged to get fellow[j]
                        if cVty[V_e_j] > cVty[V_e_i]:  # level of counter-term for fellow[i] and fellow[j]
                            rmf.update( filter(lambda x: x&(1<<j), range((1<<nrd))) ) # remove the term with higher counterterm
                        else:
                            rmf.update( filter(lambda x: x&(1<<i), range((1<<nrd))) )
            flrm = list(rmf)
            
        if debug:
            print 'flrm=', flrm
            print 'Fellows start', fellows
        
        fellows_considered = fellows_index0[:]
        # Now we remove the repetition, and keep only the essential group of diagrams, which are all distinct
        for ic in sorted( flrm, reverse=True ):
            del fellows[ic]
            del fellows_considered[ic]
            del fellows_iiperm[ic]
        if debug:
            print 'Fellows end  ', fellows

        how_many_appear = 1
        fellows_considered2 = set(fellows_considered)
        if len(fellows_considered) != len(fellows_considered2):
            fw = {}
            for i in range(len(fellows_considered)):
                fc = fellows_considered[i]
                if fw.has_key(fc):
                    fw[fc].append(i)
                else:
                    fw[fc]=[i]
            klengths = [len(fw[k]) for k in fw.keys()]
            if klengths[1:] != klengths[:-1]: # check that all klengths are equal
                print 'ERROR : We still have repetitions and not all terms appear equal number of times: ', klengths
                print '      : Here fw=', fw
                print '      : fellows_considered=', '('+str(len(fellows_considered))+')', fellows_considered
                print '      : while fellows_considered2=', '('+str(len(fellows_considered2))+')', fellows_considered2
                sys.exit(1)
                
            how_many_appear = klengths[0] # all should be equal anyway
            
            flrm = [fw[k][1:] for k in fw.keys()]  # mark to remove all but the first index
            flrm = reduce(lambda x,y: x+y,flrm)    # flatten it
            
            for ic in sorted( flrm, reverse=True ):
                del fellows[ic]
                del fellows_considered[ic]
                del fellows_iiperm[ic]

            print 'fellows_considered=', fellows_considered
            print 'WARNING : Fellows end-end  ', fellows
            
        
        fellows_unique = [Gall[ic] for ic in fellows_considered]  # some fellows repeat. We will remove that
        fellows_Vtype  = [Vtyp[ic] for ic in fellows_considered]
        fellows_indx   = [Indx[ic] for ic in fellows_considered]
        fellows_Vs     = [f[0] for f in fellows]
        
        Vs_involved = set(reduce(lambda x,y: x+y,fellows_Vs))
        if len(fellows_Vs)*how_many_appear != 2**len(Vs_involved):
            print 'WARNING The number of diagrams combined for Gc=', Gc, 'is not 2^n, hence I can not combine them yet : Vs=', fellows_Vs, ' Vs_involved=', Vs_involved
            print 'fellows are ', fellows_unique, 'with Vtypes', fellows_Vtype
            # Now saving all withouth grouping
            nord = len(fellows_unique[0])/2
            for i0,i in enumerate(fellows_considered):
                G_Hugen.append( Gall[i] )
                Vtype_all.append( Vtyp[i] )
                FIndx.append( Indx[i] )
                loops = len(lp.to_cycles(Gall[i]))
                Signs_all.append( ( (-2)**(loops)*(-1)**nord,) )
                Factor.append( 1 )
            # and removing all
            which_to_remove = sorted( fellows_considered, reverse=True ) # you have to start deleing from the back
            # actually removing the group
            for i in which_to_remove:
                del Gall[i]
                del Vtyp[i]
                del Indx[i]
        else:
            
            flps = [len(lp.to_cycles(f)) for f in fellows_unique] # how many fermionic loops
            
            imin = min(flps)
            a_which = []
            for i in range(len(flps)):
                if flps[i]==imin:
                    a_which.append(i)
            
            i_which = argmin(flps)
            print 'possible diags to choose from with minimal loops =', [fellows_indx[i] for i in a_which], 'choosen', fellows_indx[i_which]

            fellow_chosen = fellows_unique[i_which]
            cVtyp = fellows_Vtype[i_which]
            iiperm = fellows_iiperm[i_which]
            for j in Vs_involved:
                j_choosen = iiperm[j-1] # careful, since the two are equivalent after transformation involving iiperm, the interactions which are involved in Hugenholts have to be properly permuted too.
                cVtyp[j_choosen] += 10
            
            print 'Vtype=', cVtyp, 'fellow_chosen=', fellow_chosen, 'from all fellows=', fellows_unique, ' with loops=', flps, 'how_many_appear=', how_many_appear, 'iiperm=', iiperm
            print 'choosen with loops=', flps[i_which]
            
            # BUG-Jan 2019 : This generates the order, which is incompatible with C++ sampling. We need the order corresponding to the proper binary tree
            Exchanges = [ i for i,v in enumerate(cVtyp) if v>=10][::-1]
            print 'Exchanges=', Exchanges
            #norder = len(Exchanges)+1
            #ncases = 2**(norder-1) # there are 2^(norder-1) cases of interaction exchange. Let's enumerate both using binary representation of integer.
            Gc = fellow_chosen
            loops=[flps[i_which]]
            for i in range(1,2**len(Exchanges)):
                Gp = copy(Gc)                              # the diagram for which we try to find all Hugenholtz equivalents
                Vexchanged=[ Exchanges[j] for j in range(len(Exchanges)) if i & (1<<j) ] # which interactions need to be exchanged for this case, which is one out of 2^(n-1)
                print '   x: Vexchanged=', Vexchanged
                print '   x:', i, ': ',
                for ii in Vexchanged:
                    j1, j2 = 2*ii, 2*ii+1  # these are the vertices that need to be exchanged
                    print 'Exchanging (', j1,',', j2, ') -> (', j2,',',j1,')  ',# 'with Vtyp=', cVtyp,
                    i1, i2 = Gc.index(j1), Gc.index(j2)        # vertices to exchange
                    Gp[i1], Gp[i2] = Gp[i2], Gp[i1]            # with Hugenholtz exchange becomes Gp
                ic = eq.FindEquivalent(Gp, 0, fellows_unique, V0perm)  # finding if the exchanged diagram has an equivalent in the rest of the list. Returns index to it.
                loops.append(flps[ic])
                print '\n   x:  and got G=', Gp, ' equvalent ic=', ic, 'with loops=', flps[ic]

            nord = len(fellow_chosen)/2
            Signs_all.append( [(-2)**l*(-1)**nord / how_many_appear for l in loops] )
            Vtype_all.append( cVtyp )
            G_Hugen.append( fellow_chosen )
            FIndx.append( fellows_indx[i_which] )
            Factor.append( how_many_appear )
            # actually removing the group
            for i in sorted(fellows_considered, reverse=True ): # you have to start deleing from the back
                del Gall[i]
                del Vtyp[i]
                del Indx[i]
            
    return (G_Hugen, Vtype_all, FIndx, Signs_all, Factor)


if __name__ == '__main__':
    DynamicCT = False
    Hugenholtz = True
    execfile('params.py')
    exec( 'DynamicCT = '+str(p['DynamicCT']) )
    Hugenholtz = p['Hugenholtz']
    
    if len(sys.argv)<2:
        print 'Give input filename'
        sys.exit(1)
    filename = sys.argv[1]
    
    fi = open(filename,'r')
    diagP=[]
    for lin in fi:
        ii, sdiagP = lin.split(' ', 1)
        #print ii
        diagP.append( eval(sdiagP) )
    print '# diagrams=', len(diagP)

    n = len(diagP[0])/2
    norder=n
    print 'I think the order is', n
    
    V0 = [(2*i,2*i+1) for i in range(n)]
    if n==1:
        V0perm=[]
    else:
        V0perm = GiveInteractionPermutations_new(V0[1:])  # all equivalent forms of interactions
        V0perm = [reduce(lambda x,y: x+y, V0[1:])] + V0perm
        print 'interaction permtations are', V0perm

    Viiperm={}
    Vperm={}
    for ni in range(2,n+1):
        Vt0 = [(2*i,2*i+1) for i in range(ni)]
        Vtperm = GiveInteractionPermutations_new(Vt0[1:])  # all equivalent forms of interactions
        Vtperm = [reduce(lambda x,y: x+y, Vt0[1:])] + Vtperm
        Vperm[ni] = Vtperm
        Viiperm[ni] = []
        for i in range(len(Vtperm)):
            Vtiiperm = [(Vtperm[i][2*j]+Vtperm[i][2*j+1])/4 for j in range(ni-1)] 
            #print 'i=', i, 'Vtperm=', Vtperm[i], 'Vtiiperm=', Vtiiperm
            Viiperm[ni].append( Vtiiperm )


    ### new
    if not Hugenholtz:
        Gnew, indx = FindFunctionals(diagP,V0perm)
        containsB = WhichContainBubbles(Gnew)
        pointer = sorted(range(len(indx)), key= lambda i: indx[i]) # want to sort therm according to their index
        pointer = sorted(pointer, key= lambda i: 1-containsB[i])   # but even more important is if they contain bubble
        Glast = [Gnew[i] for i in pointer]  # finally, the sorted diagrams
        indx = [indx[i] for i in pointer]   # their index
        containsB = [containsB[i] for i in pointer] # and if they contain a bubble
        
        Gcounter_start = [Glast[i] for i in range(len(Glast)) if containsB[i]]     # 
        indx_counter_start = [indx[i] for i in range(len(indx)) if containsB[i]]   #

        if len(Gcounter_start)>0:
            Gall_corder, Vtyp_corder, Indx_corder = CreateAllCounterTerms(Gcounter_start, indx_counter_start, V0, Vperm, Viiperm, AllCounters=(not DynamicCT))
            print 'CounterTerms='
            for nord in Gall_corder.keys():
                print 'Dimension=', nord
                _indx_ = sorted(range(len(Gall_corder[nord])), key=lambda i: Indx_corder[nord][i])

                outfile = os.path.join(os.path.dirname(filename), os.path.basename(filename)+'_corder_'+str(nord))
                fco = open(outfile, 'w')
                for i in range(len(Gall_corder[nord])):
                    print >> fco, Indx_corder[nord][i], tuple(Gall_corder[nord][i]), ' ; ', tuple(Vtyp_corder[nord][i])
                fco.close()
        
        outfile = os.path.join(os.path.dirname(filename), os.path.basename(filename)+'_bubbles')
        fo = open(outfile, 'w')
        for i, d in enumerate(Glast):
            if containsB[i]:
                print >> fo, indx[i], d
        fo.close()
        outfile = os.path.join(os.path.dirname(filename), os.path.basename(filename)+'_nobubble')
        fo = open(outfile, 'w')
        #print >> fo, '# ----------------- above have bubbles'
        for i, d in enumerate(Glast):
            if not containsB[i]:
                print >> fo, indx[i], d
        fo.close()
        print '# diagrams=', len(Glast)
    else:
        Hugenholtz_All = False
        if Hugenholtz == 'All':
            Hugenholtz_All = True
        
        Gnew, indx = FindFunctionals(diagP,V0perm)
        containsB = WhichContainBubbles(Gnew)
        pointer = sorted(range(len(indx)), key= lambda i: indx[i]) # want to sort therm according to their index
        Glast = [Gnew[i] for i in pointer]  # finally, the sorted diagrams
        indx = [indx[i] for i in pointer]   # their index
        containsB = [containsB[i] for i in pointer] # and if they contain a bubble

        Gcounter_start = [Glast[i] for i in range(len(Glast)) if containsB[i]]     # 
        indx_counter_start = [indx[i] for i in range(len(indx)) if containsB[i]]   #

        if len(Gcounter_start)>0:
            Gall_corder, Vtyp_corder, Indx_corder = CreateAllCounterTerms(Gcounter_start, indx_counter_start, V0, Vperm, Viiperm, AllCounters=(not DynamicCT))
                
            for nord in Gall_corder.keys():
                #print 'nord=', nord, 'G_corder=', Gall_corder[nord], 'Vtyp=', Vtyp_corder[nord]
                (G_Hugen, Vtyp, FIndx, Signs, Factor) = Find_Hugenholtz(Gall_corder[nord], Vtyp_corder[nord], Indx_corder[nord], Vperm[nord], Hugenholtz_All, Viiperm[nord])
                
                print 'len(G_Hugen)=', len(G_Hugen), 'len(Vtyp)=', len(Vtyp), 'len(Signs)=', len(Signs), 'len(FIndx)=', len(FIndx)
                
                _indx_ = sorted(range(len(G_Hugen)), key=lambda i: FIndx[i])
                outfile = os.path.join(os.path.dirname(filename), os.path.basename(filename)+'_cworder_'+str(nord))
                fco = open(outfile, 'w')
                for i in _indx_: 
                    print >> fco, FIndx[i], tuple(G_Hugen[i]), ' ; ', tuple(Vtyp[i]), ' ; ', tuple(Signs[i])
                fco.close()
            
                hug_num = 0
                for i in range(len(G_Hugen)):
                    deg = len(filter(lambda x: x>=10, Vtyp[i] ))
                    real_deg = 2**deg/Factor[i]
                    hug_num += real_deg
                print
                print '('+str(nord)+')', '# diagrams=', len(Gall_corder[nord]), ' #Hugen=', len(G_Hugen), '#all-Hugen=', hug_num

        
        Vtype = [zeros(norder, dtype=int) for i in range(len(Glast))]
        (G_Hugen, Vtype_all, FIndx, Signs, Factor) = Find_Hugenholtz(Glast, Vtype, indx, V0perm, Hugenholtz_All)

        outfile = os.path.join(os.path.dirname(filename), os.path.basename(filename)+'_cworder_0')
        fo = open(outfile, 'w')

        _indx_ = sorted(range(len(G_Hugen)), key=lambda i: FIndx[i])
        for i in _indx_:
            print >> fo, FIndx[i], tuple(G_Hugen[i]), ' ; ', tuple(Vtype_all[i]), ' ; ', tuple(Signs[i])
        fo.close()

        hug_num = 0
        for i in range(len(G_Hugen)):
            deg = len(filter(lambda x: x>=10, Vtype_all[i] ))
            real_deg = 2**deg/Factor[i]
            hug_num += real_deg
        print
        print '# diagrams=', len(Glast), ' #Hugen=', len(G_Hugen), '#all-Hugen=', hug_num
    
