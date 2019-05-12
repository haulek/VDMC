#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
import sys
import re
import os
from loops import *
import numpy


def SignDiagram(diagsG):
    diagSign=[]
    for id,d in enumerate(diagsG):
        cc = to_cycles(d)
        N_ferm_loop = len(cc)
        diagSign.append( 2**N_ferm_loop * (-1)**(N_ferm_loop + Norder) )
    return diagSign
def FindPrimeNumbers(lower,upper):
    pm=[]
    for num in range(lower,upper + 1):
       if num > 1:
           for i in range(2,num):
               if (num % i) == 0:
                   break
           else:
               pm.append(num)
    return pm

def FindAllVertices(diagsG, diagsV):
    VertexTypeAll=[]
    for id in range(len(diagsG)):
        # The inverse of the perturbation, which corresponds to following the fermionic propagators in opposite direction than arrows
        diag = diagsG[id]
        i_diag = zeros(len(diag),dtype=int)
        for i in range(len(diag)): i_diag[diag[i]]=i

        type_1_vertices=[]
        type_2_vertices=[]
        VertexType = []
        for it in range(1,len(diagsV)):
            v1,v2 = diagsV[it]
            v1_next = diag[v1]
            v1_previous = i_diag[v1]
            v2_next = diag[v2]
            v2_previous = i_diag[v2]
            
            if (v1_next not in type_1_vertices) and (v1_previous not in type_1_vertices):
                type_1_vertices.append( v1 )
                VertexType.append( [v1,1] )
            elif (v2_next not in type_1_vertices) and (v2_previous not in type_1_vertices):
                type_1_vertices.append( v2 )
                VertexType.append( [v2,1] )
            else:
                type_2_vertices.append( v1 )
                VertexType.append( [v1,2] )
        #print id, diag, 'type=', VertexType, 'type2=', type_2_vertices
        VertexTypeAll.append( VertexType )
    return VertexTypeAll


def Transform2Phi(Gp):
    Gn = Gp[:]
    Gn[0]=0; Gn[1]=1
    for i in range(2,len(Gp)):
        if Gp[i]==0 and Gp[Gp[i]]!=1 or Gp[i]==1 and Gp[Gp[i]]!=0:
            Gn[i] = Gp[Gp[i]]
        if Gp[i]==0 and Gp[Gp[i]]==1 or Gp[i]==1 and Gp[Gp[i]]==0:
            Gn[i] = Gp[Gp[Gp[i]]]
    return Gn    

def CompareSimilar(Phis,loop1,loop2,type1,type2,debug=False,shorten=False):
    if Phis:
        _loop1_ = [ i for i in (loop1) if i not in (0,1) ]  # removing vertices 0&1 from the loop to get the corresponding generating functional
        _loop2_ = [ i for i in (loop2) if i not in (0,1) ]  # removing vertices 0&1 from the loop to get the corresponding generating functional
        if _loop1_ == _loop2_ : # if all things equal, prefer loops which do not go through vertex 0 for Baym-Kadanoff approach
            if len(loop1)==len(loop2):
                is_zero_in_loop1 = 0 in loop1
                is_zero_in_loop2 = 0 in loop2
                if is_zero_in_loop1 != is_zero_in_loop2: 
                    if is_zero_in_loop1:
                        return 2
                    else:
                        return 1
                else:                         # if all other things equal,
                    if sum(type1)>sum(type2): # prefer loops in positive direction
                        return 1
                    else:
                        return 2
            if len(loop1)<len(loop2):
                return 1
            else:
                return 2
        if shorten:
            for il in xrange(len(Phis)):
                if debug:
                    print '  :Compare similarity of loop1=', _loop1_, 'amd loop2=', _loop2_,'with Phis=', Phis[il]
                    print '   but here we shorthen to loop1=',_loop1_,'with Phi=', Phis[il][:len(_loop1_)], ' and loop2=', _loop2_, 'with', Phis[il][:len(_loop2_)]
                if Phis[il][:len(_loop1_)] == _loop1_:
                    return 1
                if Phis[il][:len(_loop2_)] == _loop2_:
                    return 2
        else:
            for il in xrange(len(Phis)):
                if debug:
                    print '  :Compare similarity of loop1=', _loop1_,  'and loop2=', _loop2_, 'with Phis=', Phis[il]
                if Phis[il] == _loop1_:
                    return 1
                if Phis[il] == _loop2_:
                    return 2
    return None
def StoreIfUnique(loops, Phi_loop_it):
    phi_loop = [ i for i in (loops) if i not in (0,1) ] # remove measssuring vertices
    Already_stored=False
    for previous_phi_loop in Phi_loop_it:
        if previous_phi_loop==phi_loop:
            Already_stored=True
            break
    if not Already_stored:
        Phi_loop_it.append(phi_loop)


def FindSimilarOrShortestPath(V_vertices, v_final, diag, i_diag, Phis, debug=False, shorten=False):
    """Given two interaction vertices V_vertices[0..1] and another vertex v_final in the same fermionic loop as V_vertices[1]
       it gives the closest path from V_vertices[1] to v_final. We could follow the fermionic propagators in either positive or
       negative direction.
       For positive direction of fermionic propagator type=+1, for negative direction of fermionic propagator type=-1
       for positive direction of bosonic (interaction) propagator type=+2
       loop will contain the list of vertices visited
       type will show if the connection between successive vertices is fermionic (+1 or -1) of bosonic (+2 or -2).
    """
    # interaction V(v1,v2)
    if debug: print 'Debug V_vertices=', V_vertices, ' should find way to v_final=', v_final
    loop1 = [V_vertices[0]]
    type1 = [2 * sign(V_vertices[1]-V_vertices[0])]
    i1 = V_vertices[1]
    while i1!=v_final:
        loop1.append(i1)
        type1.append(1)
        i1 = diag[i1]
    loop2 = [V_vertices[0]]
    type2 = [2 * sign(V_vertices[1]-V_vertices[0])]
    i1 = V_vertices[1]
    while i1!=v_final:
        loop2.append(i1)
        type2.append(-1)
        i1 = i_diag[i1]

    which = CompareSimilar(Phis,loop1,loop2,type1,type2,debug=debug,shorten=shorten)
    if debug:
        print '  We were comparing loop1=', loop1, ' and loop=2', loop2, ' to Phis=', Phis, 'and decided on which iloop=', which
    if which is None:
        if len(loop1)!=len(loop2):
            if len(loop1)<len(loop2):
                which=1
            else:
                which=2
        else:
            _loop1_ = Strip_01(loop1)
            _loop2_ = Strip_01(loop2)
            if debug and len(Phis): print '   ###Here comparing stripped loop1=', _loop1_, 'and stripped loop2=', _loop2_, 'to Phi=', Phis[0]
            if len(Phis) and _loop1_ == Phis[0]:
                which = 1
            elif len(Phis) and _loop2_ == Phis[0]:
                which = 2
            else:
                is_zero_in_loop2 = 0 in loop2
                if is_zero_in_loop2:
                    which = 1
                else:
                    which = 2
    if debug:
        print '  We were comparing loop1=', loop1, ' and loop=2', loop2, ' to Phis=', Phis, 'and decided on which iloop=', which
        
    if which==1:
        return loop1, type1
    else:
        return loop2, type2


def GivePhiCycles(cycles):
    phi_cycles = [sorted(c[:]) for c in cycles[:]]
    for c in phi_cycles:
        if 0 in c: c.remove(0)
        if 1 in c: c.remove(1)
    return phi_cycles


def Resort_If_Different(cycles, phi_cycles0):
    phi_cycles = GivePhiCycles(cycles)
    #print 'phi_cycles=', phi_cycles, 'phi_cycles0=', phi_cycles0
    if phi_cycles==phi_cycles0 or len(phi_cycles)!=len(phi_cycles0):
        return cycles
    else:
        indx=range(len(phi_cycles))
        for i in range(len(phi_cycles)):
            for j in range(len(phi_cycles)):
                if phi_cycles[i]==phi_cycles0[j]:
                    indx[j]=i
                    break
        new_cycles=[]
        for i in range(len(indx)):
            new_cycles.append( cycles[indx[i]] )
        return new_cycles

def Strip_01(loop):
    _loop_ = loop[:]
    if 0 in _loop_: _loop_.remove(0)
    if 1 in _loop_: _loop_.remove(1)
    return _loop_

    
def FindAllLoopsN(diagsG, diagsV, indx, debug):
    """ It goes over all diagrams, and finds all necessary loops for each diagram.
    Each fermionic loop contributes one loop. In addition, each interaction line contributes one loop (except in some very high symmetry cases).
    It can connect up to four fermionic loops for single momentum loop, but not yet more than four. At very high order one needs to increase that.
    """
    Ndiags = len(diagsG)
    Loop_vertex=[]
    Loop_type=[]
    indx_previous=-1
    Norder = len(diagsG[0])/2
    for id in range(Ndiags):
        diag = diagsG[id]
        # Cycles give fermionic loops.
        cycles = to_cycles(diag)
        
        if indx[id]!= indx_previous: # new type of phi
            # We will try to make loops similar for the diagrams which are derived from the same Phi functional
            phi = Transform2Phi(list(diag))   # What is the deriving functional
            N_cycles = len(to_cycles(phi)[2:])   # How many fermionic loops do we have
            N_cycles = max(1,N_cycles)
            Phi_loop = [[] for i in range(len(diag)/2+N_cycles)]  # there are max so many different momenta
            #print 'The size of Phi_loop=', len(diag)/2+N_cycles, 'because', len(diag)/2, N_cycles, len(Phi_loop)
            indx_previous = indx[id]          # now check which diagram is part of the same functional
            phi_cycles0 = GivePhiCycles(cycles)

        print 'cycles_old=', cycles, 'phi_cycles0=', phi_cycles0
        cycles = Resort_If_Different(cycles, phi_cycles0)
        if debug: print 'cycles_new=', cycles
        
        if debug: print '****** Analizing diagram', diag, 'with phi=', phi, 'and indx=', indx[id]
        # The inverse of the perturbation, which corresponds to following the fermionic propagators in opposite direction than arrows
        i_diag = zeros(len(diag),dtype=int)
        for i in range(len(diag)): i_diag[diag[i]]=i

        #print 'diag=', diag, 'cycles=', cycles
        
        # For each vertex we want to know in each fermionic loop is located
        which_loop = zeros(len(diag), dtype=int) # this vertex is in which fermionic loop?
        for i,c in enumerate(cycles):
            for j in range(len(c)): which_loop[c[j]]=i

        FermiLoopsConnected={}
        for it in range(1,len(diagsV)):  # over all interactions but external loop
            v1,v2 = diagsV[it]
            il1, il2 = which_loop[v1], which_loop[v2] # in which loop are the two vertices
            if (il1!=il2):  # loop connects two fermionic loops
                ic1,ic2 = min(il1,il2),max(il1,il2)
                if FermiLoopsConnected.has_key( (ic1,ic2) ):
                    FermiLoopsConnected[(ic1,ic2)].append(it)
                else:
                    FermiLoopsConnected[(ic1,ic2)] = [it]
        
        if debug:
            print 'cycles=', cycles, 'which_loop=', which_loop, 'Phis=', Phi_loop[:]
            print 'FermiLoopsConnected=', FermiLoopsConnected
            
        how_many_interaction_loops = Norder - len(cycles)
        
        loop_vertex=[]
        loop_type=[]
        #skipV=[]
        V_already_used=[]
        # first create mixed loops, which contain at least one interaction line, and several fermionic lines
        for it in range(len(diagsV)):
            #if it in skipV: continue # this interaction was already used in another loop, and does not need to be considered again
            v1,v2 = diagsV[it]
            if which_loop[v1]==which_loop[v2]:
                # interaction is within the same fermionic loop, hence we just follow fermionic propagators to close the loop
                loops, types = FindSimilarOrShortestPath( (v1,v2), v1, diag, i_diag, Phi_loop[it], debug=debug)
            else:
                # at least two fermionic loops are connected by this momentum
                il1, il2 = which_loop[v1], which_loop[v2]
                ic1,ic2 = min(il1,il2),max(il1,il2)

                if it>0:
                    it_already_used = any([it in used for used in V_already_used])
                    if it_already_used:
                        if debug: print '    Because this interaction', tuple([v1,v2]), 'was already used, we skip it '
                        continue

                FoundConnection = {}
                if FermiLoopsConnected.has_key( (ic1,ic2) ):
                    for jt in FermiLoopsConnected[(ic1,ic2)]:
                        if it==jt: continue
                        if jt>it and it>FermiLoopsConnected[(ic1,ic2)][0]: continue
                        v3, v4 = diagsV[jt]
                        if (which_loop[v3]==il2 and which_loop[v4]==il1):
                            FoundConnection[jt]=(v3,v4)  # diagsV[jt] connects the same fermionic loops as diagsV[it]
                        elif (which_loop[v4]==il2 and which_loop[v3]==il1):
                            FoundConnection[jt]=(v4,v3)
                    
                if debug: print '  While working with interaction ', diagsV[it], 'Found connections by interaction =', FoundConnection
                if FoundConnection:
                    possible_loops=[]
                    possible_types=[]
                    possible_V=[]
                    for jt in FoundConnection.keys():
                        if debug: print 'connection ', FoundConnection[jt]
                        if tuple(sorted((it,jt))) in V_already_used:
                            if debug: print '.... But was already used, hence skipping it'
                            continue
                        v3, v4 = FoundConnection[jt]
                        # we need to consider exactly two fermionic loops
                        ph1=[]; ph2=[]
                        if it>0 and len(Phi_loop[it]):
                            ph = Phi_loop[it][0]
                            if [v1,v2] == ph[:2] and v3 in ph:  # In this case I can properly split Phi into two pices
                                ir = ph.index(v3)
                                ph1 = ph[:ir]     # up to next interaction vertex
                                ph2 = ph[ir:]  # need to add the last vertex, because it is missing by default
                                print '@@v1,v2,...,v3=', v1,v2,v3, 'ph=', ph[:ir+1]
                                print '@@v3,v4,...,v1=', v3,v4,v1, 'ph=', ph[ir:]+[v1]
                        if ph1 and ph2:
                            loops1, types1 = FindSimilarOrShortestPath( (v1,v2), v3, diag, i_diag, [ph1], debug=debug, shorten=False)
                            loops2, types2 = FindSimilarOrShortestPath( (v3,v4), v1, diag, i_diag, [ph2], debug=debug, shorten=False)
                        else:
                            loops1, types1 = FindSimilarOrShortestPath( (v1,v2), v3, diag, i_diag, Phi_loop[it], debug=debug, shorten=True)
                            loops2, types2 = FindSimilarOrShortestPath( (v3,v4), v1, diag, i_diag, Phi_loop[it], debug=debug, shorten=True)
                        possible_loops.append( loops1+loops2 )
                        possible_types.append( types1+types2 )
                        possible_V.append( jt )
                    if len(possible_loops)==0:
                        if debug: print '     We did not find any possible connection at this point'
                        continue
                    # See if we can optimize for Baym-Kadanoff approach, by avoid to change k of the zeroth propagator.
                    # We would like zeroth propagator for G to be only part of loop1, and none else.
                    igood = 0
                    if len(possible_loops) > 1: # otherwise we have no choice anyway
                        # Just try to choose the shortest
                        # But here we will meassure length in the absence of 0,1 vertices
                        lngths = zeros(len(possible_loops),dtype=int)
                        for i,loop in enumerate(possible_loops):
                            lng = len(loop)
                            if 0 in loop: lng-=1
                            if 1 in loop: lng-=1
                            lngths[i] = lng
                        #lngths = array([len(loop) for loop in possible_loops])
                        
                        igoods = where(lngths==lngths.min())[0] # which are among the shortest. Could me several
                        if debug: print 'lngths=', lngths, 'igoods=', igoods
                        if len(igoods)==1:
                            igood = igoods[0] # just take the shortest
                        else:
                            FoundGood=False
                            if len(Phi_loop[it])>0:
                                for ii in igoods:   # Checking if any of these is equal to Phi, which appears in all other diagrams
                                    loop = possible_loops[ii]
                                    _loop_ = Strip_01(loop)
                                    if _loop_ == Phi_loop[it][0]:
                                        igood = ii
                                        FoundGood=True
                                        break
                            if not FoundGood:
                                if it != 0: # this is not external loop
                                    for ii in igoods: # We just want to avoid vertex 0 if none is in Phi
                                        if 0 not in loop:
                                            igood = ii
                                            break
                                else: # this is the external momentum loop, hence vertex 0 can not be avoided.
                                    # We want the propagator to come into vertex 0, not out of it.
                                    for ii in igoods:
                                        loop = possible_loops[ii]
                                        v0 = loop.index( 0 )               # found index for vertex 0
                                        v0_type = possible_types[ii][v0-1] # is propagator comming in or going out of vertex 0?
                                        if v0_type > 0:
                                            igood = ii
                                            break
                    loops = possible_loops[igood]
                    types = possible_types[igood]
                    if it>0:
                        V_already_used.append( tuple(sorted([it,possible_V[igood]])) ) # external momentum does not count here
                    
                    if debug:
                        print 'We have here possible loops=', possible_loops, 'with types=', possible_types, 'and we choose loop', loops, 'with type', types
                        print 'We already used the following interactions=', V_already_used
                else:
                    if debug:
                        print '!! Did not yet found connection between v1=', v1, 'v2=', v2, 'for interaction', it
                        print 'We need connection from loop', il2, 'to loop', il1

                    FoundConnection=[]
                    for iln in range(len(cycles)):  # Are il1 and il2 loops connected through iln?
                        if iln==il1 or iln==il2: continue
                        il2_2_iln = tuple(sorted([il2,iln]))
                        iln_2_il1 = tuple(sorted([iln,il1]))
                        if FermiLoopsConnected.has_key(il2_2_iln) and FermiLoopsConnected.has_key(iln_2_il1): 
                            jt_all = FermiLoopsConnected[ il2_2_iln ]
                            kt_all = FermiLoopsConnected[ iln_2_il1 ]
                            if debug:
                                print 'jt_all=', jt_all, 'kt_all=', kt_all
                            for jt in jt_all:
                                if jt==it: continue
                                for kt in kt_all:
                                    if kt==it: continue
                                    conn = [jt,kt,it]
                                    if tuple(sorted(conn)) not in V_already_used:
                                        FoundConnection.append(conn)
                            if debug:
                                print 'FoundConnection is now', FoundConnection
                            
                    if not FoundConnection:
                        if debug:
                            print '!!!! Still did not found connection between v1=', v1, 'v2=', v2, 'for interaction', it
                            print 'We need connection between the loops', il1, il2

                        print 'FermiLoopsConnected=', FermiLoopsConnected
                        for iln in range(len(cycles)):
                            for ilm in range(len(cycles)):
                                if iln==il1 or iln==il2 or ilm==il1 or ilm==il2 or iln==ilm: continue
                                il2_2_iln = tuple(sorted([il2,iln]))
                                iln_2_ilm = tuple(sorted([iln,ilm]))
                                ilm_2_il1 = tuple(sorted([ilm,il1]))

                                print 'il2_2_iln=', il2_2_iln, 'iln_2_ilm=', iln_2_ilm, 'ilm_2_il1=', ilm_2_il1
                                
                                if FermiLoopsConnected.has_key(il2_2_iln) and FermiLoopsConnected.has_key(iln_2_ilm) and FermiLoopsConnected.has_key(ilm_2_il1):
                                    jt_all = FermiLoopsConnected[ il2_2_iln ]
                                    kt_all = FermiLoopsConnected[ iln_2_ilm ]
                                    lt_all = FermiLoopsConnected[ ilm_2_il1 ]
                                    for ii in range(len(jt_all)): # now we take the first one, which is not it
                                        jt = jt_all[ii]
                                        if jt!=it: break
                                    for ii in range(len(kt_all)): # also take the first, which is not it
                                        kt = kt_all[ii]
                                        if kt!=it: break
                                    for ii in range(len(lt_all)): # also take the first, which is not it
                                        lt = lt_all[ii]
                                        if lt!=it: break
                                    if tuple(sorted([jt,kt,lt,it])) not in V_already_used:
                                        FoundConnection.append([jt,kt,lt,it])  # diagsV[jt] & diagsV[kt] connect the same fermionic loops as diagsV[it]
                                        break
                    if debug: print 'FoundConnection=', FoundConnection
                    # Now we create the path through interactions V[it],V[jt],V[kt],V[lt], and back to V[it]
                    possible_loops=[]
                    possible_types=[]
                    for conn in FoundConnection:
                        xloop=[]
                        xtype=[]
                        v1,v2 = diagsV[it]
                        for jt in conn:
                            iloop_start = which_loop[v2]
                            if which_loop[diagsV[jt][0]]==iloop_start:
                                v3 = diagsV[jt][0]
                                v4 = diagsV[jt][1]
                            else:
                                v3 = diagsV[jt][1]
                                v4 = diagsV[jt][0]

                            phi=[]
                            if it>0 and len(Phi_loop[it]):
                                ph = Phi_loop[it][0]

                                if v1 in ph and v3 in ph:
                                    i1=ph.index(v1)
                                    i3=ph.index(v3)
                                    phi=ph[i1:i3]
                                    print '@@@v1,v2,...,v3=', v1,v2,v3, 'ph=', phi
                            if phi:
                                loop1, type1 = FindSimilarOrShortestPath( (v1,v2), v3, diag, i_diag, [phi], debug=debug, shorten=False)
                            else:
                                loop1, type1 = FindSimilarOrShortestPath( (v1,v2), v3, diag, i_diag, Phi_loop[it], debug=debug, shorten=True)
                            if debug: 
                                print 'v1,v2=', v1, v2, 'v3,v4=', v3,v4
                                print 'loop1=', loop1, 'type1=', type1
                            xloop += loop1
                            xtype += type1
                            v1,v2 = v3,v4
                        possible_loops.append( xloop )
                        possible_types.append( xtype )
                    if debug:
                        print '....Possible loops are', possible_loops, 'with types', possible_types
                    igood = 0
                    if len(possible_loops) > 1: # otherwise we have no choice anyway
                        # Just try to choose the shortest
                        # But here we will meassure length in the absence of 0,1 vertices
                        lngths = zeros(len(possible_loops),dtype=int)
                        for i,loop in enumerate(possible_loops):
                            lng = len(loop)
                            if 0 in loop: lng-=1
                            if 1 in loop: lng-=1
                            lngths[i] = lng
                        #lngths = array([len(loop) for loop in possible_loops])
                        igoods = where(lngths==lngths.min())[0] # which are among the shortest. Could me several
                        if debug: print 'lngths=', lngths, 'igoods=', igoods
                        if len(igoods)==1:
                            igood = igoods[0] # just take the shortest
                        else:
                            FoundGood=False
                            if len(Phi_loop[it])>0:
                                for ii in igoods:   # Checking if any of these is equal to Phi, which appears in all other diagrams
                                    loop = possible_loops[ii]
                                    _loop_ = Strip_01(loop)
                                    if _loop_ == Phi_loop[it][0]:
                                        igood = ii
                                        FoundGood=True
                                        break
                            if not FoundGood:
                                if it != 0: # this is not external loop
                                    for ii in igoods: # We just want to avoid vertex 0 if none is in Phi
                                        if 0 not in loop:
                                            igood = ii
                                            break
                                else: # this is the external momentum loop, hence vertex 0 can not be avoided.
                                    # We want the propagator to come into vertex 0, not out of it.
                                    for ii in igoods:
                                        loop = possible_loops[ii]
                                        v0 = loop.index( 0 )               # found index for vertex 0
                                        v0_type = possible_types[ii][v0-1] # is propagator comming in or going out of vertex 0?
                                        if v0_type > 0:
                                            igood = ii
                                            break
                    loops = possible_loops[igood]
                    types = possible_types[igood]
                    if it>0: V_already_used.append( tuple(sorted( FoundConnection[igood] )) ) 

            loop_vertex.append(loops)
            loop_type.append(types)
            if debug: print 'Finally loop=', loops, 'types=', types
            StoreIfUnique(loops, Phi_loop[it])
            #phi_loop = [ i for i in (loops) if i not in (0,1) ]
            #Already_stored=False
            #for previous_phi_loop in Phi_loop[it]:
            #    if previous_phi_loop==phi_loop:
            #        Already_stored=True
            #        break
            #if not Already_stored:
            #    Phi_loop[it].append(phi_loop)

        if how_many_interaction_loops+1 != len(loop_vertex):
            Nmissing = how_many_interaction_loops+1-len(loop_vertex)
            print 'WARNING Number of interaction loops =', len(loop_vertex)-1, 'while it is required to have', how_many_interaction_loops, 'hence missing=', Nmissing
            print 'The diagram is', diag, 'with phi=', phi, 'and indx=', indx[id]
            print 'We have the following loops=', loop_vertex, 'and need to add extra loops now'
            
            possible_loops=[]
            possible_types=[]
            for il1_il2 in FermiLoopsConnected:
                conn = FermiLoopsConnected[il1_il2]
                print 'il1_il2=', il1_il2, 'conn=', conn
                print 'V_already_used=', V_already_used
                
                iused=[]
                for iu,used in enumerate(V_already_used):
                    if any([u in conn for u in used]):
                        iused.append(iu)
                if len(iused) < len(conn)-1:
                    print 'Yes, we need to add', len(iused), len(conn)-1
                    possible=set([])
                    for it in conn:
                        for jt in conn:
                            it_jt = tuple(sorted([it,jt]))   
                            if it!=jt and it_jt not in V_already_used:
                               possible.add( tuple(sorted([it,jt])) )
                    print 'possible=', possible
                    for it,jt in possible:
                        if debug: print '  While working with interaction ', diagsV[it], 'Found connections by interaction =', diagsV[jt]
                        v1,v2 = diagsV[it]
                        v3, v4 = diagsV[jt]
                        if (which_loop[v4]==which_loop[v2] and which_loop[v3]==which_loop[v1]):
                            v3,v4 = v4,v3
                        print 'We have v1,v2,v3,v4=', v1, v2, v3, v4
                            
                        # we need to consider exactly two fermionic loops
                        loops1, types1 = FindSimilarOrShortestPath( (v1,v2), v3, diag, i_diag, [], debug=debug, shorten=False)
                        loops2, types2 = FindSimilarOrShortestPath( (v3,v4), v1, diag, i_diag, [], debug=debug, shorten=False)
                        possible_loops.append( loops1+loops2 )
                        possible_types.append( types1+types2 )

                    print 'possible_loops=', possible_loops, 'corresponding types=', possible_types

            for imis in range(Nmissing):
                ## See if we can optimize for Baym-Kadanoff approach, by avoid to change k of the zeroth propagator.
                ## We would like zeroth propagator for G to be only part of loop1, and none else.
                igood = 0
                if len(possible_loops) > 1: # otherwise we have no choice anyway
                    # Just try to choose the shortest
                    lngths = array([len(loop) for loop in possible_loops])
                    igoods = where(lngths==lngths.min())[0] # which are among the shortest. Could be several
                    print 'lngths=', lngths, 'igoods=', igoods
                    if len(igoods)==1:
                        igood = igoods[0] # just take the shortest
                    else:
                        for ii in igoods:
                            loop = possible_loops[ii]
                            if 0 not in loop:
                                igood = ii
                                break
                loop_vertex.append( possible_loops.pop(igood) )
                loop_type.append( possible_types.pop(igood) )
                if debug:
                    print 'Finally loop=', loop_vertex[-1], 'types=', loop_type[-1]
                #StoreIfUnique(loops, Phi_loop[it])
            print 'all loops=', loop_vertex
            
        if how_many_interaction_loops+1 != len(loop_vertex):
            print 'ERROR: Number of interaction loops =', len(loop_vertex)-1, 'while it is required to have', how_many_interaction_loops
            sys.exit(1)
            
        # now we add pure fermionic loops, which do not include interaction lines.
        # each fermionic loop contributes exactly one momentum loop.
        N=len(diagsV)

        if debug:
            print 'Finished Interaction lines. Now follow fermionic loops'
            print 'len(Phi_loop)=', len(Phi_loop), 'i+N=', N

        for i,c in enumerate(cycles):
            #if i==0: continue # skip zero-th loop, because it was done above
            loop1=[]
            type1=[]
            i1 = i_diag[c[0]]
            for j in range(len(c)):
                loop1.append(i1)
                type1.append(1)
                i1 = diag[i1]
            loop2=[]
            type2=[]
            i1 = diag[c[0]]
            for j in range(len(c)):
                loop2.append(i1)
                type2.append(-1)
                i1 = i_diag[i1]
            if debug:
                print 'loop1=', loop1, 'loop2=', loop2, 'Phi=', Phi_loop[i+N]
            
            which = CompareSimilar(Phi_loop[i+N],loop1,loop2,type1,type2,debug=debug)
            
            if which is None:
                if len(loop1)<=len(loop2):
                    which=1
                else:
                    which=2
            if which==1:
                loop, type = loop1, type1
            else:
                loop, type = loop2, type2
            StoreIfUnique(loop, Phi_loop[i+N])
            if debug: print 'Phi_loop is now', Phi_loop[i+N], 'where loop=', loop
            
            loop_vertex.append(loop)
            loop_type.append(type)
            if debug: print 'loop=', loop, 'type=', type
        Loop_vertex.append( loop_vertex )
        Loop_type.append( loop_type )


        print 'Matrix='
        MTV = zeros((len(loop_vertex),2*Norder))
        for iloop in range(len(loop_vertex)):
            lv = loop_vertex[iloop]
            lt = loop_type[iloop]
            for i in range(len(lv)):
                if lt[i]==1:
                    MTV[iloop, lv[i]] += lt[i]
                elif lt[i]==-1:
                    MTV[iloop, lv[(i+1)%len(lv)] ] += lt[i]
        rank = numpy.linalg.matrix_rank(MTV)
        if rank != (Norder+1):
            print 'ERROR : The loops are not linearly independent for diagram', diag
            print 'Matrix=', MTV
            print 'rank=', rank, Norder+1
            sys.exit(0)
        
    # Finally we add Loop_index to existing Loop_vertex list
    # We need Vindex list for that..
    Norder = len(diagsG[0])/2
    Vindex=zeros(2*Norder,dtype=int)
    for i in range(len(diagsV)):
        Vindex[diagsV[i][0]] = i
        Vindex[diagsV[i][1]] = i
    print 'Vindex=', Vindex    
    
    Loop_index = []
    for id in range(Ndiags):
        loop_vertex = Loop_vertex[id]
        loop_type = Loop_type[id]
        loop_index=[]
        for iloop in range(len(loop_vertex)):
            lvertex = loop_vertex[iloop]
            ltype   = loop_type[iloop]
            lindex=[]
            for i in range(len(lvertex)):
                if ltype[i]==1   : ii = lvertex[i]
                elif ltype[i]==-1: ii = lvertex[(i+1)%len(lvertex)]
                else: ii = Vindex[abs(lvertex[i])]
                lindex.append(ii)
            loop_index.append(lindex)
        Loop_index.append( loop_index )
        
    return (Loop_vertex, Loop_type, Loop_index )


if __name__ == '__main__':
    #filename = 'Pdiags.3'
    #filename = raw_input("Enter filename : ")
    
    #diagsG = [(1, 7, 4, 5, 6, 8, 2, 9, 3, 0)]
    #diagsV = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    #indx = [0]
    #FindAllLoopsN(diagsG, diagsV, indx, debug=True)
    #sys.exit(0)



    
    if len(sys.argv)<2:
        print 'Give input filename'
        sys.exit(1)
    filename = sys.argv[1]

    diagsG, indx, Vtype, Factor = ReadPDiagrams(filename)
     
    diagsV = [[2*i,2*i+1] for i in range(len(diagsG[0])/2)]
    
    Norder = len(diagsG[0])/2
    Ndiags = len(diagsG)
    print '# Feynman diagrams=', Ndiags, 'interaction (diagsV)=', diagsV, 'Norder=', Norder
    print 'indx=', indx, 'Vtype=', Vtype, 'diagsV=', diagsV

    
    (Loop_vertex, Loop_type, Loop_index) = FindAllLoopsN(diagsG, diagsV, indx, debug=True)

    Vertices = FindAllVertices(diagsG, diagsV)
    diagSign = SignDiagram(diagsG)
    
    
    outfile = os.path.join(os.path.dirname(filename), 'loops_'+os.path.basename(filename)+'_')
    #outfile = 'loops_'+filename+'_'
    fu = open(outfile, 'w')
    print >> fu, '# ind N_loops [vertex|type|index]'
    for id in range(Ndiags):
        n = len(Loop_vertex[id])
        if (n!=Norder+1):
            print 'ERROR In diagram', id, ':', diagsG[id], ' the number of loops should be Norder+1, but it is not. Norder=' , Norder, 'and Nloops=', n
            sys.exit(1)
        print >> fu, indx[id], diagsG[id],
        if Vtype: print >> fu, ';', Vtype[id],
        print >> fu
        print >> fu, indx[id], n, Loop_vertex[id], ' # loop vertex'
        print >> fu, indx[id], n, Loop_type[id], ' # loop type'
        print >> fu, indx[id], n, Loop_index[id], ' # loop index'
        print >> fu, indx[id], n, Vertices[id], '# vertex type'
        if type(Factor[id])==tuple:
            print >> fu, indx[id], n, Factor[id], ' # Hugenholtz diagram signs'
        else:
            print >> fu, indx[id], n, diagSign[id]/float(Factor[id]), ' # diagram sign'
    fu.close()
    
    ### Printing loops
    print '************** Loop_vertex **************'
    for id in range(Ndiags):
        n = len(Loop_vertex[id])
        print id, n, 'vertex=', Loop_vertex[id]
        print id, n, 'type  =', Loop_type[id]
        print id, n, 'index =', Loop_index[id]


    sys.exit(0)
    ### The rest is not needed as is done in C++
    
    # Prime numbers are good for finding unique propagators.
    pm = FindPrimeNumbers(11,10000)
    
    # need some experimenting with these numbers to get unique combinations, so that 
    # we do not have accidentally equal propagators
    times = [pm[-1],0] + pm[5::9][:(2*Norder-2)]  # for each vertex we have one time
    kx = pm[2::5][:(Norder+1)]  # for each iloop we have one momenta
    ky = pm[0::4][:(Norder+1)]
    # two dimensional momenta, so that it is less probable to have an accidental cancelation
    momenta = array([[kx[i],ky[i]] for i in range(len(kx))])
    
    print 'momenta=', momenta
    
    G_mom = zeros((Ndiags, 2*Norder, 2),dtype=int)
    V_mom = zeros((Ndiags, Norder, 2), dtype=int)
    for id in range(Ndiags):
        #loop_vertex = Loop_vertex[id]
        loop_index  = Loop_index[id]
        loop_type   = Loop_type[id]
        for iloop in range(len(loop_index)):
            #lvertex = loop_vertex[iloop]
            lindex  = loop_index[iloop]
            ltype   = loop_type[iloop]
            for i in range(len(lindex)):
                #v1, v2 = lvertex[i], lvertex[(i+1)%len(lvertex)]
                if abs(ltype[i])==1 :
                    print 'id=', id, 'lindex[i]=', lindex[i], 'iloop=', iloop, 'ltype[i]=', ltype[i]
                    G_mom[id, lindex[i]] += momenta[iloop] * sign(ltype[i])
                    #print 'momenta=', momenta[iloop]
                    #print id, 'G('+str(v1)+'->'+str(v2)+') +='+str(ltype[i])+'*'+str(momenta[iloop]), 'iloop=', iloop
                else:
                    #print 'id=', id, 'v=', min(v1,v2)
                    V_mom[id,lindex[i]] += momenta[iloop] * sign(ltype[i])
            #print 'G_mom=', G_mom
    
    print 'Example:'
    print 'momenta=', momenta
    print 'times=', times
    
    whichGs={}
    whichVs={}
    uniqueG=[]
    uniqueV=[]
    Gunique = zeros((Ndiags, 2*Norder), dtype=int)
    Vunique = zeros((Ndiags,  Norder), dtype=int)
    ig=0
    iv=0
    for id in range(Ndiags):
        print 'diag ',id
        for i in range(2*Norder):
            K = G_mom[id,i]
            i_final = diagsG[id][i]
            t1, t2 = times[i_final],times[i]
            tk = (t1-t2,K[0],K[1])
            if whichGs.has_key(tk):
                iig = whichGs[tk]
                uniqueG[ iig ].append((id,i))
                Gunique[id,i] = iig
                #print '      using existing ig=', whichGs[tk]
            else:
                whichGs[tk] = ig
                uniqueG.append(  [(id,i)] )
                Gunique[id,i] = ig
                #print '      ig=', ig, 'uniqueG=', uniqueG
                ig += 1
            print i, 'G('+str(t1)+','+str(t2)+'|'+str(K)+')'
        for i in range(Norder):
            Q = V_mom[id,i]
            i0, i1 = diagsV[i]
            t1, t2 = times[i0], times[i1]
            tq = (abs(t1-t2),Q[0],Q[1])
            if whichVs.has_key(tq):
                iiv = whichVs[tq]
                uniqueV[ iiv ].append((id,i))
                Vunique[id,i] = iiv
            else:
                whichVs[tq] = iv
                uniqueV.append(  [(id,i)] )
                Vunique[id,i] = iv
                iv += 1
            print i, 'V('+str(t1)+','+str(t2)+'|'+str(Q)+')'
    
    print 'all possible Gs='
    gall=0
    for ik in range(len(uniqueG)):
        id, ii = uniqueG[ik][0]
        K = G_mom[id,ii]
        i_final = diagsG[id][ii]
        t1, t2 = times[i_final],times[ii]
        print "%3d %3d %-15s" % (ik, len(uniqueG[ik]), ':  G('+str(t1)+','+str(t2)+'|'+str(K)+')' ), uniqueG[ik]
        gall += len(uniqueG[ik])
    print ' with all Gs=', gall
    print 'all possible Vs='
    vall=0
    for ik in range(len(uniqueV)):
        id, ii = uniqueV[ik][0]
        Q = V_mom[id,ii]
        i0, i1 = diagsV[ii]
        t1, t2 = times[i0], times[i1]
        print "%3d %3d %-14s" % (ik, len(uniqueV[ik]), 'V('+str(t1)+','+str(t2)+'|'+str(Q)+')'), uniqueV[ik]
        vall += len(uniqueV[ik])
    print ' with all Vs=', vall
    
    outfile = os.path.join(os.path.dirname(filename), '_'+os.path.basename(filename)+'_')
    fo = open(outfile, 'w')
    print >> fo, '# ', len(uniqueG), '   ', len(uniqueV), '# all: ', gall, vall
    for id in range(Ndiags):
        print >> fo, "%4d   " % (id,),
        for i in range(2*Norder):
            print >> fo, "%4d " % (Gunique[id,i],),
        print >> fo, '   ',
        for i in range(Norder):
            print >> fo, "%3d " % (Vunique[id,i],),
        print >> fo
    fo.close()

