#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
import copy as cp
from scipy import *
import itertools
import time
#from scipy import weave
import weave
import sys
#from numba import jit
import equivalent as eq

t_connect=0
t_unique=0
imax = 2**30 - 1

#@jit(nopython=True, cache=True)
def Connected(Gprop, Vs):
    t0 = time.time()
    global t_connect
    #print 'Gs=', Gprop, 'Vs=', Vs
    vertices = set([0]) # connected vertices, which contain vertex 0
    visited = set([])
    current_v = 0
    for i in range(len(Gprop)):
        #print 'current_vertex=', current_v
        visited.add(current_v) # this vertex is now visited, and should not be visited again
        # add the other end of the G propagator
        vertices.add(Gprop[current_v]) # this is also not in the same set
        # find which vertex is connected through interaction by vertex i
        vc = filter(lambda x: current_v in x, Vs)
        if vc:  # some vertices might not be connected by interaction
            v = vc[0] # relevant interaction propagator
            v_other_end = v[1] if current_v==v[0] else v[0]   # the other end
            #print 'v_other_end=', v_other_end, 'and v=', v
            vertices.add(v_other_end)   # this is also now in the same set
        #print 'vertices=', vertices
        #print 'visited=', visited, len(visited)
        if len(vertices)==len(Gprop):
            #print 'all already included'
            break  # all are in single set, hence connected
        
        found_more = False
        for vx in vertices:
            if vx not in visited:
                current_v = vx # need to visit something new
                found_more = True
                break
        if not found_more:
            break
    t_connect += time.time()-t0
    return len(vertices)==len(Gprop)

#@jit(nopython=True, cache=True)
def ContainsBubble(Gprop):
    for i in range(2, len(Gprop)):
        if Gprop[Gprop[i]]==i:
            return True
    return False

#@jit(nopython=True, cache=True)
def ContainsHartree(Gprop):
    contains_Hartree = False
    for i in range(2,len(Gprop)):
        if Gprop[i]==i:
            contains_Hartree = True
            break
    return contains_Hartree

#@jit(nopython=True, cache=True)
def to_cycles(perm):
    #pi = {i+1: perm[i] for i in range(len(perm))}
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi)) # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break

        cycles.append(cycle)

    return cycles

#@jit(nopython=True, cache=True)
def SetTheLastInteractionLine(ii0, Vr, Vs, loop, which_loop, Ints0):
    # changes Vr[ii0]
    all_iis = Ints0[loop]
    qsum=0
    for jj in all_iis:
        if jj==ii0: continue  # this one is not jet set, and needs to be a sum
        if which_loop[Vs[jj][0]]==loop: # interaction starts on loop, hence outgoing
            qsum -= Vr[jj]
        else:                            # interaction outgoing
            qsum += Vr[jj]
    sign=1
    if which_loop[Vs[ii0][1]]==loop: # interaction is incoming
        sign=-1
    return qsum*sign

#@jit(nopython=True, cache=True)
def ContainsTadpoleOrPolarization(Gprop, Vs):
    #print 'Checking graph ', Gprop, Vs
    # Finding all fermionic loops
    cycles = to_cycles(Gprop)   # fermionic loops
    which_loop = zeros(len(Gprop), dtype=int) # this vertex is in which fermionic loop?
    for i,c in enumerate(cycles):
        for j in range(len(c)): which_loop[c[j]]=i
    #print 'loops=', cycles
    #print 'which_loop=', which_loop
    Ints = [set([]) for i in range(len(cycles))]  # which interaction lines stick out of this loop?
    for ii in range(len(Vs)):
        v0 = Vs[ii][0]  # vertices of this interaction line
        v1 = Vs[ii][1]  # second vertex of this interaction line
        loop0 = which_loop[v0]
        loop1 = which_loop[v1]
        #print 'for interaction ii=', ii, 'vertices are=', v0, v1, 'in loops=', which_loop[v0], which_loop[v1]
        if loop0 != loop1: # we want to save only interactions that connect different fermonic loops
            Ints[loop0].add(ii)
            Ints[loop1].add(ii)
    #print 'Ints=', Ints
    ## Finding possible tadpoles, which need special care of zero momenta
    for i,c in enumerate(cycles):
        if len(Ints[i])==1: # we have a tadpole, ii0 is the measuring line
            ii0 = Ints[i]
            #print 'We found a tadpole in ', c, 'with interaction line', Vs[ii0]
            return True
        if len(Ints[i])==2:# we have a polarization diagram
            ii2 = Ints[i]
            #print 'We found a polarization in ', c, 'with interaction line', [Vs[ii] for ii in ii2]
            return True
    return False

def ContainsTadpolePolarization(Gprop, Vs, which):
    if which=='Tadpole': 
        iwhich=1
    elif which=='Polarization': 
        iwhich=2
    else:
        print 'Give which= Tadpole | Polarization'
    
    #print 'Checking graph ', Gprop, Vs
    # Finding all fermionic loops
    cycles = to_cycles(Gprop)   # fermionic loops
    which_loop = zeros(len(Gprop), dtype=int) # this vertex is in which fermionic loop?
    for i,c in enumerate(cycles):
        for j in range(len(c)): which_loop[c[j]]=i
    #print 'loops=', cycles
    #print 'which_loop=', which_loop
    Ints = [set([]) for i in range(len(cycles))]  # which interaction lines stick out of this loop?
    for ii in range(len(Vs)):
        v0 = Vs[ii][0]  # vertices of this interaction line
        v1 = Vs[ii][1]  # second vertex of this interaction line
        loop0 = which_loop[v0]
        loop1 = which_loop[v1]
        #print 'for interaction ii=', ii, 'vertices are=', v0, v1, 'in loops=', which_loop[v0], which_loop[v1]
        if loop0 != loop1: # we want to save only interactions that connect different fermonic loops
            Ints[loop0].add(ii)
            Ints[loop1].add(ii)
    #print 'Ints=', Ints
    ## Finding possible tadpoles, which need special care of zero momenta
    
    res = [False,False]
    for i,c in enumerate(cycles):
        if len(Ints[i])==iwhich: # we have a tadpole, ii0 is the measuring line
            ii0 = Ints[i]
            #print 'We found a tadpole in ', c, 'with interaction line', Vs[ii0]
            return True
    return False


def ContainsTadpoleOrReducible(Gprop, Vs):
    
    def ConnectsWhichTwoLoops(iloop, Ints, Vs, which_loop):
        ii0,ii1 = Ints[iloop] # the two interaction lines
        vertices = Vs[ii0] + Vs[ii1] # the fours vertices
        #print 'vertices=', vertices
        # only loops which are not equal to iloop we want to return. This can be single loop or two loops.
        loops = [which_loop[v] for v in vertices if which_loop[v]!=iloop]
        return loops
    
    def AreTheseLoopsConnected(iloop0,iloop1,Ints2):
        #print 'Checking connectness of loop ',iloop0,' and ', iloop1
        
        if Ints2[iloop0] & Ints2[iloop1]:
            # There is a direct connection between the two loops, because they share a common interaction line
            #print 'Connected directly'
            return True
        ilps = range(len(Ints2))
        ilps.remove(iloop0)
        ilps.remove(iloop1)
        for iloop2 in ilps:
            # There is indirect connection between loop0 and loop1 through loop2
            if iloop2!=iloop1 and iloop2!=iloop0:
                if (Ints2[iloop0] & Ints2[iloop2]) and (Ints2[iloop2] & Ints2[iloop1]):
                    #print 'Connected through loop', iloop2
                    return True
            ilps2 = ilps[:]
            ilps2.remove(iloop2)
            for iloop3 in ilps2:
                # There might be indirect connection between loop0 and loop1 through loop2 and loop3
                if (Ints2[iloop0] & Ints2[iloop3]) and (Ints2[iloop3] & Ints2[iloop2]) and (Ints2[iloop2] & Ints[iloop1]):
                    return True
                ilps3 = ilps2[:]
                ilps3.remove(iloop3)
                for iloop4 in ilps3:
                    if (Ints2[iloop0] & Ints2[iloop4]) and (Ints2[iloop4] & Ints2[iloop3]) and (Ints2[iloop3] & Ints2[iloop2]) and (Ints2[iloop2] & Ints[iloop1]):
                        return True
                    
        # we would need to go down as deep as necessary...
        return False
    
        
    #print 'Checking graph ', Gprop, Vs
    # Finding all fermionic loops
    cycles = to_cycles(Gprop)   # fermionic loops
    which_loop = zeros(len(Gprop), dtype=int) # this vertex is in which fermionic loop?
    for i,c in enumerate(cycles):
        for j in range(len(c)): which_loop[c[j]]=i
    #print 'loops=', cycles
    #print 'which_loop=', which_loop
    Ints = [set([]) for i in range(len(cycles))]  # which interaction lines stick out of this loop?
    for ii in range(len(Vs)):
        v0 = Vs[ii][0]  # vertices of this interaction line
        v1 = Vs[ii][1]  # second vertex of this interaction line
        loop0 = which_loop[v0]
        loop1 = which_loop[v1]
        #print 'for interaction ii=', ii, 'vertices are=', v0, v1, 'in loops=', which_loop[v0], which_loop[v1]
        if loop0 != loop1: # we want to save only interactions that connect different fermonic loops
            Ints[loop0].add(ii)
            Ints[loop1].add(ii)
    #print 'Ints=', Ints
    ## Finding possible tadpoles, which need special care of zero momenta
    for i,c in enumerate(cycles):
        if len(Ints[i])==1: # we have a tadpole, ii0 is the measuring line
            ii0 = Ints[i]
            #print 'We found a tadpole in ', c, 'with interaction line', Vs[ii0]
            return True
        if len(Ints[i])==2:# we have a polarization diagram, maybe reducible
            if 0 in Ints[i]: # meassuring line connects this loop, which should be cut
                # only one connection left
                #ii0,ii1 = Ints[i]
                #vertices = Vs[ii0] + Vs[ii1] # the fours vertices
                # You should check that... Not sure
                return True
            
            loops = ConnectsWhichTwoLoops(i, Ints, Vs, which_loop)
            #print 'Loop',i,'and interactions', Ints[i], 'connect two loops=', loops
            if loops[0] == loops[1]: continue # connects the same loop, hence it is irreducible polarization diagram
            
            # We know that this loop connects two different loops, rather than grows out of a single loop
            # Hence it is most likely reducible. However, to be reducible, these other two loops should be connected by only
            # the current interaction line. To see if this is the case, we eliminate from Ints the current loop and its connection
            # as well as the meassuring line. We then check if the two loops are still connected with some alternative path
            Ints2 = cp.deepcopy(Ints) # copy
            # Now, removing the meassuring line, which has interaction ii=0, and current loop, which we just considered
            ii0,ii1 = Ints[i]
            for s in Ints2:
                if 0 in s:
                    s.remove(0)
                if ii0 in s:  # one interaction from the current loop
                    s.remove(ii0)
                if ii1 in s:  # the second interaction from the current loop
                    s.remove(ii1)
            #print 'Ints2=', Ints2
            connected = AreTheseLoopsConnected(loops[0],loops[1],Ints2)
            if not connected: # Hence it is reducible
                return True
    return False


#@jit(nopython=True, cache=True)
def Irreducible(Gprop, Vs):
    #print 'Checking graph ', Gprop, Vs
    global imax
    
    Vr = imax*ones(len(Vs), dtype=int) # imax means not yet set, can not be 0, because 0 is possible in tadpoles
    could_go = set([])

    # Finding all fermionic loops
    cycles = to_cycles(Gprop)   # fermionic loops
    which_loop = zeros(len(Gprop), dtype=int) # this vertex is in which fermionic loop?
    for i,c in enumerate(cycles):
        for j in range(len(c)): which_loop[c[j]]=i
    #print 'loops=', cycles
    #print 'which_loop=', which_loop
    Ints = [set([]) for i in range(len(cycles))]  # which interaction lines stick out of this loop?
    for ii in range(len(Vs)):
        v0 = Vs[ii][0]  # vertices of this interaction line
        v1 = Vs[ii][1]  # second vertex of this interaction line
        loop0 = which_loop[v0]
        loop1 = which_loop[v1]
        #print 'for interaction ii=', ii, 'vertices are=', v0, v1, 'in loops=', which_loop[v0], which_loop[v1]
        if loop0 != loop1: # we want to save only interactions that connect different fermonic loops
            Ints[loop0].add(ii)
            Ints[loop1].add(ii)
    Ints0 = cp.deepcopy(Ints)
    #print 'Ints=', Ints
    ## Finding possible tadpoles, which need special care of zero momenta
    for i,c in enumerate(cycles):
        if len(Ints[i])==1: # we have a tadpole
            ii0 = Ints[i].pop()
            #print 'We found a tadpole in ', c, 'with interaction line', Vs[ii0]
            Vr[ii0]=0
            could_go.update(Vs[ii0][:])
    
    Ginv = [0]*len(Gprop)
    Vind = [0]*len(Gprop)
    for i in range(len(Gprop)): Ginv[Gprop[i]]=i  # Ginv shows from which vertex the e-propagator is coming from
    for i in range(len(Vs)): # Vind knows which interacting line is this
        Vind[Vs[i][0]]=i     # If V=[(0,1),(2,3),(4,5)]
        Vind[Vs[i][1]]=i     # then vertices V[0]=V[1]=0 correspond to 0-th interaction line
                             #      vertices V[2]=V[3]=1 correspond to 1-th interaction line....
    visited = set([])
    Gr = zeros(len(Gprop), dtype=int)  # means not yet set
    Gr[0] = 1
    c_v=Gprop[0]           # Start vertex=1
    could_go.add(c_v)
    follow_g = True # start by following electron propagator
    for ic in range(len(Gprop)): # exactly n-vertices need to be visited
        visited.add(c_v) # this vertex is now visited, and should not be visited again
        
        g_out = Gprop[c_v] # electron goes into this vertex
        g_in  = Ginv[c_v]  # electron comes from this vertex
        ii = Vind[c_v]     # interaction ii

        if c_v == Vs[ii][0]: # interaction goes out
            v_other = Vs[ii][1] # the other vertex connected by interaction

            #print 'At', c_v, 'g_in=', g_in, 'g_out=', g_out, 'ii=', ii, 'v_other=', v_other, 'follow_g=', follow_g, 'visited=', visited
        
            if follow_g: # we know that Gr[g_in] is set, and we just need to take care of the other two
                if Gr[c_v]==0:
                    # both unset
                    if Vr[ii]==imax:
                        Vr[ii] = random.randint(2, imax-1) # set the interaction line at random
                        could_go.add(v_other)
                        loop1,loop2= which_loop[c_v], which_loop[v_other]
                        if loop1!=loop2:  # one of connecting lines
                            #print 'Removing ', ii, 'from loops=', loop1, loop2
                            Ints[loop1].remove(ii)   # update of how many interaction line has been set 
                            Ints[loop2].remove(ii)   # for each loop. Once only one is left, we need to conserve momenta
                            if len(Ints[loop1])==1:     # only one interaction left to set on this loop. Then it must be the sum of all other momenta
                                ii0 = Ints[loop1].pop() # due to consevration law.
                                #print 'Only one line left, hence ii=', ii0, 'needs to be set to conserve momentum'
                                Vr[ii0] = SetTheLastInteractionLine(ii0, Vr, Vs, loop1, which_loop, Ints0)
                                could_go.add(Vs[ii0][0])
                                could_go.add(Vs[ii0][1])
                            # Such line always connects two different fermionic loops, hence we need to check if this is
                            # also the last on the other loop
                            if len(Ints[loop2])==1:
                                ii0 = Ints[loop2].pop()
                                #print 'Only one line left, hence ii=', ii0, 'needs to be set to conserve momentum'
                                Vr[ii0] = SetTheLastInteractionLine(ii0, Vr, Vs, loop2, which_loop, Ints0)
                                could_go.add(Vs[ii0][0])
                                could_go.add(Vs[ii0][1])
                    # now conserve momenta on outgoing propagator
                    Gr[c_v] = Gr[g_in] - Vr[ii]
                    #could_go.add(Gprop[c_v])
                elif Vr[ii]==imax:
                    Vr[ii] = Gr[g_in] - Gr[c_v]
                    could_go.add(v_other)
                else:
                    if (Gr[c_v] + Vr[ii] != Gr[g_in]): print 'ERROR, no conservation of momenta!'
            else: # follow interaction, hence Vr[ii] is set
                if Gr[c_v]==0:
                    if Gr[g_in]==0:
                        Gr[g_in] = random.randint(2, imax-1)
                        #could_go.add(g_in)
                    Gr[c_v] = Gr[g_in] - Vr[ii]
                    #could_go.add(Gprop[c_v])                    
                elif Gr[g_in]==0:
                    Gr[g_in] = Gr[c_v] + Vr[ii]
                    #could_go.add(g_in)
                else:
                    if (Gr[c_v] + Vr[ii] != Gr[g_in]): print 'ERROR, no conservation of momenta!'

        else:               # interaction comes in
            v_other = Vs[ii][0]  # the other vertex connected by interaction
            
            #print 'At', c_v, 'g_in=', g_in, 'g_out=', g_out, 'ii=', ii, 'v_other=', v_other, 'follow_g=', follow_g, 'visited=', visited
            
            if follow_g: # we know that Gr[g_in] is set, and we just need to take care of the other two
                if Gr[c_v]==0:
                    # both unset
                    if Vr[ii]==imax:
                        Vr[ii] = random.randint(2, imax-1)
                        could_go.add(v_other)
                        loop1,loop2= which_loop[c_v], which_loop[v_other]
                        if loop1!=loop2: # one of connecting lines
                            #print 'Removing ', ii, 'from loops=', loop1, loop2
                            Ints[loop1].remove(ii) # update of how many interaction line has been set 
                            Ints[loop2].remove(ii) # for each loop. Once only one is left, we need to conserve momenta
                            #print 'So that Ints=', Ints
                            if len(Ints[loop1])==1:     # only one interaction left to set on this loop. Then it must be the sum of all other momenta
                                ii0 = Ints[loop1].pop() # due to consevration law.
                                Vr[ii0] = SetTheLastInteractionLine(ii0, Vr, Vs, loop1, which_loop, Ints0)
                                could_go.add(Vs[ii0][0])
                                could_go.add(Vs[ii0][1])
                            # Such line always connects two different fermionic loops, hence we need to check if this is
                            # also the last on the other loop
                            if len(Ints[loop2])==1:
                                ii0 = Ints[loop2].pop()
                                Vr[ii0] = SetTheLastInteractionLine(ii0, Vr, Vs, loop2, which_loop, Ints0)
                                could_go.add(Vs[ii0][0])
                                could_go.add(Vs[ii0][1])
                    # now conserve momenta on outgoing propagator
                    Gr[c_v] = Gr[g_in] + Vr[ii]
                    #could_go.add(Gprop[c_v])
                elif Vr[ii]==imax:
                    Vr[ii] = -Gr[g_in] + Gr[c_v]
                    could_go.add(v_other)
                else:
                    if (Gr[c_v] - Vr[ii] != Gr[g_in]): print 'ERROR, no conservation of momenta!'
            else: # follow interaction, hence Vr[ii] is set
                if Gr[c_v]==0:
                    if Gr[g_in]==0:
                        Gr[g_in] = random.randint(2, imax-1)
                        #could_go.add(g_in)
                    Gr[c_v] = Gr[g_in] + Vr[ii]
                    #could_go.add(Gprop[c_v])
                elif Gr[g_in]==0:
                    Gr[g_in] = Gr[c_v] - Vr[ii]
                    #could_go.add(g_in)
                else:
                    if (Gr[c_v] - Vr[ii] != Gr[g_in]): print 'ERROR, no conservation of momenta!'

        if g_out not in visited:
            c_v = g_out # following electron propagator
            follow_g = True
        elif v_other not in visited:
            c_v = v_other    # following interaction
            follow_g = False
        else:
            for i in could_go:
                if i not in visited:
                    c_v = i
                    follow_g=False
                    break
        #print 'Gr=', Gr, 'Vr=', Vr, 'next is', c_v, 'follow_g=', follow_g, 'could_go=', could_go, 'Ints=', Ints
        
    if 0 in Gr:
        print 'ERROR: It seems a propagator was not set Gr=', Gr, 'Vr=', Vr, 'with Gprop=', Gprop, 'and Vs=', Vs
    if imax in Vr:
        print 'ERROR: It seems the interaction was not set Gr=', Gr, 'Vr=', Vr, 'with Gprop=', Gprop, 'and Vs=', Vs
        
    return 1 not in Gr[1:] # the meassuring line is 1, hence the rest should not be.
    
            

#@jit(nopython=True, cache=True)
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

#@jit(nopython=True, cache=True)
def Graphs(n):
    global t_unique
    
    # The two types of interactions are
    V0 = [(2*i,2*i+1) for i in range(n)]
    V1 = [(0,2),(1,3)]+[(2*i,2*i+1) for i in range(2,n)]

    V0perm = GiveInteractionPermutations_new(V0[1:])  # all equivalent forms of interactions
    if n>2:
        V1perm = GiveInteractionPermutations_new(V1[2:])
    else:
        V1perm=[]

    pm = itertools.permutations([0]+range(2,2*n))
    Gall0 = []  # all diagrams which use V0
    Gall1 = []  # all diagrams which use V1
    for ip,p in enumerate(pm):
        Gprop = (1,) + p[:]
        if not ContainsHartree(Gprop):
            if not ContainsBubble(Gprop):
                if Connected(Gprop, V0):
                    if not ContainsTadpoleOrPolarization(Gprop, V0):
                        if Irreducible(Gprop, V0):
                            Gall0.append( Gprop )
                if Connected(Gprop, V1):
                    if not ContainsTadpoleOrPolarization(Gprop, V1):
                        if Irreducible(Gprop, V1):
                            Gall1.append( Gprop )
        if (ip+1)%10000 == 0:
            print '#', (ip+1), 'currently we have ', len(Gall0)+len(Gall1), 'diags'

    print 'Before checking for unique: all=', len(Gall0)+len(Gall1), 'len(Gall0)=', len(Gall0), 'len(Gall1)=', len(Gall1)

    t1 = time.time()
    ip=0
    while ip < len(Gall0)-1:
        Gp = Gall0[ip]   # Is this diagram unique?
        Gn = list(Gp[:]) # convert from tuple to list, so that we can change
        for i,p in enumerate(V0perm): # We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
            for j in range(len(p)):   # If we exchange vertices according to permutation V0perm, the electron propagators has to change 
                Gn[p[j]] = Gp[j+2]    # in the following way:
            Gm=Gn[:]
            for j in range(len(Gn)):
                if Gn[j]>=2: Gm[j] = p[Gn[j]-2]
            _Gn_ = tuple(Gm)          # finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else
            #print 'Gp=', Gp, 'p=', p, 'Gn=', _Gn_, 'interm=', Gn
            for iq in range(ip+1,len(Gall0)):  # over all other diagrams
                if _Gn_==Gall0[iq]:    # is this permuted diagram somewhere in the remaining of the list?
                    del Gall0[iq]      # yes, hence we should remove it
                    break
                
        if (ip+1)%10 == 0:
            print 'At', ip, '/', len(Gall0), 'we still have ', len(Gall0), 'diags'
        ip += 1

    ip=0
    while ip < len(Gall1)-1: # Now we do the same for diagrams with interaction V1
        Gp = Gall1[ip]   # Is this diagram unique?
        Gn = list(Gp[:]) # convert from tuple to list, so that we can change
        for i,p in enumerate(V1perm): # We go over all possible permutations of V1 interaction, and generate all equivalent diagrams
            for j in range(len(p)):   # If we exchange vertices according to permutation V0perm, the electron propagators has to change 
                Gn[p[j]] = Gp[j+4]    # in the following way:
            Gm=Gn[:]
            for j in range(len(Gn)):
                if Gn[j]>=4: Gm[j] = p[Gn[j]-4]
            _Gn_ = tuple(Gm)          # finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else
            #print 'Gp=', Gp, 'p=', p, 'Gn=', _Gn_, 'interm=', Gn
            for iq in range(ip+1,len(Gall1)):  # over all other diagrams
                if _Gn_==Gall1[iq]:   # is this permuted diagram somewhere in the remaining of the list?
                    del Gall1[iq]     # yes, hence we should remove it
                    break

        if (ip+1)%10 == 0:
            print 'At', ip, '/', len(Gall1), 'we still have ', len(Gall1), 'diags'
        ip += 1
    t_unique = time.time()-t1
        
    print 'Finally: all=', len(Gall0)+len(Gall1), 'len(Gall0)=', len(Gall0), 'len(Gall1)=', len(Gall1)
    diags=[]
    for Gp in Gall0:
        diags.append( [Gp, V0] )
    for Gp in Gall1:
        diags.append( [Gp, V1] )
    return diags


#@jit(nopython=True, cache=True)
def ContainsHartreeP(Gprop):
    contains_Hartree = False
    for i in range(len(Gprop)):
        if Gprop[i]==i:
            contains_Hartree = True
            break
    return contains_Hartree

def FindAllEquivalent(Gp,istart,Gall,V0perm):
        Gn = list(Gp[:]) # convert from tuple to list, so that we can change
        which=[]
        for i,p in enumerate(V0perm): # We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
            for j in range(len(p)):   # If we exchange vertices according to permutation V0perm, the electron propagators has to change 
                Gn[p[j]] = Gp[j+2]    # in the following way:
            Gm=Gn[:]
            
            for j in range(len(Gn)):  # because (0,1) interaction line is the measuring line
                if Gn[j]>=2: Gm[j] = p[Gn[j]-2]
            
            _Gn_ = tuple(Gm)          # finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else
            #print 'Gp=', Gp, 'p=', p, 'Gn=', _Gn_, 'interm=', Gn
            for iq in range(istart,len(Gall)):  # over all other diagrams
                if _Gn_==Gall[iq]:    # is this permuted diagram somewhere in the remaining of the list?
                    which.append(iq)  # yes, hence we should remove it
                    break
        return which
    
def CheckUniqueP3(Gall,V0perm):
    ip=0
    while ip < len(Gall)-1:
        Gp = Gall[ip]   # Is this diagram unique?
        #which = FindAllEquivalent(Gp, ip+1, Gall, V0perm)  # finding if the diagram has an equivalent in the rest of the list. Returns index to it.
        which = eq.FindAllEquivalent(Gp, ip+1, Gall, V0perm)  # finding if the diagram has an equivalent in the rest of the list. Returns index to it.
        #print 'which=', which, 'iwhich=', iwhich
        
        # actually removing the group
        which = sorted( which, reverse=True )
        for i in which: # you have to start deleing from the back
            del Gall[i]
            
        if (ip+1)%10 == 0:
            print('At {0:3d} / {1:4d} we still have {2:4d} diags'.format(ip, len(Gall), len(Gall)))
            #print 'At', ip, '/', len(Gall), 'we still have ', len(Gall), 'diags'
        ip += 1
    return Gall



def CheckUniqueP(Gall,V0perm):
    ip=0
    while ip < len(Gall)-1:
        Gp = Gall[ip]   # Is this diagram unique?
        Gn = list(Gp[:]) # convert from tuple to list, so that we can change
        for i,p in enumerate(V0perm): # We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
            for j in range(len(p)):   # If we exchange vertices according to permutation V0perm, the electron propagators has to change 
                Gn[p[j]] = Gp[j+2]    # in the following way:
            Gm=Gn[:]
            
            for j in range(len(Gn)):  # because (0,1) interaction line is the measuring line
                if Gn[j]>=2: Gm[j] = p[Gn[j]-2]
            
            _Gn_ = tuple(Gm)          # finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else
            #print 'Gp=', Gp, 'p=', p, 'Gn=', _Gn_, 'interm=', Gn
            for iq in range(ip+1,len(Gall)):  # over all other diagrams
                if _Gn_==Gall[iq]:    # is this permuted diagram somewhere in the remaining of the list?
                    del Gall[iq]      # yes, hence we should remove it
                    break
                
        if (ip+1)%10 == 0:
            print('At {0:3d} / {1:4d} we still have {2:4d} diags'.format(ip, len(Gall), len(Gall)))
            #print 'At', ip, '/', len(Gall), 'we still have ', len(Gall), 'diags'
        ip += 1
    return Gall
#@jit(nopython=True, cache=True)
def CheckUniqueP2(Gall,V0perm):
    ip=0
    while ip < len(Gall)-1:
        Gp = Gall[ip]   # Is this diagram unique?
        Gn = copy(Gp[:]) # convert from tuple to list, so that we can change
        for i,p in enumerate(V0perm): # We go over all possible permutations of V0 interaction, and generate all equivalent diagrams
            for j in range(len(p)):   # If we exchange vertices according to permutation V0perm, the electron propagators has to change 
                Gn[p[j]] = Gp[j+2]    # in the following way:
            Gm= copy(Gn[:])
            for j in range(len(Gn)):  # because (0,1) interaction line is the measuring line
                if Gn[j]>=2: Gm[j] = p[Gn[j]-2]
            
            _Gn_ = Gm          # finally we have the correspondingly permuted Gp, and hence we will check if such diagram is hiding somewhere else
            #print 'Gp=', Gp, 'p=', p, 'Gn=', _Gn_, 'interm=', Gn
            for iq in range(ip+1,len(Gall)):  # over all other diagrams
                #print 'comparing', _Gn_, Gall[iq], all(_Gn_==Gall[iq])
                if all(_Gn_==Gall[iq]):    # is this permuted diagram somewhere in the remaining of the list?
                    del Gall[iq]      # yes, hence we should remove it
                    break
                
        if (ip+1)%10 == 0:
            print('At {0:3d} / {1:4d} we still have {2:4d} diags'.format(ip, len(Gall), len(Gall)))
            #print 'At', ip, '/', len(Gall), 'we still have ', len(Gall), 'diags'
        ip += 1
    return Gall

def ContainsFock(Gprop, V0):
    for iq in range(1,len(V0)):
        if Gprop[V0[iq][0]]==V0[iq][1] or Gprop[V0[iq][1]]==V0[iq][0]:
            return True
    return False

#@jit(nopython=True, cache=True)
def PGraph(np, Static=True, removeFock=True):
    def GoodPolarizationGraphStatic(Gprop):
        if not ContainsHartreeP(Gprop):
            if Connected(Gprop, V0[1:]): # V0[1:] because the meassuring line is not present
                #if not ContainsTadpoleOrPolarization(Gprop, V0):
                if not ContainsTadpoleOrReducible(Gprop, V0):
                    if not( removeFock and ContainsFock(Gprop,V0) ):
                        print '#', (ip+1), Gprop
                        return True
        else:
            return False

    def GoodPolarizationGraphDynamic(Gprop):
        if not ContainsHartreeP(Gprop):
            if Connected(Gprop, V0[1:]):  # V0[1:] because the meassuring line is not present
                if not ContainsTadpoleOrPolarization(Gprop, V0):
                    print '#', (ip+1), Gprop
                    return True
        else:
            return False
    global t_unique

    if (np==1):
        return [(1,0)]
    
    # The two types of interactions are
    V0 = [(2*i,2*i+1) for i in range(np)]
    V0perm = GiveInteractionPermutations_new(V0[1:])  # all equivalent forms of interactions
    
    t1 = range(2*np)
    t1.remove(1)
    pm1 = itertools.permutations( t1 )
    t2 = range(2*np)
    t2.remove(2)
    pm2 = itertools.permutations( t2 )
    Gall = []
    if Static:
        for ip,p in enumerate(pm1):
            Gprop = (1,)+p[:]
            if GoodPolarizationGraphStatic(Gprop):
                Gall.append(Gprop)
        for ip,p in enumerate(pm2):
            Gprop = (2,)+p[:]
            if GoodPolarizationGraphStatic(Gprop):
                Gall.append(Gprop)
    else:
        for ip,p in enumerate(pm1):
            Gprop = (1,)+p[:]
            if GoodPolarizationGraphDynamic(Gprop):
                Gall.append(Gprop)
        for ip,p in enumerate(pm2):
            Gprop = (2,)+p[:]
            if GoodPolarizationGraphDynamic(Gprop):
                Gall.append(Gprop)
        
    print 'Before checking for unique: all=', len(Gall)

    #_Gall_ = [array(p) for p in Gall]
    t1 = time.time()
    #Gall = CheckUniqueP2(_Gall_,V0perm)
    #Gall = CheckUniqueP(Gall,V0perm)
    Gall = CheckUniqueP3(Gall,V0perm)
    t_unique = time.time()-t1
        
    print 'Finally: all=', len(Gall)
    return Gall


    

if __name__ == '__main__':

    order = raw_input("Enter integer for perturbation order : ")
    n = int(order)
    
    #### Polarization diagrams
    diagP = PGraph(n, Static=True, removeFock=True)
    fi = open('Pdiags.'+str(n),'w')
    print 'Pdiags='
    for i, d in enumerate(diagP):
        print i, d
        print >> fi, i, d
    print '# diagrams=', len(diagP)

    sys.exit(0)
    
    #### Self energy diagrams
    diags = Graphs(n)
    
    fi = open('diags.'+str(n),'w')
    print 'diags='
    for i, d in enumerate(diags):
        print i, d
        print >> fi, i, d
    print '# diagrams=', len(diags)
    
    print 't_connect=', t_connect, 't_unique=', t_unique
    
