# @Copyright 2018 Kristjan Haule 
from scipy import *
import re

def to_cycles(perm):
    "breaks permutation into cycles, which represent fermionic loops"
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

    if 0 not in cycles[0]:
        print 'ERROR : You should correct function to_cycles so that 0 is in the first cycles'
        exit(1)
    return cycles

def FindShortestPath(V_vertices, v_final, diag, i_diag):
    """Given two interaction vertices V_vertices[0..1] and another vertex v_final in the same fermionic loop as V_vertices[1]
       it gives the closest path from V_vertices[1] to v_final. We could follow the fermionic propagators in either positive or
       negative direction.
       For positive direction of fermionic propagator type=+1, for negative direction of fermionic propagator type=-1
       for positive direction of bosonic (interaction) propagator type=+2
       loop will contain the list of vertices visited
       type will show if the connection between successive vertices is fermionic (+1 or -1) of bosonic (+2 or -2).
    """
    # interaction V(v1,v2)
    #print 'V_vertices=', V_vertices, 'v_final=', v_final
    loop1 = [V_vertices[0]]
    type1 = [2]
    i1 = V_vertices[1]
    while i1!=v_final:
        loop1.append(i1)
        type1.append(1)
        i1 = diag[i1]
    loop2 = [V_vertices[0]]
    type2 = [2]
    i1 = V_vertices[1]
    while i1!=v_final:
        loop2.append(i1)
        type2.append(-1)
        i1 = i_diag[i1]
    if len(loop1)<=len(loop2):
        return loop1, type1
    else:
        return loop2, type2

def FindAllLoops(diagsG, diagsV, debug):
    """ It goes over all diagrams, and finds all necessary loops for each diagram.
    Each fermionic loop contributes one loop. In addition, each interaction line contributes one loop (except in some very high symmetry cases).
    It can connect up to four fermionic loops for single momentum loop, but not yet more than four. At very high order one needs to increase that.
    """
    Ndiags = len(diagsG)
    Loop_vertex=[]
    Loop_type=[]
    for id in range(Ndiags):
        diag = diagsG[id]
        if debug: print 'Analizing diagram', diag
        # The inverse of the perturbation, which corresponds to following the fermionic propagators in opposite direction than arrows
        i_diag = zeros(len(diag),dtype=int)
        for i in range(len(diag)): i_diag[diag[i]]=i
        # Cycles give fermionic loops.
        cycles = to_cycles(diag)
        # For each vertex we want to know in each fermionic loop is located
        which_loop = zeros(len(diag), dtype=int) # this vertex is in which fermionic loop?
        for i,c in enumerate(cycles):
            for j in range(len(c)): which_loop[c[j]]=i
    
        if debug:
            print 'cycles=', cycles
            print 'which_loop=', which_loop
    
        loop_vertex=[]
        loop_type=[]
        skipV=[]
        # first create mixed loops, which contain at least one interaction line, and several fermionic lines
        for it in range(len(diagsV)):
            if it in skipV: continue # this interaction was already used in another loop, and does not need to be considered again
            v1,v2 = diagsV[it]
            if which_loop[v1]==which_loop[v2]:
                # interaction is within the same fermionic loop, hence we just follow fermionic propagators to close the loop
                loops, types = FindShortestPath( (v1,v2) , v1, diag, i_diag)
            else:
                # at least two fermionic loops are connected by this momentum
                il1, il2 = which_loop[v1], which_loop[v2]
                FoundConnection = 0
                for jt in range(1,len(diagsV)):
                    if it==jt: continue
                    v3, v4 = diagsV[jt]
                    if (which_loop[v3]==il2 and which_loop[v4]==il1):
                        FoundConnection = jt  # diagsV[jt] connects the same fermionic loops as diagsV[it]
                        skipV.append(jt)
                        break
                    elif (which_loop[v4]==il2 and which_loop[v3]==il1):
                        FoundConnection = jt  # diagsV[jt] connects the same fermionic loops as diagsV[it]
                        skipV.append(jt)
                        v3, v4 = v4, v3
                        break
                if FoundConnection:
                    # we need to consider exactly two fermionic loops
                    loops = []; types = []
                    loop1, type1 = FindShortestPath( (v1,v2) , v3, diag, i_diag)
                    loops += loop1; types += type1
                    loop1, type1 = FindShortestPath( (v3,v4) , v1, diag, i_diag)
                    loops += loop1; types += type1
                else:
                    if debug: print 'Did not yet found connection between v1=', v1, 'v2=', v2, 'for interaction', it
                    # more than two fermionic loops need to be considered. Need to find a path from loop1 to loopn through loop2, loop3,...etc
                    l1 = cycles[il1][:] # all vertices in the fermionic loop containing vertex v1
                    l2 = cycles[il2][:] # all vertices in the fermionic loop containing vertex v2
                    l1.remove(v1) # looking for path between the two fermionic loops, which is not through the current interaction diagsV[it] nor the meassuring line
                    l2.remove(v2) # now remove v1 from l1 (and v2 from l2) because we want to find connection between l1 and l2 which is not diagsV[it]
                    # Now finding interaction lines which connect pairs of fermionic loops
                    loop_connects={} # loop_connects[(fermionic_line_1,fermionic_line_2)] = index_to_interaction
                    for jt in range(1,len(diagsV)):
                        v3, v4 = diagsV[jt]
                        il3, il4 = which_loop[v3], which_loop[v4]
                        if il3 != il4:
                            if not loop_connects.has_key((il3,il4)):
                                loop_connects[(il3,il4)] = [jt]
                            else:
                                loop_connects[(il3,il4)].append(jt)
                            if not loop_connects.has_key((il4,il3)):
                                loop_connects[(il4,il3)] = [jt]
                            else:
                                loop_connects[(il4,il3)].append(jt)
                    if debug:
                        print 'loop_connects=', loop_connects
                        print 'We need connection between the loops', il2, il1
                        
                    for iln in range(len(diagsV)):
                        if iln==il1 or iln==il2: continue
                        if loop_connects.has_key( (il2,iln) ) and loop_connects.has_key( (iln,il1) ):
                            jt_all = loop_connects[(il2,iln)] # contains all possible interactions that connect these two loops
                            kt_all = loop_connects[(iln,il1)] # contains all possible interactions that connect these two loops
    
                            if debug: print 'jt_all=', jt_all, 'kt_all=', kt_all
                            
                            for ii in range(len(jt_all)): # now we take the first one, which is not it
                                jt = jt_all[ii]
                                if jt!=it: break
                            for ii in range(len(kt_all)): # also take the first, which is not it
                                kt = kt_all[ii]
                                if kt!=it: break
                            FoundConnection = [jt,kt,it]  # diagsV[jt] & diagsV[kt] connect the same fermionic loops as diagsV[it]
                            break
                    if not FoundConnection:
    
                        for iln in range(1,len(diagsV)):
                            for ilm in range(1,len(diagsV)):
                                if iln==il1 or iln==il2 or ilm==il1 or ilm==il2: continue
                                if loop_connects.has_key( (il2,iln) ) and loop_connects.has_key( (iln,ilm) ) and loop_connects.has_key( (ilm,il1) ) :
                                    jt_all = loop_connects[(il2,iln)] # contains all possible interactions that connect these two loops
                                    kt_all = loop_connects[(iln,ilm)] # contains all possible interactions that connect these two loops
                                    lt_all = loop_connects[(ilm,il1)]
                                    for ii in range(len(jt_all)): # now we take the first one, which is not it
                                        jt = jt_all[ii]
                                        if jt!=it: break
                                    for ii in range(len(kt_all)): # also take the first, which is not it
                                        kt = kt_all[ii]
                                        if kt!=it: break
                                    for ii in range(len(lt_all)): # also take the first, which is not it
                                        lt = lt_all[ii]
                                        if lt!=it: break
                                    FoundConnection = [jt,kt,lt,it]  # diagsV[jt] & diagsV[kt] connect the same fermionic loops as diagsV[it]
                                    break
                    if debug: print 'FoundConnection=', FoundConnection
                    # Now we create the path through interactions V[it],V[jt],V[kt],V[lt], and back to V[it]
                    loops=[]
                    types=[]
                    v1,v2 = diagsV[it]
                    for jt in FoundConnection:
                        iloop_start = which_loop[v2]
                        if which_loop[diagsV[jt][0]]==iloop_start:
                            v3 = diagsV[jt][0]
                            v4 = diagsV[jt][1]
                        else:
                            v3 = diagsV[jt][1]
                            v4 = diagsV[jt][0]
                        loop1, type1 = FindShortestPath( (v1,v2) , v3, diag, i_diag)
                        if debug: 
                            print 'v1,v2=', v1, v2, 'v3,v4=', v3,v4
                            print 'loop1=', loop1, 'type1=', type1
                        loops += loop1
                        types += type1
                        v1,v2 = v3,v4
                        
            loop_vertex.append(loops)
            loop_type.append(types)
            if debug: print 'loop=', loops, 'types=', types
        # now we add pure fermionic loops, which do not include interaction lines.
        # each fermionic loop contributes exactly one momentum loop.
        for i,c in enumerate(cycles):
            loop=[]
            type=[]
            i1 = i_diag[c[0]]
            for i in range(len(c)):
                loop.append(i1)
                type.append(1)
                i1 = diag[i1]
            loop_vertex.append(loop)
            loop_type.append(type)
            if debug: print 'loop=', loop, type
        Loop_vertex.append( loop_vertex )
        Loop_type.append( loop_type )


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
                else: ii = Vindex[lvertex[i]]
                lindex.append(ii)
            loop_index.append(lindex)
        Loop_index.append( loop_index )

        
    return (Loop_vertex, Loop_type, Loop_index )

def ReadPDiagrams(filename):
    #### FIndx[i], tuple(G_Hugen[i]), ' ; ', tuple(Vtyp[i]), ' ; ', Factor[i]
    
    fi = open(filename, 'r')
    diagsG=[]
    Vtype=[]
    indx=[]
    Factor=[]
    s=';'
    while (s):
        try:
            s = fi.next()
            ii, s2 = s.split(' ', 1)
            indx.append( int(ii) )
            ss = s2.split(';')
            diag = eval(ss[0])
            if len(ss)>1:
                vtyp = eval(ss[1])
            else:
                vtyp = tuple(zeros(len(diag)/2, dtype=int))
            fact = 1
            if len(ss)>2: fact = eval(ss[2])
                
            Vtype.append( vtyp )
            diagsG.append( diag )
            Factor.append( fact )
                
        except StopIteration:
            break
    return (diagsG, indx, Vtype, Factor)

def PrintDiags(Loop_vertex, Loop_type):
    Ndiags = len(Loop_vertex)
    for id in range(Ndiags):
        loop_vertex = Loop_vertex[id]
        loop_type  = Loop_type[id]
        for iloop in range(len(loop_vertex)):
            print 'diagram', id, 'and loop', iloop
            lvertex = loop_vertex[iloop]
            ltype  = loop_type[iloop]
            for i in range(len(lvertex)):
                v1, v2 = lvertex[i], lvertex[(i+1)%len(lvertex)]
                lsgn = ltype[i]
                if lsgn==1:
                    print 'G('+str(v1)+'->'+str(v2)+') '
                elif lsgn==-1:
                    print 'G(-('+str(v1)+'->'+str(v2)+')) '
                elif lsgn==2:
                    print 'V('+str(v1)+'->'+str(v2)+') '
                else:
                    print 'V(-('+str(v1)+'->'+str(v2)+')) '
                    
        print 
