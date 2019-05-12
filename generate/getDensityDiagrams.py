#!/usr/bin/env python
# @Copyright 2018 Kristjan Haule 
import sys
from scipy import *

def GetPDiags(fname, BKremove=''):
    #fname = 'input/loops_Pdiags.'+str(Norder)+'_'
    fi = open(fname, 'r')
    s = fi.next()

    diagsG = []
    Loop_vertex = []
    Loop_type = []
    Loop_index = []
    diagVertex = []
    diagSign = []
    indx = []
    Vtype = []
    i=0
    while (s):
        try:
            s = fi.next()
            ii, s2 = s.split(' ', 1)
            indx.append( int(ii) )
            if ';' in s2:
                diag, vtyp = s2.split(';',1)
                Vtype.append( eval(vtyp) )
                diagsG.append( eval(diag) )
            else:
                diagsG.append( eval(s2) )
            s = fi.next()
            ii, n, lvertex = s.split(' ', 2)
            Loop_vertex.append( eval(lvertex.partition('#')[0]) )
            s = fi.next()
            ii, n, vtype = s.split(' ',2)
            Loop_type.append( eval(vtype.partition('#')[0]) )
            s = fi.next()
            ii, n, vind = s.split(' ',2)
            Loop_index.append( eval(vind.partition('#')[0]) )
            s = fi.next()
            ii, n, vind = s.split(' ',2)
            diagVertex.append( eval(vind.partition('#')[0]) )
            s = fi.next()
            ii, n, vind = s.split(' ',2)
            sgn = eval(vind.partition('#')[0])
            if type(sgn)==tuple:
                diagSign.append( sgn )
            else:
                diagSign.append( (sgn,) )
            i+=1
        except StopIteration:
            break
    if BKremove: # remove diagrams, which are not allowed in Baym-Kadanoff approach
        invalid = Find_BK_Ladders(diagsG, Vtype, BKremove)
        print 'BK-invalid=', len(invalid)
        for i in invalid[::-1]:
            del diagsG[i]
            del diagSign[i]
            del diagVertex[i]
            del indx[i]
            del Loop_index[i]
            del Loop_type[i]
            del Loop_vertex[i]
            if (len(Vtype)>0):
                del Vtype[i]
    #else:
    diagsG   = array(diagsG , dtype=int )
    diagVertex = array(diagVertex, dtype=int)
    indx = array(indx, dtype=int)
    Vtype = array(Vtype, dtype=int)
    return (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype)

#def GetPDiags(fname, BKremove=False):
#    fi = open(fname, 'r')
#    s = fi.next()
#
#    diagsG = []
#    Loop_vertex = []
#    Loop_type = []
#    Loop_index = []
#    diagVertex = []
#    diagSign = []
#    indx = []
#    Vtype = []
#    i=0
#    while (s):
#        try:
#            s = fi.next()
#            ii, s2 = s.split(' ', 1)
#            indx.append( int(ii) )
#            if ';' in s2:
#                diag, vtyp = s2.split(';',1)
#                Vtype.append( eval(vtyp) )
#                diagsG.append( eval(diag) )
#            else:
#                diagsG.append( eval(s2) )
#            s = fi.next()
#            ii, n, lvertex = s.split(' ', 2)
#            Loop_vertex.append( eval(lvertex.partition('#')[0]) )
#            s = fi.next()
#            ii, n, vtype = s.split(' ',2)
#            Loop_type.append( eval(vtype.partition('#')[0]) )
#            s = fi.next()
#            ii, n, vind = s.split(' ',2)
#            Loop_index.append( eval(vind.partition('#')[0]) )
#            s = fi.next()
#            ii, n, vind = s.split(' ',2)
#            diagVertex.append( eval(vind.partition('#')[0]) )
#            s = fi.next()
#            ii, n, vind = s.split(' ',2)
#            diagSign.append( eval(vind.partition('#')[0]) )
#            i+=1
#        except StopIteration:
#            break
#    if BKremove: # remove diagrams, which are not allowed in Baym-Kadanoff approach
#        invalid = Find_BK_Ladders(diagsG, Vtype)
#        print 'BK-invalid=', len(invalid)
#        for i in invalid[::-1]:
#            del diagsG[i]
#            del diagSign[i]
#            del diagVertex[i]
#            del indx[i]
#            del Loop_index[i]
#            del Loop_type[i]
#            del Loop_vertex[i]
#            if (len(Vtype)>0):
#                del Vtype[i]
#    else:
#        diagSign = array(diagSign, dtype=float)
#        diagsG   = array(diagsG , dtype=int )
#        diagVertex = array(diagVertex, dtype=int)
#        indx = array(indx, dtype=int)
#        Vtype = array(Vtype, dtype=int)
#    return (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype)

def KeepDensityDiagrams2(diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype):
    diagsG_=[]
    diagSign_=[]
    Loop_index_=[]
    Loop_type_=[]
    Loop_vertex_=[]
    diagVertex_=[]
    indx_=[]
    Vtype_=[]
    for i in range(len(diagsG)):
        if diagsG[i][0]==1:
            ifound=-1
            for ip in range(len(Loop_vertex[i])):
                if Loop_vertex[i][ip]==[0,1]:
                    ifound=ip
                    break
            if ifound<0:
                print 'ERROR: We could not find [0,1] in Loop_index of diagram ', diagsG[i], '! It should not happen!'
            diagsG_.append( diagsG[i] )
            diagSign_.append( diagSign[i] )
            diagVertex_.append( diagVertex[i] )
            indx_.append( indx[i] )
            if (len(Vtype)>0): Vtype_.append( Vtype[i] )
            lindex = Loop_index[i][:]
            del lindex[ifound]
            Loop_index_.append( lindex )
            ltype = Loop_type[i][:]
            del ltype[ifound]
            Loop_type_.append( ltype )
            lvertex = Loop_vertex[i][:]
            del lvertex[ifound]
            Loop_vertex_.append( lvertex)

    diagsG_   = array(diagsG_ , dtype=int )
    #diagSign_ = array(diagSign_, dtype=float)
    diagVertex_ = array(diagVertex_, dtype=int)
    indx_ = array(indx_, dtype=int)
    Vtype_ = array(Vtype_, dtype=int)
    return (diagsG_ , diagSign_, Loop_index_, Loop_type_, Loop_vertex_, diagVertex_, indx_, Vtype_)

if __name__ == '__main__':

    if len(sys.argv)<2:
        print 'Give input filename'
        sys.exit(1)
    fname = sys.argv[1]
    (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = GetPDiags(fname, BKremove=0)
    (diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype) = KeepDensityDiagrams2(diagsG, diagSign, Loop_index, Loop_type, Loop_vertex, diagVertex, indx, Vtype)

    for i in range(len(diagsG)):
        if (len(Vtype)):
            print i, tuple(diagsG[i]), ':', tuple(Vtype[i])
        else:
            print i, tuple(diagsG[i])
        print i, Loop_vertex[i], '# loop vertex'
        print i, Loop_type[i], '# loop type'
        print i, Loop_index[i], '# loop index'
        print i, diagVertex[i].tolist(), '# diag vertex'
        print i, diagSign[i], '# diagram sign'
        
        
        
