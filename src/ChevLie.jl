###########################################################################
##                                                                       ##
#A  chevlie1r3.jl  (Julia module, https://julialang.org/)  Meinolf Geck  ##
##                                                                       ##
#Y  Copyright (C) 2026  Lehrstuhl fuer Algebra,  Universitaet Stuttgart  ##
##                                                                       ##
##  This  file  contains  functions  for constructing  the  Lie algebra  ##
##  and  the  corresponding  Chevalley groups  associated  with  a root  ##
##  system. It is an extension of similar functions written for  GAP in  ##
##  2016;  see http://www.math.rwth-aachen.de/~CHEVIE/contrib.html.      ##
##                                                                       ##
##  It is based on ideas from  Lustigs's theory of canonical bases, see  ##
##                                                                       ##
##  M. Geck,  On  the  construction  of  semisimple  Lie  algebras  and  ##
##  Chevalley  groups,  Proc. Amer. Math. Soc. 145 (2017),   3233--3247  ##
##  (and the references there).                                          ##
##                                                                       ##
##  Some  further explanations  about the programs, and the conventions  ##
##  used, can be found in                                                ##
##                                                                       ##
##  M. Geck,  Computing   Green  functions  in  small  characteristisc,  ##
##  Journal of Algebra 561 (2020), 163--199.                             ##
##                                                                       ##
##  M. Geck,  ChevLie:  Constructing Lie algebras and Chevalley groups,  ##
##  Journal of Software for Algebra and Geometry 10 (2020), 51--60.      ##
##                                                                       ##
##  This program  is  free software:  you can  redistribute  it  and/or  ##
##  modify  it under the terms of  the   GNU General Public License  as  ##
##  published by the  Free Software Foundation, either version 3 of the  ##
##  License, or (at your option) any later version.                      ## 
##                                                                       ##
##  This  program  is distributed in the hope  that it will be  useful,  ##
##  but  WITHOUT ANY WARRANTY;  without  even the implied  warranty  of  ##
##  MERCHANTABILITY  or  FITNESS  FOR A  PARTICULAR  PURPOSE.  See  the  ##
##  GNU General Public License for more details.                         ## 
##                                                                       ##
##  You should have received a  copy of the  GNU General Public License  ##
##  along with this program. If not, see <http://www.gnu.org/licenses/>  ##
##                                                                       ##
"""`ChevLie`

Welcome to version 1.3 of the Julia module `ChevLie`: 

     CONSTRUCTING  LIE  ALGEBRAS  AND  CHEVALLEY  GROUPS  

    Meinolf Geck,  University of Stuttgart,  22 February 2026  

    https://pnp.mathematik.uni-stuttgart.de/idsr/idsr1/geckmf/

    Type   ?LieAlg for first help;    all comments welcome! 
"""
module ChevLie

using SparseArrays

Pkg.add("Nemo); using Nemo
     
import Base.show

export cartanmateps,rootsystem,permcarteps,LieAlg,allelms,allwords,permword,
        wordperm,weightorbit,lietest,checkrels,rep_minuscule,rep_adj,
         rep_sc,reflections,reflsubgrp,canchevbasis,canchevbasis_adj,
          canchevbasis_sc,canchevbasis_min,structconst,structconsts,
           chevrootelt,monomialelts,expliemat,cross_regular,cross_regular1,
            commutator_rels,collect_chevrootelts,rcollect_chevrootelts,
             generic_gram_wdd,weighted_dynkin_diagrams,eval_gram_wdd,
              gram_wdd_search,gfp_points,fq_points,borelcosets,
               twistedborelcosets,borelcosets1,torusorbit,rankmat,
                jordanblocks,partitions,bipartitions,combinations,det_ring,
                 det_field,pfaffian,variables_list,jordandecelt

##########################################################################
#I This file is organised in sections:
#I ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#I Section 1: Some utility functions
#I Section 2: Lie algebras and their representations
#I Section 3: Weighted Dynkin diagrams
#I Section 4: Constructions with Chevalley group elements
##

##########################################################################
##
#I Section 1: Some utility functions
##
"""unpacksparse( <sp> )

unpacks a Julia sparse array to a 2D array. (The Julia function
`Array` does not seem to work for all types of sparse arrays.)
"""
function unpacksparse(sp)
  f=findnz(sp)
  if length(f[3])==0
    return false
  end
  t0=0*f[3][1]
  n=size(sp)[1]
  a=[t0 for i in 1:n,j in 1:n]
  for i in 1:length(f[1])
    a[f[1][i],f[2][i]]=f[3][i]
  end
  return a
end

# apply function to all entries of sparse matrix
function funcsparse(smat,func)
  n=size(smat)[1]
  f=findnz(smat)
  return sparse(f[1],f[2],[func(x) for x in f[3]],n,n)
end

"""`variables_list( <l> )`

produces a list of variables `ChevLie.x1`, `ChevLie.x2`, ... which 
take values `l[1]`,`l[2]`,... (Subsequently, the symbols `:(ChevLie.x1)`,
`:(ChevLie.x2)`, ...  can be evaluated to those values.) Example:

```julia-repl
julia> variables_list([-10,2,19])

julia> ChevLie.x3
19

julia> a=:(ChevLie.x3)
:(ChevLie.x3)

julia> eval(a)
19
```
"""
function variables_list(l)
  for i in 1:length(l)
    s=Symbol("x",i)
    xx=l[i]
    eval(:($s=$xx))
  end
end

"""`nemopol( <R>, <coeff>, <expon> )`

builds a Nemo polynomial over the integers from a list of 
coefficients <coeff> and a corresponding list of exponent 
vectors <expon>.
"""
function nemopol(R,coeff,expon)
  f=Nemo.MPolyBuildCtx(R)
  for i in 1:length(expon)
    Nemo.push_term!(f,Nemo.ZZ(coeff[i]),expon[i])
  end
  Nemo.finish(f)
end

function combinations1(mset::Array{Int,1},m::Int,n::Int,k::Int,
                                        comb::Array{Int,1},i::Int)
  if k==0 
    comb=comb[:]
    combs=[comb]
  else
    combs=Array{Int,1}[]
    for l in m:(n-k+1)
      if l==m || mset[l]!=mset[l-1]
        comb[i]=mset[l]
        append!(combs,combinations1(mset,l+1,n,k-1,comb,i+1))
      end
    end
  end
  return combs
end
    
"""`combinations( <l>, <k> )`

returns all <k>-elements sublists of the list `[1,2,...,l]`. (The code
is copied from GAP.)
"""
function combinations(l::Int,k::Int)
  mset=collect(1:l)
  c=zeros(Int,k)
  return combinations1(mset,1,l,k,c,1)
end
  
function helppart(n::Int,m::Int,part::Array{Int,1},i::Int)
  if n==0
    pp=part[:]
    parts=[pp]
  elseif n<=m
    pp=part[:]
    parts=[pp]
    for l in 2:n
      pp[i]=l
      append!(parts,helppart(n-l,l,pp,i+1));
    end
    for l in i:(i+n-1)
      pp[l]=1
    end
  else
    pp=part[:]
    parts=[pp]
    for l in 2:m
      pp[i]=l
      append!(parts,helppart(n-l,l,pp,i+1))
    end
    for l in i:(i+n-1)
      pp[l]=1
    end
  end
  return parts;
end

"""`partitions( <n> )`

returns all partitions of the integer <n>. (The code is copied from GAP.)
"""
function partitions(n::Int)
  return [[i for i in p if i>0] for p in helppart(n,n,zeros(Int,n),1)]
end

function bipartitions(n::Int)
  pp=Array{Array{Int,1},1}[]
  for l in 0:n
    for p1 in partitions(l),p2 in partitions(n-l)
      push!(pp,[p1,p2])
    end
  end
  return pp
end
       
"""`permsgn( <perm> )`

returns the signum of the permutation <perm>.
"""
function permsgn(perm::Array{Int,1})
  p=[i for i in perm]
  sgn=1
  for i in 1:length(perm)
    if p[i]!=i
      j=findfirst(==(i),p)
      p[i],p[j]=i,p[i]
      sgn=-sgn
    end
  end
  return sgn
end

"""`rankmat( <mat> )`

returns a triple `rk,new,piv` where `rk` is the rank of <mat>,
`new` is a list of rows in echelon form, and `piv` are the positions 
of the pivots in each row of `new`. Here, no permutation of the rows 
have been performed. More precisely, the rows of `new` are non-zero 
and linearly independent; the entries in a column below a pivot 
are zero; the pivots are not 1 but just non-zero. (This works for 
any 2D array with entries in a field.) Example:

```julia-repl
julia> a=cartanmateps(:b,4)[1];

julia> a1=[a[i,j] for i in [4,3,2,1], j in [1,2,3,4]]
4×4 Array{Int8,2}:
  0   0  -1   2
  0  -1   2  -1
 -1   2  -1   0
  2  -2   0   0

julia> using Memo;

julia> r=rankmat(matrix(QQ,a1));

julia> r[1]          # rank of a1
4

julia> r[2]          # echelon form
4-element Array{Array{fmpq,1},1}:
 fmpq[0, 0, -1, 2]
 fmpq[0, -1, 0, 3]
 fmpq[-1, 0, 0, 4]
 fmpq[0, 0, 0, 2] 


julia> r[3]          # positions of pivots
4-element Array{Int64,1}:
 3
 2
 1
 4
```
Using the output of `rankmat`, it is then easy to compute the 
determinant of <mat>; this is done with the function `det_field( <mat> )`.
"""
function rankmat(mat)
  m,n=size(mat)
  if m==0  
    return 0,mat
  end
  # pi contains positions of pivots
  # ri contains inverse of pivot element
  new=Array{typeof(mat[1,1]),1}[]
  piv=Int[]
  ri=typeof(mat[1,1])[]
  rnk=0
  null=0*mat[1,1]
  for i in 1:m
    row=[mat[i,j] for j in 1:n]
    # clear row with rows in new 
    for r in 1:rnk
      if row[piv[r]]!=null
        p=-row[piv[r]]*ri[r]
        for j in piv[r]:n 
          if new[r][j]!=null
            row[j]+=p*new[r][j]
          end
        end
      end
    end
    # if row still non-zero, add to new
    i0=1
    while i0<=n && row[i0]==0
      i0+=1
    end
    if i0<=n 
      push!(piv,i0)
      push!(new,row)
      push!(ri,row[i0]^(-1))
      rnk+=1
    end
  end
  return rnk,new[1:rnk],piv
end

"""`det_ring( <mat> ) and det_field( <mat> )`

returns the determinant of <mat>. In the first case, we simply
develop along the first row (and use recursion); in the second case, 
we first apply `rankmat` and then just multiply together the pivots
(with the correct signs attached).
"""
function det_ring(mat)
  n=size(mat)[1]
  if n==1 
    return mat[1,1]
  else
    null=0*mat[1,1]
    d=null
    for k in 1:n
      if mat[k,1]!=null
        j1=[j for j in 1:n if j!=k]
        if (k % 2)==0
          d+=-mat[k,1]*det_ring(mat[j1,2:n])
        else
          d+=mat[k,1]*det_ring(mat[j1,2:n])
        end
      end
    end
    return d
  end
end
        
function det_field(mat)
  # code copied from rankmat
  n=size(mat)[1]
  new=Array{typeof(mat[1,1]),1}[]
  piv=Int[]
  rnk=0
  null=0*mat[1,1]
  d=mat[1,1]^0
  for i in 1:n
    row=[mat[i,j] for j in 1:n]
    # clear row with rows in new 
    for r in 1:rnk
      if row[piv[r]]!=null
        p=-row[piv[r]]
        for j in piv[r]:n 
          if new[r][j]!=null
            row[j]+=p*new[r][j]
          end
        end
      end
    end
    i0=1
    while i0<=n && row[i0]==0
      i0+=1
    end
    if i0<=n 
      d*=row[i0]
      push!(piv,i0)
      ri=row[i0]^(-1)
      for j in i0:n 
        if row[j]!=null
          row[j]*=ri
        end
      end
      push!(new,row)
      # clear previous rows with row
      #for r in 1:rnk
      #  if new[r][i0]!=null
      #    p=-new[r][i0]
      #    for j in i0:n 
      #      if row[j]!=null
      #        new[r][j]+=p*row[j]
      #     end
      #    end
      #  end
      #end
      rnk+=1
    else
      # det 0 because rows are linearly dependent
      return null
    end
  end
  if permsgn(piv)==-1
    return -d
  else
    return d
  end
end

# helper for pfaffian
function pfaffian1(mat)
  n=size(mat)[1]
  null=0*mat[1,1]
  if n==2   
    return mat[1,2]
  elseif n==4
    return mat[1,2]*mat[3,4]-mat[1,3]*mat[2,4]+mat[2,3]*mat[1,4]
  else
    pf=null
    for i in 2:n
      if mat[1,i]!=null
        l=[j for j in 2:n if j!=i]
        if (i%2)==0
          pf+=mat[1,i]*pfaffian1(mat[l,l])
        else
          pf+=-mat[1,i]*pfaffian1(mat[l,l])
        end
      end
    end
  end
  return pf
end

"""`pfaffian( <mat> )`

returns the pfaffian of the skew-symmetric matrix <mat>. 
(Here, <mat> can be any Julia 2D array, or Nemo matrix.)
"""
function pfaffian(mat)
  n=size(mat)[1]
  null=0*mat[1,1]
  if (n%2)==1 || mat!=-transpose(mat) || any(i->mat[i,i]!=null,1:n)
    println("#W matrix not skew-symmetric")
    flush(stdout)
    return nothing
  end
  if n==2   
    return mat[1,2]
  elseif n==4
    return mat[1,2]*mat[3,4]-mat[1,3]*mat[2,4]+mat[2,3]*mat[1,4]
  else
    print("#I ")
    pf=null
    for i in 2:n
      if mat[1,i]!=null
        print(i," ")
        flush(stdout)
        l=[j for j in 2:n if j!=i]
        if (i%2)==0
          pf+=mat[1,i]*pfaffian1(mat[l,l])
        else
          pf+=-mat[1,i]*pfaffian1(mat[l,l])
        end
      end
    end
    println("")
    return pf
  end
end
        
##########################################################################
##
#Y Section 2: Lie algebras and their representations
##
"""`cartanmateps( <typ>, <l> )`

returns a triple consisting of the Cartan matrix of an irreducible root 
system of type  <typ> (a Julia symbol) and rank <l>,  a list of signs 
`eps[i]` such that `eps[i]=-eps[j]` whenever i,j are joined in the 
Dynkin diagram, and a minimal set of positive integers `symf[i]` 
symmetrizing the Cartan matrix. All `symf[i]` are equal to 1 if
the Cartan matrix is symmetric; otherwise, `symf[i]` equals 1 precisely
when the corresponing simple root is short. Example:

```julia-repl
julia> cartanmateps(:f,4)
(Int8[2 -1 0 0; -1 2 -1 0; 0 -2 2 -1; 0 0 -1 2], Int8[1, -1, 1, -1].
Int8[2, 2, 1, 1])
```

The complete list of Dynkin diagrams with their labelling is as follows:

```
         1   2   3           n                1   2   3           n
   A_n   o---o---o-- . . . --o          B_n   o=<=o---o-- . . . --o

         1   3   4           n                1   2   3           n
   D_n   o---o---o-- . . . --o          C_n   o=>=o---o-- . . . --o
             |
             o 2

         l   2             1   2   3   4          1   3   4   5   6
   G_2   0->-0        F_4  o---o=>=o---o     E_6  o---o---o---o---o
           6                                              |
                                                          o 2

         1   3   4   5   6   7            1   3   4   5   6   7   8
   E_7   o---o---o---o---o---o       E_8  o---o---o---o---o---o---o
                 |                                |
                 o 2                              o 2
```
Here, the same conventions are used as in the original GAP-CHEVIE. In
particular, the `(i,j)`-entry of the Cartan matrix is given by 
`2(e_i,e_j)/(e_i,e_i)`, where the `e_i` are the simple roots and `( , )` 
is an invariant bilinear form. 

For the convenience of the reader, we also provide the following versions
of the diagrams:

```
          1   2          n-1  n                 1   2          n-1  n
   Bp_n   o---o-- . . . --o=>=o          Cp_n   o---o-- . . . --o=<=o

          l   2                 1   2   3   4    
   Gp_2   0-<-0          Fp_4   o---o=<=o---o     
            6                              
```
Example:

```julia-repl
julia> cartanmateps(:fp,4)
(Int8[2 -1 0 0; -1 2 -2 0; 0 -1 2 -1; 0 0 -1 2], Int8[1, -1, 1, -1], 
Int8[1, 1, 2, 2])
```

In all cases, the epsilon function is uniquely determined by the 
condition that `eps[1]=1`.

See also `permcarteps`.
"""
function cartanmateps(typ::Symbol,l::Int)
  a=zeros(Int8,l,l)
  a[1,1]=2
  for i in 2:l
    a[i,i]=2
    a[i,i-1]=-1
    a[i-1,i]=-1
  end
  if typ==:b && l>=2
    a[1,2]=-2
  elseif typ==:bp && l>=2
    a[l,l-1]=-2
  elseif typ==:c && l>=2
    a[2,1]=-2
  elseif typ==:cp && l>=2
    a[l-1,l]=-2
  elseif typ==:d && l>=3
    a[1,2]=0
    a[1,3]=-1
    a[2,1]=0
    a[2,3]=-1
    a[3,1]=-1
    a[3,2]=-1
  elseif typ==:g && l==2
    a[2,1]=-3
  elseif typ==:gp && l==2
    a[1,2]=-3
  elseif typ==:f && l==4
    a[3,2]=-2
  elseif typ==:fp && l==4
    a[2,3]=-2
  elseif typ==:e && l>=6 && l<=8
    a[1,2]=0
    a[1,3]=-1
    a[1,4]=0
    a[2,1]=0
    a[2,3]=0
    a[2,4]=-1
    a[3,1]=-1
    a[3,2]=0
    a[3,4]=-1
    a[4,1]=0
    a[4,2]=-1
    a[4,3]=-1
  end
  eps=ones(Int8,l)
  if typ in [:a,:b,:bp,:c,:cp,:g,:gp,:f,:fp]
    for i in 1:l
      if (i % 2)==0
        eps[i]=-eps[i]
      end
    end
  elseif typ==:d
    for i in 3:l
      if (i % 2)==1
        eps[i]=-eps[i]
      end
    end
  else
    eps[2]=-eps[2]
    eps[3]=-eps[3]
    eps[5]=-eps[5]
    if l>=7
      eps[7]=-eps[7]
    end
  end
  symf=ones(Int8,l)
  if typ==:b
    symf=2*symf
    symf[1]=1
  elseif typ==:bp
    symf=2*symf
    symf[l]=1
  elseif typ==:c
    symf[1]=2
  elseif typ==:cp
    symf[l]=2
  elseif typ==:g
    symf[1]=3
  elseif typ==:gp
    symf[2]=3
  elseif typ==:f
    symf[1]=2
    symf[2]=2
  elseif typ==:fp
    symf[3]=2
    symf[4]=2
  end
  return a,eps,symf
end

"""`permcarteps( <sgnmat>, <perm> )`

applies the permutation <perm> to the output of `cartanmateps`. 
The `(i,j)`-entry of the new Cartan matrix is `<cmat>[<perm>[i],
<perm>[j]]`, the `i`-entry of the new list of signs is `eps[i][<perm>[i]]`
and the `i`-entry of the new list of positive integers is 
`symf[i][<perm>[i]]`.
"""
function permcarteps(sgnmat::Tuple{Array{Int8,2},Array{Int8,1},
                                      Array{Int8,1}},perm::Array{Int,1})
  cmat,eps,symf=sgnmat[1],sgnmat[2],sgnmat[3]
  l=size(cmat)[1]
  c=zeros(Int8,l,l) 
  for i in 1:l, j in 1:l
    c[i,j]=cmat[perm[i],perm[j]]
  end
  return c,eps[perm],symf[perm]
end

function scalrootco(l::Int,cmat::Array{Int8,2},alpha::Array{Int8,1},i::Int)
  scal=Int8(0)
  for j in 1:l
    if cmat[i,j]!=0 && alpha[j]!=0
      scal+=alpha[j]*cmat[i,j]
    end
  end
  return scal
end
  
function reflection(cmat::Array{Int8,2},r::Array{Int8,1},s::Int)
  nr=r[:]
  nr[s]-=sum(cmat[s,i]*nr[i] for i in 1:size(cmat)[1])
  return nr
end

function rootsystem(cmat::Array{Int8,2})
  l=size(cmat)[1]
  r=[zeros(Int8,l) for i in 1:l]
  for i in 1:l
    r[i][i]=1
  end
  for r1 in r
    for s in 1:l
      sc=-scalrootco(l,cmat,r1,s)  
      if sc>0
        r2=r1[:]
        r2[s]+=sc
        if !(r2 in r)
          push!(r,r2)
        end
      end
    end
  end
  sort!(r,rev=true)
  sort!(r,by=sum)
  return r
end

function old_permroots(cmat::Array{Int8,2},r::Array{Array{Int8,1},1})
  l=size(cmat)[1]
  [([findfirst(==(reflection(cmat,i,s)),r) for i in r]...,) for s in 1:l]
end

# this version suggested by Jean Michel, >20 times faster
function permroots(cmat::Array{Int8,2},r::Array{Array{Int8,1},1})
  p1=sortperm(r)
  return [(p1[sortperm(sortperm([reflection(cmat,i,s) 
                            for i in r]))]...,) for s in 1:size(cmat)[1]]
end
  
# is mat upper triangular
function upptr(mat)
  n=size(mat)[1]
  null=0*mat[1,1]
  for i in 2:n
    for j in 1:(i-1)
      if mat[i,j]!=null
        return false
      end
    end
  end
  return true
end

function lowtr(mat)
  n=size(mat)[1]
  null=0*mat[1,1]
  for i in 1:(n-1)
    for j in (i+1):n
      if mat[i,j]!=null
        return false
      end
    end
  end
  return true
end

function diagm(mat)
  n=size(mat)[1]
  null=0*mat[1,1]
  for i in 1:n
    for j in 1:n
      if i!=j && mat[i,j]!=null
        return false
      end
    end
  end
  return true
end

# same for sparse matrices
function supptr(smat)
   f=findnz(smat)
   if length(f[3])==0
     return true
   end
   null=0*f[3][1]
   for i in 1:length(f[1]) 
     if f[1][i]>f[2][i] && f[3][i]!=null
       return false
     end
  end
  return true
end

function slowtr(smat)
   f=findnz(smat)
   if length(f[3])==0
     return true
   end
   null=0*f[3][1]
   for i in 1:length(f[1]) 
     if f[1][i]<f[2][i] && f[3][i]!=null
       return false
     end
  end
  return true
end

function sdiagm(smat)
   f=findnz(smat)
   if length(f[3])==0
     return true
   end
   null=0*f[3][1]
   for i in 1:length(f[1]) 
     if f[1][i]!=f[2][i] && f[3][i]!=null
       return false
     end
  end
  return true
end

function cangens(l::Int,cmat::Array{Int8,2},roots::Array{Array{Int8,1},1})
  r=roots[end:-1:1]
  n1=length(roots)
  append!(r,-roots)
  setr=Set(r)
  ei=SparseMatrixCSC{Int8,Int}[]
  fi=SparseMatrixCSC{Int8,Int}[]
  hi=SparseMatrixCSC{Int8,Int}[]
  for s in 1:l
    meI,meJ,meV=Int64[],Int64[],Int8[]
    for a in 1:length(r)
      if r[a]==-roots[s]
        push!(meI,n1+s)
        push!(meJ,a+l)
        push!(meV,1)
      elseif (r[a]+roots[s]) in setr 
        #b=findfirst(x->x==r[a]+roots[s],r)
        b=findfirst(==(r[a]+roots[s]),r)
        n=1
        while (r[a]-n*roots[s]) in setr 
          n+=1
        end
        if a<=n1
          push!(meI,b)
          push!(meJ,a)
          push!(meV,n)
        else
          push!(meI,b+l)
          push!(meJ,a+l)
          push!(meV,n)
        end
      end
    end
    for j in 1:l
      if cmat[j,s]>=0
        push!(meI,n1-s+1)
        push!(meJ,n1+j)
        push!(meV,cmat[j,s])
      else
        push!(meI,n1-s+1)
        push!(meJ,n1+j)
        push!(meV,-cmat[j,s])
      end
    end
    for i in (length(meI)+1):(2*n1+l)
      push!(meI,i)
      push!(meJ,i)
      push!(meV,0)
    end
    push!(ei,sparse(meI,meJ,meV))
    mfI,mfJ,mfV=Int64[],Int64[],Int8[]
    for a in 1:length(r)
      if r[a]==roots[s]
        push!(mfI,n1+s)
        push!(mfJ,a)
        push!(mfV,1)
      elseif (r[a]-roots[s]) in setr
        #b=findfirst(x->x==r[a]-roots[s],r)
        b=findfirst(==(r[a]-roots[s]),r)
        n=1
        while (r[a]+n*roots[s]) in setr
          n+=1 
        end
        if a<=n1
          push!(mfI,b)
          push!(mfJ,a)
          push!(mfV,n)
        else
          push!(mfI,b+l)
          push!(mfJ,a+l)
          push!(mfV,n)
        end
      end
    end
    for j in 1:l 
      if cmat[j,s]>=0
        push!(mfI,n1+l+s)
        push!(mfJ,n1+j)
        push!(mfV,cmat[j,s])
      else
        push!(mfI,n1+l+s)
        push!(mfJ,n1+j)
        push!(mfV,-cmat[j,s])
      end
    end
    for i in (length(mfI)+1):(2*n1+l)
      push!(mfI,i)
      push!(mfJ,i)
      push!(mfV,0)
    end
    push!(fi,sparse(mfI,mfJ,mfV))
    mhI,mhJ,mhV=Int64[],Int64[],Int8[]
    for a in 1:n1
      push!(mhI,a)
      push!(mhJ,a)
      push!(mhV,scalrootco(l,cmat,r[a],s))
      push!(mhI,n1+l+a)
      push!(mhJ,n1+l+a)
      push!(mhV,scalrootco(l,cmat,r[n1+a],s))
    end
    for a in (n1+1):(n1+l)
      push!(mhI,a)
      push!(mhJ,a)
      push!(mhV,0)
    end
    push!(hi,sparse(mhI,mhJ,mhV))
  end
  omI,omJ,omV=Int64[],Int64[],Int8[]
  for i in 1:(2*n1+l)
    push!(omI,i)
    push!(omJ,2*n1+l+1-i)
    push!(omV,1)
  end
  for i in 1:l
    push!(omI,n1+i)
    push!(omJ,n1+i)
    push!(omV,1)
  end
  return ei,fi,hi,sparse(omI,omJ,omV)
end
  
function minusculeweights(typ::Symbol,l::Int)
  if typ==:a && l>=1
    return [i for i in 1:l]
  elseif typ==:b && l>=2
    return [1]
  elseif typ==:bp && l>=2
    return [l]
  elseif typ==:c && l>=2
    return [l]
  elseif typ==:cp && l>=2
    return [1]
  elseif typ==:d && l>=3
    return [1,2,l]
  elseif typ==:e && l==6
    return [1,6]
  elseif typ==:e && l==7
    return [l]
  else 
    return Int[]
  end
end

"""`LieAlg( <typ>, <l> )`

constructs a simple Lie algebra corresponding to a given Dynkin diagram,
specified by its type <typ> and the rank <l>; see `?cartanmateps` for
the conventions used. (If one wishes to use a labelling other than that 
in `cartanmateps`, then one can specify, as optional arguments, a permutation 
of the index set and a sign; see further below for an example). Here is a 
standard example:

```julia-repl
julia> g=LieAlg(:f,4)
#I dim = 52
LieAlg('F4')

julia> (g.cartan,g.epsilon)   # as in `cartanmateps`.
(Int8[2 -1 0 0; -1 2 -1 0; 0 -2 2 -1; 0 0 -1 2], 
 Int8[1, -1, 1, -1])
```
Altogether, there are the following fields:

  `rnk` (rank),

  `cartan` (Cartan matrix), 

  `N` (number of positive roots), 

  `roots` (all roots), 

  `short` (all short roots, possibly none), 

  `perms` (Weyl group permutation action on roots),

  `dynkin` (Dynkin type), 

  `epsilon` (see `cartanmateps`), 

  `symform` (see `cartanmateps`),

  `permlabel` (identity by default),

  `e_i`, `f_i`, `h_i` (canonical matrices for standard generators, stored 
   as sparse matrices), 

  `minuscule` (minuscule weights, specified by labels in diagram),

  `wmat` (matrices of Weyl group action on weights),

  `representations` (dictionary for representations, initially empty, 
    see also `canchevbasis`),

  `structconsts` (dictionary for structure constants of eps-canonical 
    Chevalley basis, initially empty).

The construction of the matrices in `e_i`, `f_i`, `h_i` is described in 

G. Lusztig, The canonical basis of the quantum adjoint representation, 
J. Comb. Alg. 1 (2017), 45-57; 

M. Geck, On the construction of semisimple Lie algebras and Chevalley
groups, To George Lusztig on his 70th birthday. Proc. Amer. Math. Soc.
145 (2017), 3233--3247.

(They are `canonical`, and do not depend on any choices. With the list of 
signs specified in the field `epsilon`, there are also canonical root 
elements `e_r` for all roots `r`; see `canchevbasis`.)
```julia-repl
julia> g=LieAlg(:a,1)
#I dim = 3
LieAlg('A1')

julia> [Array(a) for a in g.e_i]
1-element Array{Array{Int8,2},1}:
 [0 2 0; 0 0 1; 0 0 0]

julia> [Array(a) for a in g.f_i]
1-element Array{Array{Int8,2},1}:
 [0 0 0; 1 0 0; 0 2 0]

julia> [Array(a) for a in g.h_i]
1-element Array{Array{Int8,2},1}:
 [2 0 0; 0 0 0; 0 0 -2]
```
(In general, each of the fields `e_i,f_i,h_i` contains exactly 
<l> matrices; in the above example, the rank is 1 and so there is
only one matrix in each of those fields. Also note that the matrices
are stored in sparse form, so we need to apply the `Array` function in
order to see the full matrices.) The matrices satisfy the familiar 
Chevalley relations; see the function `checkrels`.

The field `perms` contains the permutations induced by the simple 
reflections of the corresponding Weyl group on the roots. There is some
very basic functionality for producing elements of that Weyl group. Example:
```julia-repl
julia> g=LieAlg(:f,4); allwords(g,3)  # all Weyl group elements 
#I dim = 52                           # of length at most 3
#I 1 4 9 16 
#I Order = 30
30-element Array{Array{Int8,1},1}:
 [] [1] [2] [3] [4] [1, 2] [1, 3] [1, 4] [2, 1] [2, 3]  ...
```
(All elements are produced by `allwords(g)`.) A word <w> (like `[2,1,4]`)
is converted into a permutation by the function `wordperm(<lie>,<w>)`;
conversely, a permutation <perm> on the roots is converted to a word
in the generators by the function `permword(<lie>,<perm>)`.

As mentioned, by default, we use the conventions in `cartanmateps`
for labelling the diagram (and the epsilon function). A different
labelling can be given as follows.

```julia-repl
julia> g1=LieAlg(:f,4,[2,3,4,1],-1)
#I new labels : 
  Int8[2 -1 0 -1; -2 2 -1 0; 0 -1 2 0; -1 0 0 2]
  [1, -1, 1, -1]
#I dim = 52
LieAlg('F4')
julia> g1.cartan==cartanmateps(:f,4)[1][g1.permlabel,g1.permlabel]
true
julia> g1.epsilon==-cartanmateps(:f,4)[2][g1.permlabel]
true
```
Internally, the function `permcarteps` is used to apply the 
permutation `[2,3,4,1]` to the result of `cartanmateps`, and 
then the epsilon-function is further multiplied by `-1`. (That
permutation will be stored in the field `g1.permlabel`.) All the 
subsequent constructions will now be perfomed using the new Cartan 
matrix and the new epsilon function.

See also `cartanmateps`, `checkrels`, `canchevbasis`, `rep_minuscule`, 
`structconst`, `chevrootelt`, `monomialelts`, `weightorbit`, 
`chevrootelt`, `cross_regular`, `weighted_dynkin_diagrams`, `reflsubgrp`.
"""
struct LieAlg
  rnk::Int
  cartan::Array{Int8,2}
  N::Int
  roots::Array{Array{Int8,1},1}
  short::Array{Int,1}
  perms::Array{Tuple{Vararg{Int}},1}
  dynkin::String
  epsilon::Array{Int8,1}
  symform::Array{Int8,1}
  permlabel::Array{Int,1}
  e_i::Array{SparseMatrixCSC{Int8,Int},1}
  f_i::Array{SparseMatrixCSC{Int8,Int},1}
  h_i::Array{SparseMatrixCSC{Int8,Int},1}
  omega::SparseMatrixCSC{Int8,Int}
  wmat::Array{Array{Int8,2}}
  minuscule::Array{Int,1}
  representations::Dict{Symbol,Array{SparseMatrixCSC{Int8,Int},1}}
  structconsts::Dict{Tuple{Int,Int},Tuple{Int,Int}}
  function LieAlg(typ::Symbol,l::Int,x...)
    cme=cartanmateps(typ,l)
    mc=minusculeweights(typ,l)
    if length(x)==0
      cmat,eps,symf=cme
      pl=[i for i in 1:length(eps)]
    else
      pl=x[1]
      cmp=permcarteps(cme,pl)
      cmat,eps,symf=cmp[1],x[2]*cmp[2],x[2]*cmp[3]
      pl1=sortperm(pl)
      mc=[pl1[m] for m in mc]
      println("#I new labels: ")
      println("#I   cartan  = ",cmat)
      println("#I   epsilon = ",eps)
      println("#I   symform = ",symf)
    end
    r=rootsystem(cmat)
    n1=length(r)
    wm=Array{Int8,2}[]
    for i in 1:l
      a=zeros(Int8,l,l)
      for j in 1:l
        for k in 1:l
          if j==k 
            if i==j 
              a[k,j]=1-cmat[k,i]
            else
              a[k,j]=1
            end
          else
            if i==j 
              a[k,j]=-cmat[k,i]
            end
          end
        end
      end
      push!(wm,a)
    end
    for i in 1:l,j in 1:l 
      if symf[i]*cmat[i,j]!=cmat[j,i]*symf[j]
        print("mist ")
      end
    end
    print("#I dim ")
    cg=cangens(l,cmat,r)
    print("= ")
    append!(r,-r)
    g=permroots(cmat,r)
    println(length(r)+l)
    longroots=Int[n1]
    for j in longroots
      for i in 1:l
        j1=g[i][j]
        if !(j1 in longroots)
          push!(longroots,j1)
        end
      end
    end
    sort!(longroots)
    shortr=[i for i in 1:2*n1 if !(i in longroots)] 
    ntyp=string(typ)
    name=uppercase(ntyp[1])
    if length(ntyp)==2
      name=name*ntyp[2]
    end
    name=name*"_"*string(l)
    new(l,cmat,n1,r,shortr,g,name,eps,symf,pl,
                   cg[1],cg[2],cg[3],cg[4],wm,mc,Dict(),Dict())
  end
end

show(io::IO, x::LieAlg) = print(io, "LieAlg('$(x.dynkin)')")

"""`allwords( <lie>, <max> )`

returns all elements of the Weyl group of the Lie algebra <lie>
as reduced expressions; the optional argument <max> specifies 
up to which length the program should proceed.

```julia-repl
julia> g=LieAlg(:f,4); allwords(g,3)  # all Weyl group elements 
#I dim = 52                           # of length at most 3
#I 1 4 9 16
#I Order = 30
30-element Array{Array{Int8,1},1}:
 [] [1] [2] [3] [4] [1, 2] [1, 3] [1, 4] [2, 1] [2, 3]  ...
```
All elements are produced by `allwords(g)`. A word <w> (like `[2,1,4]`)
is converted into a permutation (on the roots) by the function
`wordperm(<lie>,<w>)`; conversely, a permutation <perm> on the roots 
is converted to a word in the generators by the function
`permword(<lie>,<perm>)`.
"""
function allwords(lie::LieAlg,max::Int=-1)
  perms2=[p[1:lie.rnk] for p in lie.perms]
  els=Array{Int8,1}[[]]
  els1=[[i for i in 1:2*lie.N]]
  wels1=Array{Int8,1}[[]]
  print("#I ",length(els1)," ")
  flush(stdout)
  if max==-1
    max=lie.N
  end
  for l in 1:max
    els2=Array{Int,1}[]
    wels2=Array{Int8,1}[]
    cels=Set([])
    for j in 1:length(els1), s in 1:lie.rnk
      w=els1[j]
      if w[s]<=lie.N
        w1=([w[i] for i in perms2[s]]...,)
        if !(w1 in cels)
          push!(cels,w1) 
          push!(els2,[w[i] for i in lie.perms[s]])
          w2=wels1[j][:]
          push!(w2,Int8(s))
          oush!(wels2,w2)
        end
      end
    end
    append!(els,wels2)
    els1=els2[:]
    wels1=wels2[:]
    print(length(els1)," ")
    flush(stdout)
  end
  println()
  println("#I Order = ",length(els))
  return els
end

#function allwords1(lie::LieAlg,max::Int=-1)
#  return [permword(lie,p) for p in allelms(lie,max)]
#end

# convert a word to a permutation
function wordperm(lie::LieAlg,word,lng::Bool=true)
  l=lie.rnk
  p=collect(1:2*lie.N)
  for s in word
    p=[lie.perms[s][i] for i in p]
  end
  if lng==true
    return (p...,)
  else
    return (p[1:l]...,)
  end
end

# convert a permutation to a word
function permword(lie::LieAlg,perm)
  w=Int8[]
  p=[i for i in perm]
  weiter=true
  while weiter
    s=1
    while s<=lie.rnk && p[s]<=lie.N
      s+=1
    end
    if s<=lie.rnk
      p=[p[i] for i in lie.perms[s]]
      push!(w,Int8(s))
    else
      weiter=false
    end
  end
  return w
end
 
#function permword1(lie::LieAlg,perm::Tuple{Vararg{Int}})
#  l=lie.rnk
#  m=[lie.roots[r] for r in perm]
#  w=Int[]
#  weiter=true
#  while weiter
#    s=1
#    while s<=l && all(t->m[s][t]>=0, 1:l)
#      s+=1
#    end
#    if s<=l
#      m=[[m[t][u]-lie.cartan[s,t]*m[s][u] for u in 1:l] for t in 1:l]
#      push!(w,s)
#    else 
#      weiter=false
#    end
#  end
#  return w
#end

function allelms(lie::LieAlg,max::Int=-1)
  gens=lie.perms
  els=[([i for i in 1:lie.rnk]...,)]
  perms2=[p[1:lie.rnk] for p in gens]
  els1=[([i for i in 1:2*lie.N]...,)]
  print("#I ",length(els1)," ")
  flush(stdout)
  if max==-1
    max=lie.N
  end
  for l in 1:max
    els2=Array{Int,1}[]
    cels=Set([])
    for w in els1, s in 1:lie.rnk
      if w[s]<=lie.N
        w1=([w[i] for i in perms2[s]]...,)
        if !(w1 in cels)
          push!(cels,w1) 
          push!(els2,[w[i] for i in gens[s]])
        end
      end
    end
    append!(els,cels)
    els1=els2[:]
    print(length(els1)," ")
    flush(stdout)
  end
  println()
  println("#I Order = ",length(els))
  return els
end
        
# convert short permutation to a long one
function shortlongperm(lie::LieAlg,perm::Tuple{Vararg{Int}})
  l=lie.rnk
  r1=lie.roots
  [findfirst(==([sum([r[i]*r1[perm[i]][s]
          for i in 1:l]) for s in 1:l]),r1) for r in r1]
end

"""`reflections( <lie> )`

returns all reflections in the Weyl group of the Lie algebra <lie>,
as permutations on the roots. The i-th permutation in the list
is the reflection at the i-th root in `<lie>.roots`.
"""
function reflections(lie::LieAlg)
  rr=[collect(p) for p in lie.perms]
  pr=collect(1:lie.rnk)
  for r in rr
    for s in lie.perms
      srs=[s[r[s[i]]] for i in 1:2*lie.N]
      if !(srs in rr)
        push!(rr,srs)
        push!(pr,findfirst(i->srs[i]==lie.N+i,1:lie.N))
      end
    end
  end
  return [(r...,) for r in rr[sortperm(pr)]]
end

"""`reflsubgrp( <lie>, <rr> )`

returns a triple with information concerning the reflection subgroup
`W1` of the Weyl group of the Lie algebra <lie> generated by the 
reflections at the roots in the list <rr>. The first component of the 
output triple contains the canonical system of simple roots for `W1`, the 
second component contains all positive roots of `W1`, and the third 
component the unique reduced representatives in the various cosets `xW1` 
(for `x` in the whole Weyl group). There is an optional third argument 
by which one can specify the maximum length of reduced coset 
representatives to be computed. (The default value is `0`.) Examples:

```julia-repl
julia> l=LieAlg(:f,4); l.N
24
julia> W1=reflsubgrp(l,[1,2,3,48])
#I Delta = [1, 2, 3, 16] 
#I Number of cosets = 1
([1,2,3,16],                                  # simple roots for W1
 [1,2,3,5,6,8,9,11,14,16,18,20,21,22,23,24],  # all positive roots for W1
 Array{Int8,1}[[]])                           # only the identity element
julia> W1=reflsubgrp(l,[3,2,4,32],100)
#I Delta = [2, 3, 4, 8] .
#I Number of cosets = 2
([2, 3, 4, 8], 
 [2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16], 
 Array{Int8,1}[[], [1]])
```
See also `reflections`.

"""
function reflsubgrp(lie::LieAlg,rr::Array{Int,1},maxl::Int=0)
  rfl=reflections(lie)
  psi=rr[:]
  for a in psi
    for r in rr
      if r>lie.N 
        if !(rfl[r-lie.N][a] in psi)
          push!(psi,rfl[r-lie.N][a])
        end
      else
        if !(rfl[r][a] in psi)
          push!(psi,rfl[r][a])
        end
      end
    end
  end
  psi=[r for r in psi if r<=lie.N]
  sort!(psi)   # psi  =positive roots in subgroup
  delta=Int[]  # delta=simple roots for subgroup
  for r in psi 
    if count(a->rfl[r][a]>lie.N,psi)==1
      push!(delta,r)
    end
  end
  print("#I Delta = ",delta," ")
  cs=[collect(1:2*lie.N)] # reduced coset representatives
  csl=[collect(1:2*lie.N)] 
  ll=0
  while ll<maxl && length(csl)>0
    csn=Array{Int,1}[]
    csn1=Set([])
    for w in csl
      wi=sortperm(w)
      for s in 1:lie.rnk
        if wi[s]<=lie.N
          nw=[lie.perms[s][i] for i in w]
          nw1=(nw[1:lie.rnk]...,)
          if !(nw1 in csn1) && all(d->nw[d]<=lie.N,delta)
            push!(csn,nw)
            push!(csn1,nw1)
          end
        end
      end
    end
    if length(csn)>0
      append!(cs,csn)
      print(".")
      flush(stdout)
    end
    csl=[w for w in csn]
    ll+=1
  end
  println("")
  println("#I Number of cosets = ",length(cs))
  return delta,psi,[permword(lie,w) for w in cs]
end
  
"""`checkrels( <lie>, <repe>, <repf>, <reph> )`

checks if the representing matrices in <repe>, <repf>, <reph> satisfy the 
Chevalley relations of the simple Lie algebra <lie>. Here, <repe>, <repf>, 
<reph> are lists of matrices corresponding to the Chevalley generators 
`e_i`, `f_i`, `h_i`, respectively. The relations are:

`[e_i,f_i]=h_i`, `[e_i,f_j]=0` if i is not equal to j,

`[h_i,e_j]=lie.cartan[i,j]*e_j`, `[h_i,f_j]=-lie.cartan[i,j]*f_i`.

We also check if all `e_i` are upper triangular, all `f_i` are lower 
triangular and all `h_i` are diagonal matrices. Example:
```julia-repl
julia>  l=LieAlg(:f,4)
#I dim = 52
LieAlg('F4')

julia> size(l.e_i[1])
(52, 52)

julia> checkrels(l,l.e_i,l.f_i,l.h_i)
  Relations OK 
true
```
See also `LieAlg`.
"""
function checkrels(lie::LieAlg,e::Array{SparseMatrixCSC{Int8,Int},1},
                               f::Array{SparseMatrixCSC{Int8,Int},1},
                               h::Array{SparseMatrixCSC{Int8,Int},1})
  res=true
  l=lie.rnk
  for i in 1:l 
    if supptr(e[i])==false || slowtr(f[i])==false || sdiagm(h[i])==false
      res=false
      println("#W not triangular or diagonal")
    end
  end
  for i in 1:l, j in (i+1):l
    if h[i]*h[j]!=h[j]*h[i]
      res=false
      println("mist1")
    end
  end
  for i in 1:l, j in 1:l
    if i==j
      if (e[i]*f[i]-f[i]*e[i])!=h[i] 
        res=false
        println("mist2")
      end
    else
      if e[i]*f[j]!=f[j]*e[i]
        res=false
        println("mist3")
      end
    end
  end
  for i in 1:l, j in 1:l
    if (h[j]*e[i]-e[i]*h[j])!=(lie.cartan[j,i]*e[i])
      res=false
      println("mist4")
    end
    if (h[j]*f[i]-f[i]*h[j])!=(-lie.cartan[j,i]*f[i])
      res=false
      println("mist5")
    end
  end
  if res==false
    println("  ---> NOT OK <---")
  else
    println("  Relations OK ")
  end
  return res
end
  
"""`weightorbit( <lie>, <w> )`

returns the orbit of the weight <w> under the Weyl group defined by
the Lie algebra <lie>.

```julia-repl
julia> l=LieAlg(:d,20);
#I dim = 780
julia> l.minuscule[1]
1
julia> v=zeros(Int8,20);v[1]=1;
julia> length(weightorbit(l,v))
524288
```
This function is used in `rep_minuscule`.
"""
function weightorbit(lie::LieAlg,w::Array{Int8,1})
  r1=[[lie.cartan[j,i] for j in 1:lie.rnk] for i in 1:lie.rnk]
  orb=[w[:]]
  sorb=Set(orb)
  for o in orb, i in 1:lie.rnk
    if o[i]!=0
      o1=o-o[i]*r1[i]
      if !(o1 in sorb)
        push!(orb,o1)
        push!(sorb,o1)
      end
    end
  end
  return orb
end

# helper function for rep_minuscule
function rep_minorb(lie::LieAlg,r::Array{Array{Int8,1},1})
  l=lie.rnk
  r1=[[lie.cartan[j,i] for j in 1:l] for i in 1:l]
  ei=SparseMatrixCSC{Int8,Int}[]
  fi=SparseMatrixCSC{Int8,Int}[]
  hi=SparseMatrixCSC{Int8,Int}[]
  for s in 1:l
    me=zeros(Int8,length(r),length(r))
    for a in 1:length(r)
      f=findfirst(==(r[a]+r1[s]),r)
      if f!=nothing
        #me[findfirst(x->x==r[a]+r1[s],r),a]=1
        me[f,a]=1
      end
    end
    push!(ei,sparse(me))
    mf=zeros(Int8,length(r),length(r))
    for a in 1:length(r)
      f=findfirst(==(r[a]-r1[s]),r)
      if f!=nothing
        #mf[findfirst(x->x==r[a]-r1[s],r),a]=1
        mf[f,a]=1
      end
    end
    push!(fi,sparse(mf))
    mh=zeros(Int8,length(r),length(r))
    for a in 1:length(r)
      mh[a,a]=r[a][s]
    end
    push!(hi,sparse(mh))
  end
  return ei,fi,hi
end

"""`rep_minuscule( <l>, <w> )`

returns a tuple `ei,fi,hi` of lists which hold the matrices of the 
Chevalley generators of a simple Lie algebra <l> (created by `LieAlg`)
in a representation with a minuscule highest weight <w> (as in
`<l>.minuscule`). Example:

```julia-repl
julia> l=LieAlg(:d,4); l.minuscule
#I dim = 28
#I dim = 28
3-element Array{Int8,1}:
 1
 2
 4
```
(Thera are 3 minuscule weights.)
```julia-repl
julia> m=rep_minuscule(l,4);      # we use l.minuscule[3]=4
#I dim = 8
julia> m[1][1]   # The (sparse) matrix representing `e_1`
8×8 SparseArrays.SparseMatrixCSC{Int8,Int64} with 2 stored entries:
  [3, 4]  =  1
  [5, 6]  =  1
julia> Array(ans) # unpack the latter (sparse) matrix
8×8 Array{Int8,2}:
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  1  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  1  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
```
(Thus, `m[1]` contains the matrices for `e_1,e_2,e_3,e_4`, then `m[2]` 
contains the 4 matrices `f_i`, and `m[3]` contains the 4 matrices `h_i`.)

The function `rep_sc` does a similar thing and returns a tuple
corresponding to the simply-connected representation. In the above
example, this would be the spin representation, that is, the direct 
sum of the two representations with highest weights `m[1]` and `m[2]`.

For convenience, we also provide the function `rep_adj` which simply
returns the fields `e_i`,`f_i`,`h_i` from <l> itself. For type `G_2`, 
`F_4`, `E_8`, there are no minuscule weights. In these cases, `rep_sc` 
does the same as `rep_adj`.
"""
function rep_minuscule(lie::LieAlg,wl::Int,test::Bool=false)
  if !(wl in lie.minuscule)
    println("#W input is not a minuscule weight !")
  end
  w=zeros(Int8,lie.rnk)
  w[wl]=1
  o=weightorbit(lie,w)
  m=rep_minorb(lie,o)
  print("#I dim = ",length(o))
  if test==true
    checkrels(lie,m[1],m[2],m[3])
  else
   println("")
  end
  return m
end
  
function rep_adj(lie::LieAlg,test::Bool=false)
  if test 
    checkrels(lie,lie.e_i,lie.f_i,lie.h_i)
  end
  return lie.e_i,lie.f_i,lie.h_i
end

function rep_sc(lie::LieAlg,test::Bool=false)
  l=lie.rnk
  if l>=1 && lie.dynkin[1]=='A'
    return rep_minuscule(lie,lie.minuscule[1],test)
  elseif l>=2 && lie.dynkin[1]=='B'
    return rep_minuscule(lie,lie.minuscule[1],test)
  elseif l>=2 && lie.dynkin[1]=='C'
    return rep_minuscule(lie,lie.minuscule[1],test)
  elseif l>=3 && lie.dynkin[1]=='D'
    w=zeros(Int8,lie.rnk)
    w[lie.minuscule[1]]=1
    omin=weightorbit(lie,w)
    w[lie.minuscule[1]],w[lie.minuscule[2]]=0,1
    append!(omin,weightorbit(lie,w))
    rep=rep_minorb(lie,omin)
    print("#I dim = ",length(omin))
    if test
      checkrels(lie,rep[1],rep[2],rep[3])
    end
     println("")
    return rep
  elseif l==6 && lie.dynkin[1]=='E'
    return rep_minuscule(lie,lie.minuscule[1],test)
  elseif l==7 && lie.dynkin[1]=='E'
    return rep_minuscule(lie,lie.minuscule[1],test)
  else 
    return lie.e_i,lie.f_i,lie.h_i
  end
end
  
"""`canchevbasis( <l>, <repr> )`

returns the matrices of all root elements `e_r` of the simple
Lie algebra <l> (created using `LieAlg`) in the representation <repr>. 
Here, <repr> is a tuple of lists which hold the representating matrices
for the Chevalley generators of <l>. For example, these matrices
for the adjoint representation are contained in the fields `e_i`,
`f_i` and `h_i` of <l> itself. The root elements `e_r` are the 
canonical ones corresponding to the epsilon function in <l>. Example:

```julia-repl
julia> l=LieAlg(:b,2)
#I dim = 10
LieAlg('B2')

julia> cb=canchevbasis(l,(l.e_i,l.f_i,l.h_i))
8-element Array{SparseArrays.SparseMatrixCSC{Int8,Int64},1}:
...
```
The roots are ordered as in the field `roots` of <l>. The representing
matrices are in `sparse` format. In order to get the full matrices, 
apply `Array`. The construction of the matrices can be done recursively
(on the height of roots), see Section 5 in 

M. Geck, On the construction of semisimple Lie algebras and Chevalley
groups, To George Lusztig on his 70th birthday. Proc. Amer. Math. Soc.
145 (2017), 3233--3247;

```julia-repl
julia> l.roots
8-element Array{Array{Int8,1},1}:
 [1, 0]  
 [0, 1]  
 [1, 1]  
 [2, 1]  
 [-1, 0] 
 [0, -1] 
 [-1, -1]
 [-2, -1]

julia> Array(cb[4])   # the matrix of `e_{[2,1]}`
10×10 Array{Int8,2}:
 0  0  0  0  2  0  0  0  0   0
 0  0  0  0  0  0  1  0  0   0
 0  0  0  0  0  0  0  0  0   0
 0  0  0  0  0  0  0  0  1   0
 0  0  0  0  0  0  0  0  0   1
 0  0  0  0  0  0  0  0  0  -1
 0  0  0  0  0  0  0  0  0   0
 0  0  0  0  0  0  0  0  0   0
 0  0  0  0  0  0  0  0  0   0
 0  0  0  0  0  0  0  0  0   0
```
A short form of the above function is `canchevbasis_adj(l)`. There
are also functions `canchevbasis_sc` and `canchevbasis_min` which do
similar things for the simply-connected representation and 
representations with a minuscule hightest weight, respectively. Once 
computed, the result of any of these functions is added to the field
`representations` of <l>, with keys `:adj`, `:sc` or `:mc1`, `:mc2` etc.
in the case of a minuscule weight.

```julia-repl
julia> l=LieAlg(:e,6)
#I dim = 78
LieAlg('E6')

julia> cb=canchevbasis_min(l,l.minuscule[2]); 
#I dim = 27

julia> l.representations                    # result has been stored
Dict{Symbol,Array{SparseArrays.SparseMatrixCSC{Int8,Int64},1}} with 1 entry:
  :mc6 => SparseArrays.SparseMatrixCSC{Int8,Int64}[...

julia> length(cb)
72
```
(Thus, `cb` is a list containing 72 sparse matrices of size 27x27,
one for each of the 72 roots in <l>.roots.) 

See also `chevrootelt`.
"""
function canchevbasis(lie::LieAlg,repr::Tuple{
               Array{SparseMatrixCSC{Int8,Int},1}, 
               Array{SparseMatrixCSC{Int8,Int},1},
               Array{SparseMatrixCSC{Int8,Int},1}})
  all=SparseMatrixCSC{Int8,Int64}[]
  d=size(repr[1][1])[1]
  setr=Set(lie.roots)
  print("#I calculating eps-canonical base (100/.) ")
  flush(stdout)
  # first positive roots
  for r in 1:lie.N
    if (r%100)==0
      print(".")
      flush(stdout)
    end
    if r<=lie.rnk
      push!(all,lie.epsilon[r]*repr[1][r])
    else
      i=1
      while !((lie.roots[r]-lie.roots[i]) in setr)
        i+=1
      end
      #s=findfirst(x->x==(lie.roots[r]-lie.roots[i]),lie.roots)
      s=findfirst(==(lie.roots[r]-lie.roots[i]),lie.roots)
      q=1
      while (lie.roots[s]-q*lie.roots[i]) in setr
        q+=1
      end
      q=Int8(q)
      a1=lie.epsilon[i]*(all[i]*all[s]-all[s]*all[i])
      for i in 1:d, j in 1:d
        if a1[i,j]!=0
          d1=divrem(a1[i,j],q)
          if d1[2]!=0
            println("#W ---> Mist !!!! <---")
          else
            a1[i,j]=d1[1]
          end
        end
      end
      push!(all,a1)
    end
  end
  # now negative roots
  for r in 1:lie.N
    if (r%100)==0
      print(".")
      flush(stdout)
    end
    if r<=lie.rnk
      push!(all,-lie.epsilon[r]*repr[2][r])
    else
      i=1
      while !((lie.roots[lie.N+r]+lie.roots[i]) in setr)
        i+=1
      end
      #s=findfirst(x->x==(lie.roots[lie.N+r]+lie.roots[i]),lie.roots)
      s=findfirst(==(lie.roots[lie.N+r]+lie.roots[i]),lie.roots)
      q=Int8(1)
      while (lie.roots[s]+q*lie.roots[i]) in setr
        q+=1
      end
      a1=lie.epsilon[i]*(all[s]*all[lie.N+i]-all[lie.N+i]*all[s])
      for i in 1:d, j in 1:d
        if a1[i,j]!=0 
          d1=divrem(a1[i,j],q)
          if d1[2]!=0
            println("#W ---> Mist !!!! <---")
          else
            a1[i,j]=d1[1]
          end
        end
      end
      push!(all,a1)
    end
  end
  println("")
  return all
end

function canchevbasis_adj(lie::LieAlg)
  if :adj in keys(lie.representations)
    return lie.representations[:adj]
  else
    r=canchevbasis(lie,(lie.e_i,lie.f_i,lie.h_i));
    lie.representations[:adj]=r
    return r
  end
end

function canchevbasis_sc(lie::LieAlg)
  if :sc in keys(lie.representations)
    return lie.representations[:sc]
  else
    r=canchevbasis(lie,rep_sc(lie));
    lie.representations[:sc]=r
    return r
  end
end

function canchevbasis_min(lie::LieAlg,m::Int)
  m1=Symbol("mc",string(m))
  if m1 in keys(lie.representations)
    return lie.representations[m1]
  else
    r=canchevbasis(lie,rep_minuscule(lie,m));
    lie.representations[m1]=r
    return r
  end
end

"""`structconst( <l>, <r>, <s> )`

returns the structure constant `N_{rs}` of the simple Lie algebra
<l> (created using `LieAlg`) for the canonical root elements 
`e_r` and `e_s` corresponding to the roots with indices <r> and
<s> in `<l>.roots`. Thus, if `r+s` is a root, then `[e_r,e_s]=
N_{rs}e_{r+s}`; if `r+s` is not a root, we also set `N_{rs}=0`. 
By default, this function uses the result of `canchevbasis_adj`
to compute these coefficients. 

```julia-repl
julia> l=LieAlg(:b,2);
#I dim = 10

julia> for i in 1:8, j in 1:8 println(structconst(l,i,j)) end
(1, 1, 0, 0)
(1, 2, 1, 3)
(1, 3, 2, 4)
(1, 4, 0, 0)
(1, 5, 0, 0)
(1, 6, 0, 0)
(1, 7, 2, 6)
(1, 8, 1, 7)
(2, 1, -1, 3)
(2, 2, 0, 0)
...
```
For example, the output (2,1,-1,3) means that `l.roots[2]+
l.roots[1]=l.roots[3]` is a root and that `N_{rs}=-1`; the output
(1,4,0,0) means that `N_{rs}=0` for `r=l.roots[1]` and `s=l.roots[4]`.

In order to obtain the list of all structure constants, use 
`structconsts( <l> )`; these will then be stored as a dictionary in 
the field `structconsts` of <l>.

See also `canchevbasis`.
"""
function structconst(lie::LieAlg,r::Int,s::Int)
  cb=canchevbasis_adj(lie)
  rs=findfirst(==(lie.roots[r]+lie.roots[s]),lie.roots)
  if rs!=nothing
    #rs=findfirst(x->x==lie.roots[r]+lie.roots[s],lie.roots)
    f=findnz(cb[rs]) 
    l=1
    while f[3][l]==0 
      l+=1
    end
    i,j=f[1][l],f[2][l]
    x=0
    for k in 1:size(cb[1])[1]
      if cb[r][i,k]!=0 && cb[s][k,j]!=0
        x+=cb[r][i,k]*cb[s][k,j]
      end
      if cb[s][i,k]!=0 && cb[r][k,j]!=0 
        x+=-cb[s][i,k]*cb[r][k,j]
      end
    end
    return r,s,div(x,f[3][l]),rs
  else
    return r,s,0,0
  end
end

function structconsts(lie::LieAlg)
  if length(keys(lie.structconsts))==0
    println("#I calculating structconsts")
    flush(stdout)
    for r in 1:(2*lie.N), s in 1:(2*lie.N)
      c=structconst(lie,r,s)
      if c[3]!=0
        lie.structconsts[(r,s)]=(c[3],c[4])
      else
        lie.structconsts[(r,s)]=(c[3],0)
      end
    end
  end
  return lie.structconsts
end

function stringqrs(lie::LieAlg,r::Int,s::Int)
  q=0
  while lie.roots[s]-q*lie.roots[r] in lie.roots
    q+=1
  end
  return q-1
end

function stringprs(lie::LieAlg,r::Int,s::Int)
  p=0
  while lie.roots[s]+p*lie.roots[r] in lie.roots
    p+=1
  end
  return p-1
end

function structconst1(lie::LieAlg,r::Int,s::Int)
  rs=findfirst(==(lie.roots[r]+lie.roots[s]),lie.roots)
  if rs!=nothing
    if r>lie.N && s>lie.N
      return r,s,-structconst1(lie,r-lie.N,s-lie.N)[3],rs
    end
    if r>lie.N && s<=lie.N
      return r,s,-structconst1(lie,s,r)[3],rs
    end
    # now r positive
    if r<=lie.rnk
      return r,s,lie.epsilon[r]*(stringqrs(lie,r,s)+1),rs
    end
    i=1
    while lie.perms[i][r]>=r
      i+=1
    end
    if iseven(stringqrs(lie,i,r)+stringqrs1(lie,i,s)+stringqrs(lie,i,rs))
      return r,s,structconst1(lie,lie.perms[i][r],lie.perms[i][s])[3],rs
    else
      return r,s,-structconst1(lie,lie.perms[i][r],lie.perms[i][s])[3],rs
    end
  else
    return r,s,0,0
  end
end

function structconsts1(lie::LieAlg)
  res=[]
  for r in 1:(2*lie.N), s in 1:(2*lie.N)
    x=structconst1(lie,r,s)
    x1=structconst1(lie,r,s)
    if x!=x1
      print([r,s,x,x1])
      push!(res,x)
    end
  end
  if length(res)==0
    print("Hurra!")
  end
  return Set(res)
end

# helper function for chevrootelt
# assume a1 is nilpotent of order <=4 and t is in a commutative ring
# return Julia 2D array
function expliemat(a1::SparseMatrixCSC{Int8,Int},t)
  d=size(a1)[1]
  t0=0*t
  m=[t0 for i in 1:d, j in 1:d]
  for i in 1:d
    m[i,i]=t^0
  end
  if t==t0 
    return m
  end
  for i in 1:d, j in 1:d
    if a1[i,j]!=0
      m[i,j]+=a1[i,j]*t
    end
  end
  a2=a1*a1
  if count(!iszero,a2)==0
    return m
  end
  for i in 1:d, j in 1:d
    if a2[i,j]!=0
      a2[i,j]=div(a2[i,j],2)
    end
  end
  t2=t*t
  for i in 1:d, j in 1:d
    if a2[i,j]!=0
      m[i,j]+=a2[i,j]*t2
    end
  end
  a3=a2*a1 
  if count(!iszero,a3)==0
    return m
  end
  for i in 1:d, j in 1:d
    if a3[i,j]!=0
      a3[i,j]=div(a3[i,j],3)
    end
  end
  t3=t2*t
  for i in 1:d, j in 1:d
    if a3[i,j]!=0
      m[i,j]+=a3[i,j]*t3
    end
  end
  return m
end 

"""`chevrootelt( <l>, <r>, <t>, <isog> )`

returns the Chevalley group element (as a matrix) usually denoted `x_r(t)`, 
where <r> is a root of the Lie algebra <l> (created using `LieAlg`) and
<t> is an element of any commutative ring (that is available in Julia). 
Furthermore, <isog> specifies the isogeny type of the Chevalley group to
be considered. The default value is `:adj`; further options are `:sc`, or 
`:mc1`, `:mc2` etc. for one of the minuscule weights in `<l>.minuscule`
(see also `canchevbasis`).  Example:

Suppose we want to consider Chevalley root elements in the simply-connected 
group of type `C3`.
```julia-repl
julia> l=LieAlg(:c,3);
#I dim = 21

julia> l.roots[8]
3-element Array{Int8,1}:
 1
 2
 1

julia> chevrootelt(l,8,19,:sc)
6×6 Array{Int64,2}:
 1  0  0  0  -19   0
 0  1  0  0    0  19
 0  0  1  0    0   0
 0  0  0  1    0   0
 0  0  0  0    1   0
 0  0  0  0    0   1
```
In order to work with a Chevalley group over a finite field, one can
use the Julia package Nemo (from http://nemocas.org). 
```julia-repl
julia> using Nemo;

julia> R,x=finite_field(7,3,"x")
(Finite field of degree 3 over F_7, x)

julia> l=LieAlg(:g,2);
#I dim = 14

julia> a=chevrootelt(l,2,x)  # default: adjoint representation
14×14 Array{fq_nmod,2}:
 1  0  0   0     0  0       0   0   0  0   0   0     0       0
 0  1  4*x 3*x^2 0  6*x^2+4 0   0   0  0   0   0     0       0
 0  0  1   5*x   0  x^2     0   0   0  0   0   0     0       0
 0  0  0   1     0  6*x     0   0   0  0   0   0     0       0
 0  0  0   0     1  0       6*x 5*x 0  x^2 0   0     0       0
 0  0  0   0     0  1       0   0   0  0   0   0     0       0
 0  0  0   0     0  0       1   0   0  0   0   0     0       0
 0  0  0   0     0  0       0   1   0  6*x 0   0     0       0
 0  0  0   0     0  0       0   0   1  0   4*x 3*x^2 6*x^2+4 0
 0  0  0   0     0  0       0   0   0  1   0   0     0       0
 0  0  0   0     0  0       0   0   0  0   1   5*x   x^2     0
 0  0  0   0     0  0       0   0   0  0   0   1     6*x     0
 0  0  0   0     0  0       0   0   0  0   0   0     1       0
 0  0  0   0     0  0       0   0   0  0   0   0     0       1

julia> a^7      # the matrix a has order 7
14×14 Array{fq_nmod,2}:
 1  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  1  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  1  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  1  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  1  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  1  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  1  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  1  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  1  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  1  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  1  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  1
```
See also `canchevbasis`, `monomialelts`, `cross_regular`, `commutator_rels`, 
`collect_chevrootelts`.
"""
function chevrootelt(lie::LieAlg,r::Int,t,isog=:adj)
  if isog==:adj 
    cb=canchevbasis_adj(lie)
  elseif isog==:sc 
    cb=canchevbasis_sc(lie)
  else 
    cb=canchevbasis_min(lie,isog)
  end
  return expliemat(cb[r],t)
end

"""`monomialelts( <l>, <repr>, <t> )`

returns the lifts of the simple reflections of the Weyl group of the 
Lie algebra <l> (created by `LieAlg`) to elements in the representation 
of <li> that is specified by <repr>. For example, for the adjoint
representation, we can take <repr> to be the tuple formed by the fields 
`e_i,f_i,h_i` in <l> itself. Then the monomial elements `n_i` are
given by `n_i=exp(e_i)exp(-f_i)exp(e_i)`. (These satisfy the braid 
relations, and have order 2 or 4.)

```julia-repl
julia> l=LieAlg(:a,2);m=monomialelts(l,rep_adj(l));
#I dim = 8

julia> display(m[1])
8×8 Array{Int64,2}:
  0  1  0   0   0  0   0  0
 -1  0  0   0   0  0   0  0
  0  0  0   0   0  1   0  0
  0  0  0  -1  -1  0   0  0
  0  0  0   0   1  0   0  0
  0  0  1   0   0  0   0  0
  0  0  0   0   0  0   0  1
  0  0  0   0   0  0  -1  0

julia> m[1]*m[2]*m[1]==m[2]*m[1]*m[2]
true
```
There is an optional third argument by which one can specify
a non-zero element <t> in a field, in which case one obtains
the elements usually denoted `n_i(t)`.
"""
function monomialelts(lie::LieAlg,repr::Tuple{
               Array{SparseMatrixCSC{Int8,Int},1},
               Array{SparseMatrixCSC{Int8,Int},1},
               Array{SparseMatrixCSC{Int8,Int},1}},t=1)
  if t==t^0 || t==-t^0
    t1=t
  else
    t1=t^(-1)
  end
  gens1=[expliemat(repr[1][s],t) for s in 1:lie.rnk]
  gens2=[expliemat(repr[2][s],-t1) for s in 1:lie.rnk]
  return [gens1[s]*gens2[s]*gens1[s] for s in 1:lie.rnk]
end

# helper for commutator relations
function liemrsi(lie::LieAlg,r::Int,s::Int,i::Int)
  c=lie.structconsts[(r,s)]
  if i==1 && c[2]!=0
    return c[1],c[2]
  elseif i==2 && c[2]!=0
    c2=lie.structconsts[(r,c[2])]
    return div(c[1]*c2[1],2),c2[2]
  elseif i==3 && c[2]!=0
    c2=lie.structconsts[(r,c[2])]
    if c2[2]!=0 
      c3=lie.structconsts[(r,c2[2])]
      return div(c[1]*c2[1]*c3[1],6),c3[2]
    end
  end
  return 0,0
end

#function checkcommrel(lie::LieAlg,r::Int,s::Int,cr)
#  for tt in [[2,19],[1009,211],[23,10007],[17,-769],[-179,131]] 
#    t=tt[1];
#    u=tt[2];
#    xr=sparse(chevrootelt(lie,r,t))
#    xs=sparse(chevrootelt(lie,s,u))
#    xr1=sparse(chevrootelt(lie,r,-t))
#    xs1=sparse(chevrootelt(lie,s,-u))
#    a1=xs1*xr1*xs*xr
#    a2=sparse(chevrootelt(lie,r,0))
#    for l in cr 
#      a2*=sparse(chevrootelt(lie,l[4],l[3]*t^l[1]*u^l[2]))
#    end
#    if a1!=a2
#      return false 
#    end
#  end
#  return true
#end

function checkcommrel(lie::LieAlg,r::Int,s::Int,cr)
  R,tt=Nemo.polynomial_ring(Nemo.ZZ,["T","U"])
  t=tt[1];
  u=tt[2];
  xr=chevrootelt(lie,r,t)
  xs=chevrootelt(lie,s,u)
  xr1=chevrootelt(lie,r,-t)
  xs1=chevrootelt(lie,s,-u)
  a1=xs1*xr1*xs*xr
  a2=chevrootelt(lie,r,0)
  for l in cr
    a2*=chevrootelt(lie,l[4],l[3]*t^l[1]*u^l[2])
  end
  for i in 1:size(a1)[1]
    for j in 1:size(a1)[1]
      if a1[i,j]!=a2[i,j]
        return false
      end
    end
  end
  return true
end

"""`commutator_rels( <lie>, <r>, <l> )`

returns the structure constants `C(i,j,r,s)` in  the Chevalley commutator 
relations for two linearly independent roots <r>, <s> with respect to the 
canonical Chevalley basis. The formula is as follows (see R. W. Carter, 
Simple groups of Lie type, p.76): For `t,u` in the base ring, we have

     `x_s(u)^(-1)x_r(t)^(-1)x_s(u)x_r(t)
                       = prod_{i,j>0} x_{ir+js}(C(i,j,r,s)t^iu^j)`

(Note that a minus sign in Carter's formula is absorbed into `C`.) Example:

```julia-repl
julia> l=LieAlg(:g,2); commutator_rels(l,1,2)
#I calculating structconsts
#I calculating eps-canonical base (100/.) 
4-element Array{Array{Int64,1},1}:
 [1, 1, -1, 3]
 [1, 2, -1, 4]
 [1, 3, -1, 5]
 [2, 3, 2, 6] 
```
This means that for `r=1` and `s=2` we have: 
`C(1,1,r,s)=C(1,2,r,s)=C(1,3,r,s)=-1` and `C(2,3,r,s)=2` (all other
`C(i,j,r,s)` are zero). Furthermore, `r+s` is root no. 3,
`r+2s` is root no. 4, etc.

See also `collect_chevrootelts`.
"""
function commutator_rels(lie::LieAlg,r::Int,s::Int,test::Bool=false)
  #if r==s || lie.roots[r]==-lie.roots[s]
  #  println("#W roots linearly dependent")
  #end
  str=structconsts(lie)
  t=lie.structconsts[(r,s)]
  cr=Array{Int,1}[]
  if t[1]!==0
    m=liemrsi(lie,r,s,1)
    if m[1]!=0 push!(cr,[1,1,-m[1],m[2]]) end
    m=liemrsi(lie,s,r,2) 
    if m[1]!=0 push!(cr,[1,2,-m[1],m[2]]) end
    m=liemrsi(lie,r,s,2)
    if m[1]!=0 push!(cr,[2,1,m[1],m[2]]) end
    m=liemrsi(lie,s,r,3)
    if m[1]!=0 push!(cr,[1,3,m[1],m[2]]) end
    m=liemrsi(lie,r,s,3)
    if m[1]!=0 push!(cr,[3,1,-m[1],m[2]]) end
    m=liemrsi(lie,t[2],s,2)
    if m[1]!=0 push!(cr,[2,3,-div(2*m[1],3),m[2]]) end
    m=liemrsi(lie,t[2],r,2)
    if m[1]!=0 push!(cr,[3,2,-div(m[1],3),m[2]]) end
  end
  if test==true && r!=s && lie.roots[r]!=-lie.roots[s]
    println("#I commutator relation ",checkcommrel(lie,r,s,cr))
  end
  return cr
end
##Carter, p.76: C(i,1,r,s)=M(r,s,i), C(1,j,r,s)=(-1)^j*M(s,r,j)
## C(3,2,r,s)=M(r+s,r,2)/3, C(2,3,r,s)=-2*M(s+r,s,2)/3 where
## M(r,s,i)=N(r,s)N(r,r+s)...N(r,(i-1)r+s)/i! 

function structconsts2(lie::LieAlg,test::Bool=false)
  #if length(keys(lie.structconsts2))==0
  #  println("#I calculating structconsts2")
  #  flush(stdout)
  str2=Dict()
  for r in 1:lie.N, s in 1:lie.N
    str2[(r,s)]=commutator_rels(lie,r,s,test)
  end
  return str2
end

"""`collect_chevrootelts( <lie>, <wort>, <wgt> )`

collects a word <wort> in (positive) Chevalley root elements according 
to a vector of weights <wgt> (one for each positive root). The word is 
given as a sequence of pairs `[r,t]` where `r` is the index of a positive 
root and `t` is an element from the base ring. The result is a new such 
sequence that is in a normal form as defined by the given weights.
(The default value for <wgt> is the sequence `[1..<lie>.N]`.) Example:
```julia-repl
julia> l=LieAlg(:b,2);
julia> using Nemo
julia> R,t=polynomial_ring(ZZ,["t"*string(i) for i in 1:8]);
julia> collect_chevrootelts(l,[[1,t[1]],[2,t[2]],[3,t[3]],[4,t[4]],
                               [1,t[5]],[2,t[6]],[3,t[7]],[4,t[8]]])
4-element Array{Array{Any,1},1}:
 [1, t1+t5]                
 [2, t2+t6]                
 [3, -t2*t5+t3+t7]         
 [4, t2*t5^2-2*t3*t5+t4+t8]
```
Thus, the product of two general elements in a group of type `B_2` is 
given by 

  `x_1(t1+t5)*x_2(t2+t6)*x_3(-t2*t5+t3+t7)*x_4(t2*t5^2-2*t3*t5+t4+t8).`

```julia-repl
julia> collect_chevrootelts(l,[[1,t[1]],[2,t[2]],[3,t[3]],[4,t[4]],
     [1,t[5]],[2,t[6]],[3,t[7]],[4,t[8]]],[4,3,2,1]) # reverse order 
4-element Array{Array{Any,1},1}:
 [4, t1^2*t2+t1^2*t6+2*t1*t3+2*t1*t5*t6+2*t1*t7+t4+t5^2*t6+2*t5*t7+t8]
 [3, t1*t2+t1*t6+t3+t5*t6+t7]                                         
 [2, t2+t6]                                                           
 [1, t1+t5]    
```
The function uses collection from the left. In some situations, it may
be more efficient to use collection from the right; this can be done
with the analogous function `rcollect_chevrootelts'.

See also `commutator_rels`.
"""
function collect_chevrootelts(lie::LieAlg,wort,
                         wgt::Array{Int,1}=Int[],test::Bool=false)
  if length(wgt)==0
    pos=collect(1:lie.N)
  else
    pos=wgt
  end
  nw=[p[:] for p in wort]
  weiter=true
  while weiter
    while weiter # remove consecutive doubles
      i0=1
      while i0<length(nw) && nw[i0][1]!=nw[i0+1][1]
        i0+=1
      end
      if i0>=length(nw)
        weiter=false
      else
        if nw[i0][2]==-nw[i0+1][2]
          deleteat!(nw,[i0,i0+1])
        else
          nw[i0][2]+=nw[i0+1][2]
          deleteat!(nw,i0+1)
        end
      end
    end
    i0=1
    while i0<length(nw) && pos[nw[i0][1]]<=pos[nw[i0+1][1]]
      i0+=1
    end
    if i0<length(nw)
      cr=commutator_rels(lie,nw[i0+1][1],nw[i0][1],test)
      sp=[nw[i0+1],nw[i0]] 
      for l in cr 
        push!(sp,[l[4],l[3]*nw[i0+1][2]^l[1]*nw[i0][2]^l[2]])
      end
      splice!(nw,i0:i0+1,sp)
      weiter=true
    end
  end
  if test==true && length(wort)>=1
    a1=sparse(chevrootelt(lie,wort[1][1],wort[1][2]))
    for i in 2:length(wort) 
      a1*=sparse(chevrootelt(lie,wort[i][1],wort[i][2]))
    end
    a2=sparse(chevrootelt(lie,wort[1][1],0*wort[1][2]))
    for i in 1:length(nw) 
      a2*=sparse(chevrootelt(lie,nw[i][1],nw[i][2]))
    end
    println(a1==a2)
  end
  return nw
end

# collection from the right
function rcollect_chevrootelts(lie::LieAlg,wort,
                         wgt::Array{Int,1}=Int[],test::Bool=false)
  if length(wgt)==0
    pos=collect(1:lie.N)
  else
    pos=wgt
  end
  nw=[p[:] for p in wort]
  weiter=true
  while weiter
    while weiter # remove consecutive doubles
      i0=1
      while i0<length(nw) && nw[i0][1]!=nw[i0+1][1]
        i0+=1
      end
      if i0>=length(nw)
        weiter=false
      else
        if nw[i0][2]==-nw[i0+1][2]
          deleteat!(nw,[i0,i0+1])
        else
          nw[i0][2]+=nw[i0+1][2]
          deleteat!(nw,i0+1)
        end
      end
    end
    i0=length(nw)-1
    while i0>=1 && pos[nw[i0][1]]<=pos[nw[i0+1][1]]
      i0+=-1
    end
    if i0>=1
      cr=commutator_rels(lie,nw[i0+1][1],nw[i0][1])
      if length(cr)==0
        nw[i0],nw[i0+1]=nw[i0+1],nw[i0]
      else
        sp=[nw[i0+1],nw[i0]] 
        for l in cr 
          push!(sp,[l[4],l[3]*nw[i0+1][2]^l[1]*nw[i0][2]^l[2]])
        end
        splice!(nw,i0:i0+1,sp)
      end
      weiter=true
    end
  end
  if test==true && length(wort)>=1
    a1=sparse(chevrootelt(lie,wort[1][1],wort[1][2]))
    for i in 2:length(wort) 
      a1*=sparse(chevrootelt(lie,wort[i][1],wort[i][2]))
    end
    a2=sparse(chevrootelt(lie,wort[1][1],0*wort[1][2]))
    for i in 1:length(nw) 
      a2*=sparse(chevrootelt(lie,nw[i][1],nw[i][2]))
    end
    println(a1==a2)
  end
  return nw
end

# slightly different collection from left, but only correct for 
# increasing wgt function
function collect_chevrootelts1(lie::LieAlg,wort,
                         wgt::Array{Int,1}=Int[],test::Bool=false)
  if length(wgt)==0
    pos=collect(1:lie.N)
  else
    pos=wgt
  end
  pos0=collect(Set(pos))
  sort!(pos0)
  nw=[p[:] for p in wort]
  for p0 in pos0
    weiter=true
    while weiter
      while weiter # remove consecutive doubles
        i1=1
        while i1<length(nw) && nw[i1][1]!=nw[i1+1][1]
          i1+=1
        end
        if i1>=length(nw)
          weiter=false
        else
          if nw[i1][2]==-nw[i1+1][2]
            deleteat!(nw,[i1,i1+1])
          else
            nw[i1][2]+=nw[i1+1][2]
            deleteat!(nw,i1+1)
          end
        end
      end
      i0=1
      while weiter==false && i0<length(nw) 
        if p0==wgt[nw[i0+1][1]] && p0<wgt[nw[i0][1]]
          cr=commutator_rels(lie,nw[i0+1][1],nw[i0][1])
          sp=[nw[i0+1],nw[i0]] 
          for l in cr 
            push!(sp,[l[4],l[3]*nw[i0+1][2]^l[1]*nw[i0][2]^l[2]])
          end
          splice!(nw,i0:i0+1,sp)
          weiter=true
        else
          i0+=1
        end
      end 
    end
  end
  if test==true && length(wort)>=1
    a1=sparse(chevrootelt(lie,wort[1][1],wort[1][2]))
    for i in 2:length(wort) 
      a1*=sparse(chevrootelt(lie,wort[i][1],wort[i][2]))
    end
    a2=sparse(chevrootelt(lie,wort[1][1],0*wort[1][2]))
    for i in 1:length(nw) 
      a2*=sparse(chevrootelt(lie,nw[i][1],nw[i][2]))
    end
    println(a1==a2)
  end
  return nw
end

# collection from the right, for use in borelcosets 
function bcollect_chevrootelts(lie::LieAlg,wort,
                         wgt::Array{Int,1}=Int[],test::Bool=false)
  if length(wgt)==0
    pos=collect(1:lie.N)
  else
    pos=wgt
  end
  pos0=collect(Set(pos))
  sort!(pos0)
  nw=[p[:] for p in wort]
  for p0 in pos0
    weiter=true
    while weiter
      while weiter # remove consecutive doubles
        i1=1
        while i1<length(nw) && nw[i1][1]!=nw[i1+1][1]
          i1+=1
        end
        if i1>=length(nw)
          weiter=false
        else
          if nw[i1][2]==-nw[i1+1][2]
            deleteat!(nw,[i1,i1+1])
          else
            nw[i1][2]+=nw[i1+1][2]
            deleteat!(nw,i1+1)
          end
        end
      end
      i0=length(nw)-1
      while weiter==false && i0>=1 
        if p0==wgt[nw[i0+1][1]] && p0<wgt[nw[i0][1]]
          cr=commutator_rels(lie,nw[i0+1][1],nw[i0][1])
          sp=[nw[i0+1],nw[i0]] 
          for l in cr 
            push!(sp,[l[4],l[3]*nw[i0+1][2]^l[1]*nw[i0][2]^l[2]])
          end
          splice!(nw,i0:i0+1,sp)
          weiter=true
        else
          i0+=-1
        end
      end 
    end
  end
  if test==true && length(wort)>=1
    a1=sparse(chevrootelt(lie,wort[1][1],wort[1][2]))
    for i in 2:length(wort) 
      a1*=sparse(chevrootelt(lie,wort[i][1],wort[i][2]))
    end
    a2=sparse(chevrootelt(lie,wort[1][1],0*wort[1][2]))
    for i in 1:length(nw) 
      a2*=sparse(chevrootelt(lie,nw[i][1],nw[i][2]))
    end
    println(a1==a2)
  end
  return nw
end

"""`torusorbit( <lie>, <wort> )`

returns the orbit of a word <wort> in (positive) Chevalley root elements 
under the action of the diagonal elements of order 2. (This is used to 
check to what extent the word depends on the choice of the signs in the 
Chevalley basis of <lie>.)
"""
function torusorbit(lie::LieAlg,wort)
  chi=[[(-1)^sum([r[j]*lie.cartan[i,j] for j in 1:lie.rnk]) 
               for r in lie.roots[1:lie.N]] for i in 1:lie.rnk]
  orb=[wort]
  for w in orb
    for i in 1:lie.rnk
      nw=[[p[1],chi[i][p[1]]*p[2]] for p in w]
      if !(nw in orb)
        push!(orb,nw)
      end
    end
  end
  return orb
end

##########################################################################
##
#Y Section 3: Weighted Dynkin diagrams
## 

# helper function for weighted_dynkin_diagrams in type B_n
function bipartitionsB(n::Int)
  pp=Array{Array{Int,1},1}[]
  for l in 0:n 
    for p1 in partitions(l),p2 in partitions(2*n+1-2*l)
      if all(x->p2[x]!=p2[x+1],1:(length(p2)-1)) && all(x->(x%2)==1,p2)
        push!(pp,[p1,p2])
      end
    end
  end
  return pp
end

# helper function for weighted_dynkin_diagrams in type C_n
function bipartitionsC(n::Int)
  [p for p in bipartitions(n) if all(x->p[2][x]!=p[2][x+1],
                                                1:(length(p[2])-1))]
end

# helper function for weighted_dynkin_diagrams in type D_n
function bipartitionsD(n::Int)
  pp=Array{Array{Int,1},1}[]
  for l in 0:n 
    for p1 in partitions(l),p2 in partitions(2*n-2*l)
      if all(x->p2[x]!=p2[x+1],1:(length(p2)-1)) && all(x->(x%2)==1,p2)
        push!(pp,[p1,p2])
      end
    end
  end
  return pp
end

# helper for weighted_dynkin_diagrams
function partitiontowdd(n,p)
  xi=[]
  for a in p 
    a1=a-1
    while a1>0 
      push!(xi,a1)
      push!(xi,-a1)
      a1=a1-2
    end
    if (a % 2)==1 
      push!(xi,0)
    end
  end
  sort!(xi,rev=true)
  return xi
end

function wdd_data(typ::Symbol,l::Int)
  if typ==:a && l>=1
    wdd=Array{Int,1}[]
    for p in partitions(l+1)
      xi=partitiontowdd(l+1,p)
      push!(wdd,[xi[i]-xi[i+1] for i in 1:l])
    end
  elseif typ==:b && l>=2 
    wdd=Array{Int,1}[]
    for p in bipartitionsB(l) 
      np=[]
      for i in p[1]
        push!(np,i,i)
      end
      for i in p[2]
        push!(np,i)
      end
      xi=partitiontowdd(2*l+1,np)
      xi1=[xi[i]-xi[i+1] for i in 1:(l-1)]
      push!(xi1,xi[l])
      push!(wdd,xi1[end:-1:1])
    end
  elseif typ==:c && l>=2 
    wdd=Array{Int,1}[]
    for p in bipartitionsC(l) 
      np=[]
      for i in p[1]
        push!(np,i,i)
      end
      for i in p[2]
        push!(np,2*i)
      end
      xi=partitiontowdd(2*l,np)
      xi1=[xi[i]-xi[i+1] for i in 1:(l-1)]
      push!(xi1,2*xi[l])
      push!(wdd,xi1[end:-1:1])
    end
  elseif typ==:d && l>=3 
    wdd=Array{Int,1}[]
    for p in bipartitionsD(l) 
      np=[]
      for i in p[1]
        push!(np,i,i)
      end
      for i in p[2]
        push!(np,i)
      end
      xi=partitiontowdd(2*l,np)
      xi1=[xi[i]-xi[i+1] for i in 1:(l-2)]
      push!(xi1,xi[l-1]-xi[l])
      push!(xi1,xi[l-1]+xi[l])
      push!(wdd,xi1[end:-1:1])
      if length(p[2])==0 && all(x->(x%2)==0,p[1])
        xi2=xi1[:]
        xi2[l-1],xi2[l]=xi2[l],xi2[l-1]
        push!(wdd,xi2[end:-1:1])
      end
    end
  elseif typ==:g && l==2
    wdd=[[0,0],[0,1],[1,0],[0,2],[2,2]]
  elseif typ==:f && l==4
    wdd=[[0,0,0,0],[1,0,0,0],[0,0,0,1],[0,1,0,0],[2,0,0,0],[0,0,0,2],
         [0,0,1,0],[2,0,0,1],[0,1,0,1],[1,0,1,0],[0,2,0,0],[2,2,0,0],
         [1,0,1,2],[0,2,0,2],[2,2,0,2],[2,2,2,2]] 
  elseif typ==:e && l==6
    wdd=[[0,0,0,0,0,0],[0,1,0,0,0,0],[1,0,0,0,0,1],[0,0,0,1,0,0],
         [0,2,0,0,0,0],[1,1,0,0,0,1],[2,0,0,0,0,2],[0,0,1,0,1,0],
         [1,2,0,0,0,1],[1,0,0,1,0,1],[0,1,1,0,1,0],[0,0,0,2,0,0],
         [2,2,0,0,0,2],[0,2,0,2,0,0],[1,1,1,0,1,1],[2,1,1,0,1,2],
         [1,2,1,0,1,1],[2,0,0,2,0,2],[2,2,0,2,0,2],[2,2,2,0,2,2],
         [2,2,2,2,2,2]]
  elseif typ==:e && l==7
    wdd=[[0,0,0,0,0,0,0],[1,0,0,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,2],
         [0,0,1,0,0,0,0],[2,0,0,0,0,0,0],[0,1,0,0,0,0,1],[1,0,0,0,0,1,0],
         [0,0,0,1,0,0,0],[2,0,0,0,0,1,0],[0,0,0,0,0,2,0],[0,2,0,0,0,0,0],
         [2,0,0,0,0,0,2],[0,0,1,0,0,1,0],[1,0,0,1,0,0,0],[0,0,2,0,0,0,0],
         [1,0,0,0,1,0,1],[2,0,2,0,0,0,0],[0,1,1,0,0,0,1],[0,0,0,1,0,1,0],
         [2,0,0,0,0,2,0],[0,0,0,0,2,0,0],[2,0,0,0,0,2,2],[2,1,1,0,0,0,1],
         [1,0,0,1,0,1,0],[2,0,0,1,0,1,0],[0,0,0,2,0,0,0],[1,0,0,1,0,2,0],
         [1,0,0,1,0,1,2],[2,0,0,0,2,0,0],[0,1,1,0,1,0,2],[0,0,2,0,0,2,0],
         [2,0,2,0,0,2,0],[0,0,0,2,0,0,2],[0,0,0,2,0,2,0],[2,1,1,0,1,1,0],
         [2,1,1,0,1,0,2],[2,0,0,2,0,0,2],[2,1,1,0,1,2,2],[2,0,0,2,0,2,0],
         [2,0,2,2,0,2,0],[2,0,0,2,0,2,2],[2,2,2,0,2,0,2],[2,2,2,0,2,2,2], 
         [2,2,2,2,2,2,2]]
  elseif typ==:e && l==8
    wdd=[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0], 
         [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,2],[0,1,0,0,0,0,0,0],
         [1,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0],[1,0,0,0,0,0,0,2],
         [0,0,1,0,0,0,0,0],[2,0,0,0,0,0,0,0],[1,0,0,0,0,0,1,0],
         [0,0,0,0,0,1,0,1],[0,0,0,0,0,0,2,0],[0,0,0,0,0,0,2,2],
         [0,0,0,0,1,0,0,0],[0,0,1,0,0,0,0,1],[0,1,0,0,0,0,1,0],
         [1,0,0,0,0,1,0,0],[2,0,0,0,0,0,0,2],[0,0,0,1,0,0,0,0],
         [0,1,0,0,0,0,1,2],[0,2,0,0,0,0,0,0],[1,0,0,0,0,1,0,1],
         [1,0,0,0,1,0,0,0],[1,0,0,0,0,1,0,2],[0,0,0,1,0,0,0,1],
         [0,0,0,0,0,2,0,0],[2,0,0,0,0,1,0,1],[0,0,0,1,0,0,0,2],
         [0,0,1,0,0,1,0,0],[0,2,0,0,0,0,0,2],[2,0,0,0,0,0,2,0],
         [2,0,0,0,0,0,2,2],[0,0,0,1,0,0,1,0],[1,0,0,1,0,0,0,1],
         [0,0,1,0,0,1,0,1],[0,1,1,0,0,0,1,0],[1,0,0,0,1,0,1,0],
         [0,0,0,1,0,1,0,0],[1,0,0,0,1,0,1,2],[0,0,0,0,2,0,0,0],
         [2,0,0,0,0,2,0,0],[0,1,1,0,0,0,1,2],[1,0,0,1,0,1,0,0],
         [0,0,0,1,0,1,0,2],[2,0,0,0,0,2,0,2],[0,0,0,0,2,0,0,2],
         [2,1,1,0,0,0,1,2],[2,0,0,0,0,2,2,2],[1,0,0,1,0,1,0,1],
         [1,0,0,1,0,1,1,0],[1,0,0,1,0,1,0,2],[2,0,0,1,0,1,0,2],
         [0,0,0,2,0,0,0,2],[2,0,0,0,2,0,0,2],[1,0,0,1,0,1,2,2],
         [0,1,1,0,1,0,2,2],[0,0,0,2,0,0,2,0],[2,1,1,0,1,1,0,1],
         [0,0,0,2,0,0,2,2],[2,1,1,0,1,0,2,2],[2,0,0,2,0,0,2,0],
         [2,0,0,2,0,0,2,2],[2,1,1,0,1,2,2,2],[2,0,0,2,0,2,0,2],
         [2,0,0,2,0,2,2,2],[2,2,2,0,2,0,2,2],[2,2,2,0,2,2,2,2],
         [2,2,2,2,2,2,2,2]]
  end
  return wdd
end

"""`weighted_dynkin_diagrams( <lie> )`

returns a tuple `wdd,bd` where `wdd` is the list of all weighted Dynkin 
diagrams for the simple Lie algebra <lie>, and `bd` are the corresponding
invariants `dim B_u`. The diagrams are ordered according to decreasing value
of `bd`; so the first diagram corresponds to the trivial nilpotent orbit,
the last one to the regular nilpotent orbit. The data are extracted from 
the descriptions in Section 13.1 of 

R. W. Carter, Finite groups of Lie type: conjugacy classes and
complex characters, Wiley, 1985.

```julia-repl
julia> g=LieAlg(:b,3)
#I dim = 21
LieAlg('B3')

julia> wdd=weighted_dynkin_diagrams(g)
(Array{Int64,1}[[0, 0, 0], [0, 1, 0], [0, 0, 2], [1, 0, 1], 
                           [0, 2, 0], [0, 2, 2], [2, 2, 2]], 
[9, 5, 4, 3, 2, 1, 0])
```
See also `generic_gram_wdd`, `gram_wdd_search`.
"""
function weighted_dynkin_diagrams(lie::LieAlg)
  typ=Symbol(lowercase(lie.dynkin[1]))
  wdd=[d[lie.permlabel] for d in wdd_data(typ,lie.rnk)]
  bd=Int[]
  for wd in wdd
    wt=[sum(r[s]*wd[s] for s in 1:lie.rnk) for r in lie.roots]
    l1=count(i->wt[i]==1,1:lie.N)
    lg2=count(i->wt[i]>=2,1:lie.N)
    push!(bd,lie.N-lg2-div(l1,2))
  end
  pp=sortperm(bd,rev=true)
  return wdd[pp],bd[pp]
end

"""`generic_gram_wdd( <lie>, <wdd> )`

returns a triple `gram`, `g1`, `g2` where `gr` is the generic Gram 
matrix associated with the weighted Dynkin diagram <wdd> of the Lie 
algebra <lie>; furthermore, `g1` and `g2` are the lists of positive 
roots which form a basis of `g(1)` and `g(2)`, respectively. The 
construction is explained in Remark 3.7 of

M. Geck, Generalised Gelfand-Graev representations in bad characteristic?,
Transf. Groups (2020); https://doi.org/10.1007/s00031-020-09575-3

The entries of the matrix `gram` are zeros or Julia Symbols involving
expressions `x1`, `x2`, etc. These can be specialised to actual
values (in a field, polynomial ring etc.) using the function 
`eval_gram_wdd`, which takes as input the output of `generic_gram_wdd`
and a list of values to which `x1`, `x2` etc. should be specialised. (We use
this construction in order to be self-contained within Julia.) Example:

```julia-repl
julia> l=LieAlg(:f,4);wdd=weighted_dynkin_diagrams(l)
#I dim = 52
(Array{Int64,1}}
 [0, 0, 0, 0]
 [1, 0, 0, 0]
 [0, 0, 0, 1]
 [0, 1, 0, 0]
 [2, 0, 0, 0]
 [0, 0, 0, 2]
 [0, 0, 1, 0]
 [2, 0, 0, 1]
 [0, 1, 0, 1]
 [1, 0, 1, 0]
 [0, 2, 0, 0]
 [2, 2, 0, 0]
 [1, 0, 1, 2]
 [0, 2, 0, 2]
 [2, 2, 0, 2]
 [2, 2, 2, 2]],
[24, 16, 13, 10, 9, 9, 7, 6, 6, 5, 4, 3, 3, 2, 1, 0])

julia> wd=generic_gram_wdd(l,wdd[1][9])  
(Any[0 0 0 0 :(-1x1) 0 0 :(-1x4); 
     0 0 0 :(-1x1) 0 :(-1x2) :(-1x3) :(-1x5); 
     0 0 0 0 :(-1x2) 0 :(1x4) 0; 
     0 :(1x1) 0 0 :(-1x3) :(2x4) 0 0; 
     :(1x1) 0 :(1x2) :(1x3) 0 :(1x5) 0 0; 
     0 :(1x2) 0 :(-2x4) :(-1x5) 0 0 0; 
     0 :(1x3) :(-1x4) 0 0 0 0 0; 
     :(1x4) :(1x5) 0 0 0 0 0 0], 
[2, 4, 5, 6, 7, 8, 9, 11], [10, 12, 13, 14, 15])
```
Now let us specialise the symbols `x1`, `x2`, etc. to actual
indeterminates over the rationals. For this purpose, we need Nemo.

```julia-repl
julia> using Nemo
julia> R,x=polynomial_ring(QQ,["x"*string(i) for i in 1:5])
julia> wd1=eval_gram_wdd(wd,x)
 0   0   0    0      -x1  0     0    -x4
 0   0   0    -x1    0    -x2   -x3  -x5
 0   0   0    0      -x2  0     x4   0  
 0   x1  0    0      -x3  2*x4  0    0  
 x1  0   x2   x3     0    x5    0    0  
 0   x2  0    -2*x4  -x5  0     0    0  
 0   x3  -x4  0      0    0     0    0  
 x4  x5  0    0      0    0     0    0  
```
For applications (see the above article), we need to know when the
determinant of this Gram matrix is non-zero. Since it is skew-symmetric, 
we can use the `pfaffian`:
```julia-repl
julia> pfaffian(wd1)
-3*x1*x4^2*x5 + 3*x2*x3*x4^2
```
Here, we see that the determinant will only become zero when we reduce 
the coefficients modulo 3. However, for larger matrices, the function 
`pfaffian` may not be efficient enough. In that case, one can use 
SINGULAR and the function `syz` (applied to the module generated by the 
row vectors of the above matrix).
```julia-repl
julia> import Singular
julia> function prop510(lie,d,pr)
         gr=generic_gram_wdd(lie,d)
         n,m=length(gr[2]),length(gr[3])
         str=prod([string(i) for i in d])
         print("#I ",str,", dim g(1), g(2) = ",n,", ",m," -> syz ")
         if n==0 || m==0    # trivial cases
           println("= 0")
           return gr
         end
         xi=["x"*string(i) for i in 1:m]
         F=Singular.Fp(pr)              # field Z/pZ
         R,x=Singular.polynomial_ring(F,xi; ordering=:lex)
         gr1=eval_gram_wdd(gr,x)
         vecs=[Singular.vector(R,[gr1[i,j] for j in 1:n]...) for i in 1:n]
         sy=Singular.syz(Singular.Module(R,vecs...))
         if Singular.iszero(sy)==true
           println("= 0")
         else
           println("not 0 (",Singular.ngens(sy)," gens)")
         end
         return sy
       end
julia> prop510(l,wdd[1][9],2)       # work in characteristic 2
#I [0, 1, 0, 1], dim g(1) = 8 -> syz = 0
Singular Module over Singular Polynomial Ring (ZZ/2),(x1,x2,x3,x4,x5),
(lp(5),c,L(1048575)), with Generators:
0

julia> prop510(l,wdd[1][9],3)       # work in characteristic 3
#I [0, 1, 0, 1], dim g(1) = 8 -> syz not 0 
Singular Module over Singular Polynomial Ring (ZZ/3),(x1,x2,x3,x4,x5),
(lp(5),c,L(1048575)), with Generators:
x1*gen(8)-x2*gen(7)+x3*gen(6)-x4*gen(5)-x5*gen(4)
x1*gen(6)-x2*gen(4)+x3*gen(3)+x4*gen(2)-x5*gen(1)
```
Thus, since there are non-zero relations in the latter case, the matrix 
`wd1` has determinant zero over the polynomial ring with coefficients in 
the field with 3 elements. (This method was a crucial ingredient in the 
proof of Proposition 5.10 of the above paper, and was suggested by A. Steel 
and U. Thiel.)

See also `weighted_dynkin_diagrams` and `gram_wdd_search`.
"""
function generic_gram_wdd(lie::LieAlg,wdd::Array{Int,1})
  str=structconsts(lie)
  wt=[sum(r[s]*wdd[s] for s in 1:lie.rnk) for r in lie.roots]
  l1=[i for i in 1:lie.N if wt[i]==1]
  l2=[i for i in 1:lie.N if wt[i]==2]
  xi=[Symbol("x"*string(i)) for i in 1:length(l2)]
  mat=[]
  for i in l1
    row=[]
    for j in l1
      p=findfirst(==(lie.roots[i]+lie.roots[j]),lie.roots)
      if p==nothing || wt[p]!=2
        push!(row,0)
      else
        x=xi[findfirst(==(p),l2)]
        push!(row,:($(str[(i,j)][1])*$x))
      end
    end
    push!(mat,row)
  end
  return [mat[i][j] for i in 1:length(l1),j in 1:length(l1)],l1,l2
end

# wd is output von generic_gram_wdd
function eval_gram_wdd(wd,l)
  if length(wd[3])==0 
    return wd[1]
  end
  n=size(wd[1])[1]
  null=0*l[1]
  for i in 1:length(l)
    s=Symbol("x",i)
    xx=l[i]
    eval(:($s=$xx))
  end
  a=Array{typeof(l[1])}(undef,n,n)
  for i in 1:n 
    for j in 1:n 
      if wd[1][i,j]==0
        a[i,j]=null
      else
        a[i,j]=eval(wd[1][i,j])
      end
    end
  end
  return a
end 

# this function is more efficient if many calls to the gram matrix required 
# (it is used in gram_wdd_search).
function gram_wdd(lie::LieAlg,wdd::Array{Int,1},pr::Int,xi1=[])
  str=structconsts(lie)
  wt=[sum(r[s]*wdd[s] for s in 1:lie.rnk) for r in lie.roots]
  l1=[i for i in 1:lie.N if wt[i]==1]
  l2=[i for i in 1:lie.N if wt[i]==2]
  n=length(l2)
  if pr==1
    if n>0 && length(xi1)==0
      R,xi=Nemo.polynomial_ring(Nemo.ZZ,["x"*string(i) for i in 1:n]);
      null=0*xi[1]
    else
      R,xi=Nemo.QQ,xi1
      null=0//1
    end
  else
    F,t=Nemo.finite_field(pr,1,"t")
    if n>0 && length(xi1)==0
      R,xi=Nemo.polynomial_ring(F,["x"*string(i) for i in 1:n]);
      null=0*xi[1]
    else
      R,xi=F,xi1
      null=0*t
    end
  end
  mat=Array{typeof(null),1}[]
  for i in l1
    row=typeof(null)[]
    for j in l1
      p=findfirst(==(lie.roots[i]+lie.roots[j]),lie.roots)
      if p==nothing || wt[p]!=2
        push!(row,null)
      else
        x=xi[findfirst(==(p),l2)]
        push!(row,str[(i,j)][1]*x)
      end
    end
    push!(mat,row)
  end
  #return matrix(R,[mat[i][j] for i in 1:length(l1),j in 1:length(l1)]),l1,l2
  return [mat[i][j] for i in 1:length(l1),j in 1:length(l1)],l1,l2
end

"""`gram_wdd_search( <lie>, <wdd>, <pr>, <max> )`

searches for a 0,1 vector such that, for the linear map lambda
on g(2) with values given by that vector, the corresponding 
alternating form on g(1) has non-zero determinant (if <pr> is
a prime) or determinant 1,-1 (if <pr>=1).

The search starts with the vector 1,1,...1, and then increases the number 
of zeroes step by step (k=0,1,2,... corresponds to the number of zeroes). 
If a vector with the desired properties exists at all, then experiments 
show that it should be found after a very small number of steps; usually, 
four steps are sufficient. One can set the maximum number of `k` to be 
considered by an optional argument `max`. (Internally, some other default 
values are set.)

For further explanations, see Example 4.7 and the proof of 
Corollary 5.11 in 

M. Geck, Generalised Gelfand-Graev representations in bad characteristic?, 
Transf. Groups (2020); https://doi.org/10.1007/s00031-020-09575-3

For example, the computations for type `F_4` can be verified as
follows.

```julia-repl
julia> l=LieAlg(:f,4);wdd=weighted_dynkin_diagrams(l)[1]
#I dim = 52
16-element Array{Array{Int64,1},1}:
 [0, 0, 0, 0]
 [1, 0, 0, 0]
 [0, 0, 0, 1]
 [0, 1, 0, 0]
 [2, 0, 0, 0]
 [0, 0, 0, 2]
 [0, 0, 1, 0]
 [2, 0, 0, 1]
 [0, 1, 0, 1]
 [1, 0, 1, 0]
 [0, 2, 0, 0]
 [2, 2, 0, 0]
 [1, 0, 1, 2]
 [0, 2, 0, 2]
 [2, 2, 0, 2]
 [2, 2, 2, 2]

julia> gram_wdd_search(l,wdd[9],1)
#I [0, 1, 0, 1]
#I prime=1, dim g(1)=8, dim g(2)=5, k = 0 1 2 3 4 5 
# ---> orbit not in Delta bullet for prime 1
false

julia> gram_wdd_search(l,wdd[9],2)
#I [0, 1, 0, 1]
#I prime=2, dim g(1)=8, dim g(2)=5, k = 0 1 
5-element Array{Int64,1}:
 0
 1
 1
 1
 1

julia> gram_wdd_search(l,wdd[9],3)
#I [0, 1, 0, 1]
#I prime=3, dim g(1)=8, dim g(2)=5, k = 0 1 2 3 4 5 
# ---> orbit not in Delta bullet for prime 3
false
```
See also `generic_gram_wdd`.
"""
function gram_wdd_search(lie::LieAlg,wdd::Array{Int,1},pr::Int,max::Int=0)
  cb=structconsts(lie)
  wt=[sum(r[s]*wdd[s] for s in 1:lie.rnk) for r in lie.roots]
  n1=length([i for i in 1:lie.N if wt[i]==1])
  n=length([i for i in 1:lie.N if wt[i]==2])
  println("#I ",wdd)
  print("#I prime=",pr,", dim g(1)=",n1,", dim g(2)=",n)
  flush(stdout)
  if pr==1
    R,t=Nemo.QQ,1//1
  else
    R,t=Nemo.finite_field(pr,1,"t")
  end  
  if n1==0 
    println("")
    xi=[t^0 for i in 1:(n+1)]
    println("# ---> search successful for characterstic exponent ",pr)
    return xi[1:n]
  end
  if max==0
    if n<=10
      max=n
    elseif n<=30
      print(", max(k)=4")
      max=4
    elseif n<=33
      print(", max(k)=3")
      max=3
    else
      print(", max(k)=2")
      max=2
    end
  end
  print(", k = ")
  flush(stdout)
  #gg=generic_gram_wdd(lie,wdd)
  for k in 0:max
    print(k," ")
    flush(stdout)
    for c in combinations(n,k)
      xi=[t^0 for i in 1:n]
      for i in c
        xi[i]=0*t
      end
      g=det_field(gram_wdd(lie,wdd,pr,xi)[1])
      #g=det_field(eval_gram_wdd(gg,xi))
      if pr==1
        if g==1 || g==-1 
          println("")
          println("# ---> search successful for characterstic exponent ",pr)
          return xi
        end
      else
        if g!=0*t
          println("")
          println("# ---> search successful for characterstic exponent ",pr)
          return xi
        end
      end
    end
  end
  println("")
  #println("# ---> orbit not in Delta bullet for characteristic exponent ",pr)
  return false
end

##########################################################################
##
#I Section 4: Constructions with Chevalley group elements
##
"""`jordanblocks( <mat> )`

returns the partition describing the Jordan normal form of a 
unipotent matrix <mat>.
"""
function jordanblocks(mat)
  n=size(mat)[1]
  m1=[mat[i,j] for i in 1:n,j in 1:n]
  a=0*m1
  for i in 1:n
    a[i,i]=mat[1,1]^0
    m1[i,i]-=mat[1,1]^0
  end
  r=[n]
  for i in 1:(n-1)
    if r[i]>0
      a=a*m1
      push!(r,rankmat(a)[1])
    else
      push!(r,0)
    end
  end
  for i in 1:n
    for j in 2:i
      r[n+1-i]-=j*r[n-i+j]
    end
  end
  bl=Array{Int,1}[]
  for i in 1:n
    if r[i]!=0
      push!(bl,[i,r[i]])
    end
  end
  return bl[end:-1:1]
end

# order of Nemo matrix of finite order
function ordermat(mat)
  idm=mat^0
  if mat==idm 
    return 1
  end
  i=2
  l1=mat*mat
  while l1!=idm
    l1=l1*mat
    i+=1
  end
  return i
end
  
#function ordermat(mat)
#  if isidmat(mat)
#    return 1
#  end
#  i=2
#  l1=mat*mat
#  while isidmat(l1)==false
#    l1=l1*mat
#    i+=1
#  end
#  return i
#end
  
"""`jordandecelt( <mat>, <p> )`

returns the Jordan decomposition of a (Nemo) matrix <mat>
with entries in a finite field of characteristic <p>.
"""
function jordandecelt(mat1,p::Int)
  l=size(mat1)[1]
  mat=Nemo.matrix(parent(mat1[1,1]),mat1)
  idm=mat^0
  o=ordermat(mat)
  i=1
  p1=p
  while (o % p1)==0
    i+=1
    p1=p1*p
  end
  p1=div(p1,p)
  m=div(o,p1)
  if p1==1
    s=mat
    u=idm
  else
    g=gcdx(m,p1)
    e=(g[3]*p1) % o
    if e<0
      e+=o
    end
    if e==0
      s=idm
    else
      s=mat^e
    end
    e=(g[2]*m) % o
    if e<0
      e+=o
    end
    if e==0
      u=idm
    else
      u=mat^e
    end
  end
  println("#I o(s),o(u) = ",m,",",p1)
  return s,u
end
  
"""`cross_regular( <l>, <t> )`

returns the elements in Steinberg's cross section of the conjugacy
classes of regular elements in the Chevalley group of simply-connected
type corresponding to the Lie algebra <l> (as returned by `LieAlg`) over a 
finite field, specified by a generator <t> of the multiplicative group.
The elements of that cross section are defined in Theorem 1.4 of 

R. Steinberg, Regular elements in semisimple algebraic groups, Publ.
Math. IHES 25 (1965), 49--80.

(If you just want one regular element corresponding to a tuple of
length <l.rank> of elements from a field or a ring, use 
`cross_regular1(<l>,<tup>)`; see below for an example.)

Internally, the function works with Julia SparseArrays, but
the resulting matrices are unpacked as 2D arrays. There is a 
third optional argument that can be set to `true`: in that case, 
the matrices are returned in sparse format.

The function `cross_regular` requires the Julia package Nemo (see 
http://nemocas.org) in order to work with finite fields. Example:
```julia-repl
julia> l=LieAlg(:g,2);
#I dim = 14

julia> using Nemo
julia> R,x=finite_field(2,2,"x")
(Finite field of degree 2 over F_2, x)

julia> jd=[jordandecelt(a,2) for a in cross_regular(l,x)];
#I starting loop over 16 tuples
#I o(s),o(u) = 3,2
#I o(s),o(u) = 5,2
#I o(s),o(u) = 5,2
#I o(s),o(u) = 1,8
#I o(s),o(u) = 13,1
#I o(s),o(u) = 21,1
#I o(s),o(u) = 5,2
#I o(s),o(u) = 15,1
#I o(s),o(u) = 13,1
#I o(s),o(u) = 5,2
#I o(s),o(u) = 21,1
#I o(s),o(u) = 15,1
#I o(s),o(u) = 3,4
#I o(s),o(u) = 15,1
#I o(s),o(u) = 15,1
#I o(s),o(u) = 7,1
```
The function `jordandecelt(<a>,<p>)` computes the Jordan decomposition
of a matrix <a> with entries in a finite field of characteristic <p>. 
It returns a tuple `s,u` where `s` is the semisimple part of <a> and `u` is 
the unipotent part of <a>. Note that the semisimple parts of the
elements in Steinberg's cross section form a set of representatives
for the semisimple conjugacy classes. Finally, we obtain the Jordan normal 
form of each unipotent part:
```julia-repl
[jordanblocks(a[2]) for a in jd]
16-element Array{Array{Any,1},1}:
 [[2, 6], [1, 2]]
 [[2, 6], [1, 2]]
 [[2, 6], [1, 2]]
 [[8, 1], [6, 1]]
 [[1, 14]]       
 [[1, 14]]       
 [[2, 6], [1, 2]]
 [[1, 14]]       
 [[1, 14]]       
 [[2, 6], [1, 2]]
 [[1, 14]]       
 [[1, 14]]       
 [[4, 2], [3, 2]]
 [[1, 14]]       
 [[1, 14]]       
 [[1, 14]]       
```
(Thus, the first `u` has 6 blocks of size 2 and 2 blocks of size 1, etc.
The matrix with `o(s)=1` and `o(u)=8` is regular unipotent; the above 
output shows that it has one Jordan block of size 8 and one of size 6.)

As mentioned above, we can produce a single member of Steinberg's
cross section using the function `cross_regular1(<lie>,<tup>)`.
Here, we may even take elements from a ring. 
```julia-repl
julia> l=LieAlg(:b,2)
#I dim = 10
LieAlg('B2')

julia> cross_regular1(l,[7,13])
#I dim = 4
4×4 Array{Int64,2}:
 -7  -13  1  0
 -1    0  0  0
  0    7  0  1
  0    1  0  0
```
See also `rep_minuscule`, `monomialelts`. 
"""
function cross_regular(lie::LieAlg,t,spr::Bool=false)
  f=[0*t,t]
  t1=t
  while t1!=t^0
    t1*=t
    push!(f,t1)
  end
  n=length(f)
  r=rep_sc(lie)
  # same as in monomialelts but with sparse matrices
  gens1=[sparse(expliemat(r[1][s],t^0)) for s in 1:lie.rnk]
  gens2=[sparse(expliemat(r[2][s],-t^0)) for s in 1:lie.rnk]
  m=[gens1[s]*gens2[s]*gens1[s] for s in 1:lie.rnk]
  print("#I ");
  flush(stdout)
  gens=[[sparse(expliemat(r[1][i],j))*m[i] for j in f] for i in 1:lie.rnk]
  d=size(m[1])[1]
  cr=SparseMatrixCSC{typeof(t),Int}[]
  tups=Array{typeof(t),1}[]
  print("starting loop over ",n^lie.rnk," tuples (100/.) ")
  flush(stdout)
  z=0
  for c in Iterators.product([1:n for i in 1:lie.rnk]...)
    z+=1
    if (z % 100)==0
      print(".")
      flush(stdout)
    end
    x=gens[1][c[1]]
    for i in 2:lie.rnk
      x*=gens[i][c[i]]
    end
    push!(cr,x)
    push!(tups,[f[i] for i in c])
  end
  println("")
  if spr==true
    return cr
  else
    return unpacksparse.(cr)
  end
end

function cross_regular_old(lie::LieAlg,t)
  R=parent(t)
  f=[0*t,t]
  t1=t
  while t1!=t^0
    t1*=t
    push!(f,t1)
  end
  n=length(f)
  r=rep_sc(lie)
  # same as in monomialelts but with Nemo matrices
  gens1=[Nemo.matrix(R,expliemat(r[1][s],t^0)) for s in 1:lie.rnk]
  gens2=[Nemo.matrix(R,expliemat(r[2][s],-t^0)) for s in 1:lie.rnk]
  m=[gens1[s]*gens2[s]*gens1[s] for s in 1:lie.rnk]
  print("#I ");
  gens=[[Nemo.matrix(R,expliemat(r[1][i],j))*m[i] for j in f] 
                                              for i in 1:lie.rnk]
  d=size(m[1])[1]
  cr,tups=[],[]
  print("starting loop over ",n^lie.rnk," tuples ")
  z=0
  for c in Iterators.product([1:n for i in 1:lie.rnk]...)
    z+=1
    if (z % 100)==0
      print(".")
    end
    x=gens[1][c[1]]
    for i in 2:lie.rnk
      x*=gens[i][c[i]]
    end
    push!(cr,x)
    push!(tups,[f[i] for i in c])
  end
  println("")
  return cr
end

# just for one tuple of values from a field, or even a ring R
function cross_regular1(lie::LieAlg,tup)
  r=rep_sc(lie)
  gens1=[sparse(expliemat(r[1][s],tup[1]^0)) for s in 1:lie.rnk]
  gens2=[sparse(expliemat(r[2][s],-tup[1]^0)) for s in 1:lie.rnk]
  m=[gens1[s]*gens2[s]*gens1[s] for s in 1:lie.rnk]
  x=sparse(expliemat(r[1][1],tup[1]))*m[1]
  for i in 2:lie.rnk
    x*=sparse(expliemat(r[1][i],tup[i]))*m[i]
  end
  return unpacksparse(x)
end

# reduce all elements in integer matrix mod p
# requires Jean's Mod
#function matmodp(mat,p)
#  return [Mod{p}(mat[i,j]) for i in 1:size(mat)[1],j in 1:size(mat)[2]]
#end
 
# helper functions for borelcoets
function helpborel(lie::LieAlg,u0,maxl)
  n0=[lie.roots[x[1]] for x in u0]
  n00=[[a for a in lie.roots[1:lie.N] 
             if a!=r && all(i->r[i]<=a[i],1:lie.rnk)] for r in n0]
  res=Int[]
  for i in 1:length(n0) 
    if all(j->j==i || !(n0[i] in n00[j]),1:length(n0))
      push!(res,u0[i][1])
    end
  end
  return reflsubgrp(lie,res,maxl)[3]
end

# find all variables involved in a list of integer polynomials
function varsinv(fs,xn,pr)
  v=Int[]
  es=Array{Int,1}[]
  cs=[]
  for f in fs 
    append!(es,collect(Nemo.exponent_vectors(f)))
    append!(cs,collect(Nemo.coeffs(f)))
  end
  for i in 1:xn
    if any(j->es[j][i]>0 && (cs[j]%pr)!=0,1:length(es))
      push!(v,i)
    end
  end
  return v
end

# version for singular
function varsinv_sing(fs,xn)
  v=Int[]
  es=Array{Int,1}[]
  cs=[]
  for f in fs 
    append!(es,collect(exponent_vectors(f)))
    append!(cs,collect(coeffs(f)))
  end
  for i in 1:xn
    if any(j->es[j][i]>0,1:length(es))
      push!(v,i)
    end
  end
  return v
end

function myevaluate(f,c1)
  v=0
  for z in zip(Singular.exponent_vectors(f),Singular.coeffs(f))
    x=1
    for j in 1:length(c1)
      if z[1][j]>0
        x*=c1[j]^z[1][j]
      end
    end
    v+=x*z[2]
  end
  return v
end

# convert a singular polynomial into an integer Nemo polynomial
# f=Singular pol over Fp(pr), R=Nemo polynomial ring over ZZ
# (Need this for the time being because there is a bug in Singular.evaluate)
function pol_singtonemo(R,f,pr)
  e=collect(Singular.exponent_vectors(f))
  c=collect(Singular.coeffs(f))
  K=Singular.Fp(pr)
  if pr==2
    f=[0,1]
  else
    f=[0]
    for i in 1:div(pr-1,2)
      push!(f,i)
      push!(f,-i)
    end
  end
  g=Nemo.MPolyBuildCtx(R)
  for i in 1:length(e)
    n=1
    while K(f[n])!=c[i]
      n+=1
    end
    Nemo.push_term!(g,Nemo.ZZ(f[n]),e[i])
  end
  Nemo.finish(g)
end

function oldpol_singtonemo(f,xn,pr)
  e=collect(Singular.exponent_vectors(f))
  c=collect(Singular.coeffs(f))
  K=Singular.Fp(pr)
  if pr==2
    f=[0,1]
  elseif pr==3
    f=[0,1,-1]
  elseif pr==5
    f=[0,1,-1,2,-2]
  else
    f=collect(0:pr-1)
  end
  g=0*xn[1]
  for i in 1:length(e)
    n=1
    while K(f[n])!=c[i]
      n+=1
    end
    x=f[n]*xn[1]^0
    for j in 1:length(xn)
      if e[i][j]>0
        x*=xn[j]^e[i][j]
      end
    end
    g+=x
  end
  return g
end

# ... and in the other direction, here yn=Singular variables.
function pol_nemotosing(f,yn)
  e=collect(Nemo.exponent_vectors(f))
  c=collect(Nemo.coeffs(f))
  g=0*yn[1]
  for i in 1:length(e)
    x=c[i]*yn[1]^0
    for j in 1:length(yn)
      if e[i][j]>0
        x*=yn[j]^e[i][j]
      end
    end
    g+=x
  end
  return g
end

# create singular polynomial in variabales yn
function singularpol(yn,coeff,expon)
  g=0*yn[1]
  for i in 1:length(coeff)
    x=coeff[i]*yn[1]^0
    for j in 1:length(yn)
      if expon[i][j]>0
        x*=yn[j]^expon[i][j]
      end
    end
    g+=x
  end
  return g
end

# simplify polynomial system
# reduce coefficients mod p
function simplifypol1(R,pol,pr)
  coeff=collect(Nemo.coeffs(pol))
  expon=collect(Nemo.exponent_vectors(pol))
  f=Nemo.MPolyBuildCtx(R)
  for i in 1:length(coeff)
    p=coeff[i]%pr
    if p!=0
      Nemo.push_term!(f,Nemo.ZZ(p),expon[i])
    end
  end
  Nemo.finish(f)
end

# reduce exponents mod (q-1)
function simplifypol2(R,pol,pr,ffe)
  q=pr^ffe
  coeff=collect(Nemo.coeffs(pol))
  expon=collect(Nemo.exponent_vectors(pol))
  f=Nemo.MPolyBuildCtx(R)
  for i in 1:length(coeff)
    if coeff[i]%pr!=0
      e=expon[i]
      for j in 1:length(e)
        ee=e[j]
        while ee>=q
          ee+=1-q
        end
        e[j]=ee
      end
      Nemo.push_term!(f,Nemo.ZZ(coeff[i]),e[:])
    end
  end
  simplifypol1(R,Nemo.finish(f),pr)
end

# look for isolated variables and replace
function simplifypols(R,xn,pols,pr,ffe)
  if pr==2
    npols=[simplifypol2(R,pol,pr,ffe) for pol in pols]
    return [pol for pol in npols if pol!=0],0
  end
  npols=[pol for pol in pols if pol!=0]
  piv=0
  for f in 1:length(npols)
    e=collect(Nemo.exponent_vectors(npols[f]))
    c=collect(Nemo.coeffs(npols[f]))
    weiter=true
    i0=length(c)
    while weiter && i0>=1
      ee,cc=e[i0],c[i0]
      if sum(ee)==1 && (cc%pr)!=0
        i=1
        while ee[i]==0
          i+=1
        end
        el=[e[j] for j in 1:length(c) if j!=i0]
        if all(x->x[i]==0,el)
          weiter=false
          piv+=1
          if (cc==1 || cc==-1)
            c1=cc
          else
            if cc<0
              cc+=pr
            end
            c1=1
            while ((c1*cc)%pr)!=1
              c1+=1
            end
          end
          v=[xi for xi in xn]
          v[i]=xn[i]-c1*npols[f]
          npols=[simplifypol2(R,
                  Nemo.evaluate(pol,v),pr,ffe) for pol in npols]
        end
      end
      i0+=-1
    end
  end
  return [pol for pol in npols if pol!=0],piv
end

# for singular polynomials
function simplifypol2sing(yn,pol,pr)
  coeff=collect(Singular.coeffs(pol))
  expon=collect(Singular.exponent_vectors(pol))
  g=0*yn[1]
  for i in 1:length(coeff)
    e=expon[i]
    for j in 1:length(e)
      ee=e[j]
      while ee>=pr
        ee+=1-pr
      end
      e[j]=ee
    end
    x=coeff[i]*yn[1]^0
    for j in 1:length(yn)
      if e[j]>0
        x*=yn[j]^e[j]
      end
    end
    g+=x
  end
  return g
end

# look for isolated variables and replace
function simplifypolssing(R,xn,pols,pr)
  if pr==2
    npols=[simplifypol2sing(R,pol,pr) for pol in pols]
    return [pol for pol in npols if pol!=0],0
  end
  npols=[pol for pol in pols]
  piv=0
  for f in 1:length(npols)
    e=collect(Singular.exponent_vectors(npols[f]))
    c=collect(Singular.coeffs(npols[f]))
    weiter=true
    i0=length(c)
    while weiter && i0>=1
      ee,cc=e[i0],c[i0]
      if sum(ee)==1 && (cc%pr)!=0
        i=1
        while ee[i]==0
          i+=1
        end
        el=[e[j] for j in 1:length(c) if j!=i0]
        if all(x->x[i]==0,el)
          weiter=false
          piv+=1
          if (cc==1 || cc==-1)
            c1=cc
          else
            if cc<0
              cc+=pr
            end
            c1=1
            while ((c1*cc)%pr)!=1
              c1+=1
            end
          end
          v=[xi for xi in xn]
          v[i]=xn[i]-c1*npols[f]
          npols=[simplifypol2(R,Singular.evaluate(pol,v),pr) for pol in npols]
        end
      end
      i0+=-1
    end
  end
  return [pol for pol in npols if pol!=0],piv
end

"""`gfp_points( <R>, <xs>, <pols>, <pr> )`

returns the number of rational points of a system of polynomial equations
over the prime field with <p> elements; here, <pols> is a list of 
polynomials in the (Nemo) polynomial ring <R> in indeterminates <xs> 
over the integers. Example:

```julia-repl
julia> using Nemo
julia> R,x=polynomial_ring(ZZ,["x1","x2","x3"])
julia> fs=[-x[1]*x[3],-x[1]-x[3]+1,-x[2]]
3-element Array{fmpz_mpoly,1}:
 -x1*x3  
 -x1-x3+1
 -x2     
julia> gfp_points(R,x,fs)
2
```
(The two solutions are `[1,0,0]` and `[0,0,1]`.) There is also a version
of this function for general finite fields, called `fq_points`, where
there is an additional fifth argument <ffe> specifying the degree
of the field extension over the prime field.
"""
function gfp_points(R,xs,pols,pr)
  if length(pols)==0
    return pr^length(xs)
  end
  if length(pols)==1 && pols[1]==1
    return -1
  end
  npols,pivot=simplifypols(R,xs,pols,pr,1)
  if length(npols)==0
    return pr^(length(xs)-pivot)
  end
  if length(npols)==1 && npols[1]==1
    return -1
  end
  if pr==2
    ff=Int8[0,1]
  else
    ff=Int8[0]
    for i in 1:div(pr-1,2)
      push!(ff,i)
      push!(ff,-i)
    end
  end
  vv=[varsinv([f],length(xs),pr) for f in npols]
  p=sortperm([length(v) for v in vv])
  vp,fs=vv[p],npols[p] # pols ordered according to numbers of variables
  if length(vp[1])==0
    return -1          # one of the polynomials is a constant
  end
  sols=Array{Int8,1}[]
  c0=zeros(Int8,length(xs))  # consider first polynomial 
  for c in Iterators.product([ff for j in 1:length(vp[1])]...)
    for k in 1:length(vp[1])
      c0[vp[1][k]]=c[k]
    end
    if (Nemo.evaluate(fs[1],c0)%pr)==0
      push!(sols,c0[:])
    end
  end
  if length(sols)==0
    return 0
  end
  xn=vp[1][:]
  for i in 2:length(fs)
    vn=[k for k in vp[i] if !(k in xn)]
    if length(vn)==0
      sols=[c for c in sols if (Nemo.evaluate(fs[i],c)%pr)==0]
      if length(sols)==0
        return 0
      end
    else
      soln=Array{Int8,1}[]
      itrv=Iterators.product([ff for j in 1:length(vn)]...)
      for c0 in sols
        for c in itrv
          for k in 1:length(vn)
            c0[vn[k]]=c[k]
          end
          if (Nemo.evaluate(fs[i],c0)%pr)==0
            push!(soln,c0[:])
          end
        end
      end
      if length(soln)==0
        return 0
      end
      sols=soln
      append!(xn,vn)
    end
  end
  if length(xn)<length(xs)
    return length(sols)*pr^(length(xs)-length(xn)-pivot)
  else
    return length(sols)
  end
end

# old version
function gfp_points1(pols,xs,pr,groebner=false)
  if length(pols)==0
    return pr^length(xs)
  end
  if pr==2
    ff=Int8[0,1]
  elseif pr==3
    ff=Int8[0,1,-1]
  elseif pr==5
    ff=Int8[0,1,-1,2,-2]
  else
    ff=Int8[i-1 for i in 1:pr]
  end
  vv=[varsinv([f],length(xs),pr) for f in pols]
  if groebner==false
    p=sortperm([length(v) for v in vv])
    vp,fs=vv[p],pols[p] # pols ordered according to numbers of variables
  else
    vp,fs=vv,pols 
  end
  if length(vp)==0 || length(vp[1])==0
    return -1          # one of the polynomials is a constant
  end
  sols=Array{Int8,1}[]   
  c0=zeros(Int8,length(xs))  # consider first variable set only
  i2=2
  while i2<=length(fs) && all(k->k in vp[1],vp[i2])
    i2+=1
  end
  for c in Iterators.product([ff for j in 1:length(vp[1])]...)
    for k in 1:length(vp[1])
      c0[vp[1][k]]=c[k]
    end
    if all(i->Nemo.evaluate(fs[i],c0)%pr==0,1:i2-1)
      push!(sols,c0[:])
    end
  end
  if length(sols)==0
    return 0
  end
  xn=vp[1][:]
  i1=i2
  #print("sols:")
  while i1<=length(fs)
    vn=[k for k in vp[i1] if !(k in xn)]
    i2=i1+1
    while i2<=length(fs) && all(k->k in xn || k in vn,vp[i2])
      i2+=1 
    end
    soln=Array{Int8,1}[]
    itrv=collect(Iterators.product([ff for j in 1:length(vn)]...))
    for c0 in sols
      for c in itrv
        for k in 1:length(vn)
          c0[vn[k]]=c[k]
        end
        if all(i->Nemo.evaluate(fs[i],c0)%pr==0,i1:i2-1)
          push!(soln,c0[:])
        end
      end
    end
    if length(soln)==0
      return 0
    end
    sols=soln
    #print("(",length(vn),",",length(sols),")");
    flush(stdout)
    append!(xn,vn)
    i1=i2
  end
  if length(xn)<length(xs) 
    return length(sols)*pr^(length(xs)-length(xn))
  else
    return length(sols)
  end
end        

# for general finite fields
function fq_points(R,xs,pols,pr,ffe)
  q=pr^ffe
  if length(pols)==0
    return q^length(xs)
  end
  if length(pols)==1 && pols[1]==1
    return -1
  end
  npols,pivot=simplifypols(R,xs,pols,pr,ffe)
  if length(npols)==0
    return q^(length(xs)-pivot)
  end
  if length(npols)==1 && npols[1]==1
    return -1
  end
  K,t=Nemo.finite_field(pr,ffe,"t")
  null=0*t
  ff=[0*t]
  for i in 1:(q-1)
    push!(ff,t^i)
  end
  vv=[varsinv([f],length(xs),pr) for f in npols]
  p=sortperm([length(v) for v in vv])
  vp,fs=vv[p],npols[p] # pols ordered according to numbers of variables
  if length(vp[1])==0
    return -1          # one of the polynomials is a constant
  end
  sols=[]
  c0=[0*t for i in 1:length(xs)]  # consider first polynomial 
  for c in Iterators.product([ff for j in 1:length(vp[1])]...)
    for k in 1:length(vp[1])
      c0[vp[1][k]]=c[k]
    end
    if Nemo.evaluate(fs[1],c0)==null
      push!(sols,c0[:])
    end
  end
  if length(sols)==0
    return 0
  end
  xn=vp[1][:]
  for i in 2:length(fs)
    vn=[k for k in vp[i] if !(k in xn)]
    if length(vn)==0
      sols=[c for c in sols if Nemo.evaluate(fs[i],c)==null]
      if length(sols)==0
        return 0
      end
    else
      soln=[]
      itrv=Iterators.product([ff for j in 1:length(vn)]...)
      for c0 in sols
        for c in itrv
          for k in 1:length(vn)
            c0[vn[k]]=c[k]
          end
          if Nemo.evaluate(fs[i],c0)==null
            push!(soln,c0[:])
          end
        end
      end
      if length(soln)==0
        return 0
      end
      sols=soln
      append!(xn,vn)
    end
  end
  if length(xn)<length(xs)
    return length(sols)*q^(length(xs)-length(xn)-pivot)
  else
    return length(sols)
  end
end

function ogfp_points(pols,xs,pr,groebner=false)
  if length(pols)==0
    return pr^length(xs)
  end
  if pr==2
    ff=Int8[0,1]
  else
    ff=Int8[0]
    for i in 1:div(pr-1,2)
      push!(ff,i)
      push!(ff,-i)
    end
  end
  vv=[varsinv([f],length(xs),pr) for f in pols]
  if groebner==false
    p=sortperm([length(v) for v in vv])
    vp,fs=vv[p],pols[p] # polynomials ordered according to numbers of variables
  else
    vp,fs=vv,pols 
  end
  if length(vp)==0 || length(vp[1])==0
    return -1          # one of the polynomials is a constant
  end
  sols=Array{Int8,1}[]   
  c1=zeros(Int8,length(xs))  # consider first variable set only
  i2=2
  while i2<=length(fs) && all(k->k in vp[1],vp[i2])
    i2+=1
  end
  for c in Iterators.product([ff for j in 1:length(vp[1])]...)
    for k in 1:length(vp[1])
      c1[vp[1][k]]=c[k]
    end
    if all(i->Nemo.evaluate(fs[i],c1)%pr==0,1:i2-1)
      push!(sols,c1[:])
    end
  end
  if length(sols)==0
    return 0
  end
  xn=vp[1][:]
  i1=i2
  while i1<=length(fs)
    vn=[k for k in vp[i1] if !(k in xn)]
    i2=i1+1
    while i2<=length(fs) && all(k->k in xn || k in vn,vp[i2])
      i2+=1 
    end
    soln=[cc for cc in sols]
    sols=Array{Int8,1}[]
    for c0 in soln
      c1=c0[:]
      for c in Iterators.product([ff for j in 1:length(vn)]...)
        for k in 1:length(vn)
          c1[vn[k]]=c[k]
        end
        if all(i->Nemo.evaluate(fs[i],c1)%pr==0,i1:i2-1)
          push!(sols,c1[:])
        end
      end
    end
    if length(sols)==0
      return 0
    end
    append!(xn,vn)
    i1=i2
  end
  if length(xn)<length(xs) 
    return length(sols)*pr^(length(xs)-length(xn))
  else
    return length(sols)
  end
end        

"""`borelcosets( <lie>, <pr>, <g0> )`

returns the number of cosets <gB>, where <B> is a Borel subgroup of <G> 
(i.e., upper triangular matrices in <G>), that are fixed by a given 
unipotent element <g0>, given as a word in Chevalley root elements 
corresponding to positive roots. There is a fourth, optional argument 
by which one can give the maximum length of Weyl group elements to 
consider (the default value is <lie>.N.) This function requires the 
Nemo package and the Singular interface to Julia. (If the latter is not
available, then there is a 5th optional argument that must be set to `false`.)
The following example is taken from Section 5 in  

M. Geck,  Computing   Green  functions  in  small  characteristisc, 
J. Algebra 561 (2020), 163--199.                                       

We consider a group of type `F_4` and the unipotent classes denoted
`F_4(a_1)` and `C_3(a_1)`, with representatives

`u0=x_{1000}(1)x_{0100}(1)x_{0110}(1)x_{0011}(1)` and 
`u1=x_{0100}(1)x_{0001}(1)x_{0120}(1)`, respectively.

```julia-repl
julia> l=LieAlg(:f,4)
#I dim = 52
julia> u0=[[[1,0,0,0],1],[[0,1,0,0],1],[[0,1,1,0],1],[[0,0,1,1],1]]
julia> borelcosets(l,3,u0)
#I Prime = 3, u0 = Array{Int64,1}[[1, 1], [2, 1], [6, 1], [7, 1]]
#I Delta = [1, 2, 7] ..............
#I Number of cosets = 24
#I Number of cosets to consider = 7263016
#I Considering w of length 
0: 1 ++
1: 7 +
2: 13 +-
3: 19 --
4: 19 --
5: 19 --
6: 19 --
7: 19 --
8: 19 --
9: 19 --
10: 19 -
11: 19 -
12: 19 -
13: 19 -
19
```
The function `borelcosets1` is similar but it considers all elements that 
are obtained from a given representative by conjugation with diagonal 
elements of order 2. (This is used in the above paper to see to what 
extent the representatives depend on the choice of the signs of the 
Chevalley basis.) As argument of the function, we just use the list of 
roots involved in <g0>.

```julia-repl
julia> u1=[[0,1,0,0],[0,0,0,1],[0,1,2,0]]
julia> borelcosets1(l,3,u1)
#I Torus orbits: 2
#I Prime = 3, u0 = Array{Int64,1}[[2, 1], [4, 1], [9, 1]]
#I Delta = [2, 4] ......................
#I Number of cosets = 288
#I Number of cosets to consider = 79312105600
#I Considering w of length 
0: 1 ++
1: 7 ++!+
2: 22 +++!--
3: 49 ++!++----
4: 121 +++!-++-----
5: 292 ++++!!-++------
6: 544 !++++-!--++-------
7: 796 -!++++-!---++-------
8: 1066 -!+++++-!----+--------
9: 1408 -!+++++-!-----+--------
10: 1714 -!!+++++-------+--------
11: 2200 --!!++++-------+-------
12: 2983 --!!++++--------------
13: 3793 ---!+++-------------
14: 4360 ---!+++-----------
15: 4927 ---!++---------
16: 5413 ---!+-------
17: 5656 ---------
18: 5656 ------
19: 5656 ----
20: 5656 --
21: 5656 -
#I borelcosets=5656
#I Prime = 3, u0 = Array{Int64,1}[[2, -1], [4, 1], [9, 1]]
#I Delta = [2, 4] ......................
#I Number of cosets = 288
#I Number of cosets to consider = 79312105600
#I Considering w of length 
0: 1 ++
1: 7 ++++
2: 28 ++++--
3: 85 +++++----
4: 211 ++++-++-----
5: 436 ++++++-++------
6: 796 +++++-+--++-------
7: 1264 -+++++-+---++-------
8: 1750 -++++++-+----+--------
9: 2272 -++++++-+-----+--------
10: 2830 -+++++++-------+--------
11: 3640 --++++++-------+-------
12: 4747 --++++++--------------
13: 5881 ---++++-------------
14: 6934 ---++++-----------
15: 7987 ---+++---------
16: 8959 ---++-------
17: 9688 ---------
18: 9688 ------
19: 9688 ----
20: 9688 --
21: 9688 -
#I borelcosets=9688
2-element Array{Any,1}:
 Any[Array{Int64,1}[[2, 1], [4, 1], [9, 1]], 
 Array{Int64,1}[[7, 1], [6, 2], [5, 1], [4, 4], [3, 3], [1, 3]], 5656] 
 Any[Array{Int64,1}[[2, -1], [4, 1], [9, 1]], 
 Array{Int64,1}[[7, 1], [6, 2], [5, 1], [4, 4], [3, 3], [1, 3]], 9688]
```
(Thus, we have actually discovered an error in the above paper: the
two representatives in `C_3(a_1)` mentioned there, are not conjugate
in `F_4(3)`. Another misprint is in Section 8.4: it should be 
`Q_1(y_{46})>17323` for `q=2`.)

The computation in Section 9 of the paper is verified as follows.

```julia-repl
julia> l=LieAlg(:e,8); 
julia> z77=[23,24,25,27,26,28,15,22] # indices of roots 
julia> @time borelcosets(l,2,[[l.roots[i],1] for i in z77])
#I Prime = 2, u0 = Array{Int64,1}[[23, 1], [24, 1], [25, 1], [27, 1], [26, 1], [28, 1], [15, 1], [22, 1]]
#I calculating structconsts
#I calculating eps-canonical base (100/.) ..
#I Delta = [15, 23, 24, 25, 26, 27, 28] ...........................................................
#I Number of cosets = 17280
#I Number of cosets to consider = 1484991951529417005
#I Considering w of length 
0: 1 ++++++++
1: 17 +++++++++++++++++++++++++++++++++
2: 149 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
3: 933 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

...

41: 4599869 +------------------------------!-------------!-------------------
42: 4601917 +------------------------------------------------------
43: 4603965 ----------------------------------------------
44: 4603965 --------------------------------------
45: 4603965 --------------------------------
46: 4603965 --------------------------
47: 4603965 ---------------------
48: 4603965 -----------------
49: 4603965 -------------
50: 4603965 ----------
51: 4603965 --------
52: 4603965 ------
53: 4603965 ----
54: 4603965 ---
55: 4603965 --
56: 4603965 -
57: 4603965 -
58: 4603965 -
4603965
107.129535 seconds (354.10 M allocations: 16.967 GiB, 9.39% gc time)
```
In order to deal with groups of twisted type `E_6`, we also provide
a function which takes a symmetry of oder `2` of the Cartan matrix 
as additional argument. The computation in Section 7 of the paper is 
verified as follows.

```julia-repl
julia> l=LieAlg(:e,6); 
julia> tw=[6,2,5,4,3,1] # permutation given by graph automorphism
julia> twistedborelcosets(l,3,[[i,1] for i in ChevLie.E6critical["e6a3"]],tw)
#I Prime = 3, u0 = Array{Int64,1}[[1, 1], [6, 1], [3, 1], [5, 1], [4, 1], [27, 1]]
#I Delta = [1, 3, 4, 5, 6] .....................
#I Number of cosets = 72
#I Number of cosets to consider = 122946474063654874360
#I Considering w of length 
0: 1 +
1: 4 +
2: 7 
3: 7 +
4: 16 +
5: 25 ++
6: 43 +
7: 70 +
8: 97 +
9: 124 ++
10: 178 ++
11: 214 !
12: 214 +
13: 268 -
14: 268 +!
15: 322 -
16: 322 !
17: 322 
18: 322 !
19: 322 !
20: 322 !
322
```
See also `borelcosets1`, `torusorbit`.
"""
function borelcosets(lie::LieAlg,pr::Int,g0,maxl::Int=-1,singular::Bool=true)
  u0=Array{Int,1}[]
  for i in 1:length(g0)
    r8=[Int8(j) for j in g0[i][1]]
    push!(u0,[findfirst(==(r8),lie.roots),g0[i][2]])
  end
  println("#I Prime = ",pr,", u0 = ",u0)
  str=structconsts(lie)
  if maxl==-1
    alle=helpborel(lie,u0,lie.N)
  else
    alle=helpborel(lie,u0,maxl)
  end
  print("#I Number of cosets to consider = ")
  flush(stdout)
  poin=BigInt(0)
  for a in alle
    poin+=BigInt(pr)^length(a)
  end
  println(poin)
  print("#I Considering w of length ")
  flush(stdout)
  cc=1
  for len in 1:length(alle[end])
    println("")
    print(len-1,": ",cc," ")
    flush(stdout)
    R,x=Nemo.polynomial_ring(Nemo.ZZ,["x"*string(i) for i in 1:len])
    if singular==true
      S,y=Singular.polynomial_ring(Singular.Fp(pr),
             ["y"*string(i) for i in 1:len],ordering=:revlex)
    end
    eins=x[1]^0
    allw=[w for w in alle if length(w)==len]
    for w in allw
      cc1=0
      p=collect(1:2*lie.N)
      for s in w
        p=[lie.perms[s][i] for i in p]
      end
      rl=[i for i in 1:lie.N if p[i]>lie.N]
      wgt=Int[]
      for i in 1:lie.N
        if (i in rl)
          push!(wgt,i)
        else
          push!(wgt,lie.N+1)
        end
      end
      u01=[[rl[i],-x[i]] for i in length(rl):-1:1]
      append!(u01,[[u0[i][1],u0[i][2]*eins] for i in 1:length(u0)])
      append!(u01,[[rl[i],x[i]] for i in 1:length(rl)])
      u01=bcollect_chevrootelts(lie,u01,wgt)
      zua=[u01[i][2] for i in 1:length(u01) if u01[i][1] in rl]
      if singular==true
        szua=Singular.Ideal(S,[pol_nemotosing(pol,y) for pol in zua]...)
        gb=Singular.std(szua,complete_reduction=true)
        ngb=[pol_singtonemo(R,pol1,pr) for pol1 in Singular.gens(gb)]
      else
        ngb=collect(Set(zua))
      end
      gf=gfp_points(R,x,ngb,pr)
      if gf==-1
        print("-")
      elseif gf==0
        print("!")
      else
        print("+")
        cc+=gf
      end
      flush(stdout)
    end
  end
  println("")
  return cc
end

# plain singular version
function borelcosets_sing(lie::LieAlg,pr::Int,g0,maxl::Int=-1)
  u0=Array{Int,1}[]
  for i in 1:length(g0)
    r8=[Int8(j) for j in g0[i][1]]
    push!(u0,[findfirst(==(r8),lie.roots),g0[i][2]])
  end
  println("#I Prime = ",pr,", u0 = ",u0)
  str=structconsts(lie)
  if maxl==-1
    alle=helpborel(lie,u0,lie.N)
  else
    alle=helpborel(lie,u0,maxl)
  end
  print("#I Number of cosets to consider = ")
  flush(stdout)
  poin=BigInt(0)
  for a in alle
    poin+=BigInt(pr)^length(a)
  end
  println(poin)
  print("#I Considering w of length ")
  flush(stdout)
  cc=1
  for len in 1:length(alle[end])
    println("")
    print(len-1,": ",cc," ")
    flush(stdout)
    R,x=Nemo.polynomial_ring(Nemo.ZZ,["x"*string(i) for i in 1:len])
    S,y=Singular.polynomial_ring(Singular.Fp(pr),
           ["y"*string(i) for i in 1:len],ordering=:revlex)
    eins=y[1]^0
    allw=[w for w in alle if length(w)==len]
    for w in allw
      cc1=0
      p=collect(1:2*lie.N)
      for s in w
        p=[lie.perms[s][i] for i in p]
      end
      rl=[i for i in 1:lie.N if p[i]>lie.N]
      wgt=Int[]
      for i in 1:lie.N
        if (i in rl)
          push!(wgt,i)
        else
          push!(wgt,lie.N+1)
        end
      end
      u01=[[rl[i],-y[i]] for i in length(rl):-1:1]
      append!(u01,[[u0[i][1],u0[i][2]*eins] for i in 1:length(u0)])
      append!(u01,[[rl[i],y[i]] for i in 1:length(rl)])
      u01=bcollect_chevrootelts(lie,u01,wgt)
      zua=[u01[i][2] for i in 1:length(u01) if u01[i][1] in rl]
      gb=Singular.std(Singular.Ideal(S,zua...),complete_reduction=true)
      ngb=[pol_singtonemo(R,pol1,pr) for pol1 in Singular.gens(gb)]
      gf=gfp_points(R,x,ngb,pr)
      if gf==-1
        print("o")
      elseif gf==0
        print("!")
      else
        print("+")
        cc+=gf
      end
      flush(stdout)
    end
  end
  println("")
  return cc
end

# a version for twisted groups, where twist has order 2 (not yet optimized)
function twistedborelcosets(lie::LieAlg,pr::Int,g0,twist::Array{Int,1})
  u0=Array{Int,1}[]
  for i in 1:length(g0)
    r8=[Int8(j) for j in g0[i][1]]
    push!(u0,[findfirst(==(r8),lie.roots),g0[i][2]])
  end
  println("#I Prime = ",pr,", u0 = ",u0)
  str=structconsts(lie)
  alle=Array{Int,1}[]
  for w in helpborel(lie,u0,lie.N)
    pw=wordperm(lie,w)
    if pw==wordperm(lie,[twist[i] for i in w])
      push!(alle,w)
    end
  end
  print("#I Number of cosets to consider = ")
  flush(stdout)
  poin=BigInt(0)
  for a in alle
    poin+=BigInt(pr)^(2*length(a))
  end
  println(poin)
  proots=Array{Int,1}[]
  for r in 1:lie.N
    twr=[Int8(lie.roots[r][twist[i]]) for i in 1:length(lie.roots[r])]
    fr=findfirst(==(twr),lie.roots)
    if r==fr
      push!(proots,[r])
    else
      if r<fr 
        nr=[r,fr]
      else
        nr=[fr,r]
      end
      if !(nr in proots)
        push!(proots,nr)
      end
    end
  end
  print("#I Considering w of length ")
  flush(stdout)
  cc=1
  for len in 1:length(alle[end])
    println("")
    print(len-1,": ",cc," ")
    flush(stdout)
    R,x=Nemo.polynomial_ring(Nemo.ZZ,["x"*string(i) for i in 1:len])
    eins=x[1]^0
    allw=[w for w in alle if length(w)==len]
    for w in allw
      cc1=0
      p=collect(1:2*lie.N)
      for s in w
        p=[lie.perms[s][i] for i in p]
      end
      rl=[i for i in 1:lie.N if p[i]>lie.N]
      wgt=Int[]
      for i in 1:lie.N
        if (i in rl)
          push!(wgt,i)
        else
          push!(wgt,lie.N+1)
        end
      end
      u01=[[rl[i],-x[i]] for i in length(rl):-1:1]
      append!(u01,[[u0[i][1],u0[i][2]*eins] for i in 1:length(u0)])
      append!(u01,[[rl[i],x[i]] for i in 1:length(rl)])
      u01=bcollect_chevrootelts(lie,u01,wgt)
      zua=[u01[i][2] for i in 1:length(u01) if u01[i][1] in rl]
      for rr in proots  
        if rr[1] in rl
          xr1=findfirst(==(rr[1]),rl)
          if length(rr)==1
            push!(zua,x[xr1]^pr-x[xr1])
          else
            xr2=findfirst(==(rr[2]),rl)
            push!(zua,x[xr2]-x[xr1]^pr)
          end
        end
      end
      ngb=collect(Set(zua))
      gf=fq_points(R,x,ngb,pr,2)
      if gf==-1
        print("-")
      elseif gf==0
        print("!")
      else
        print("+")
        cc+=gf
      end
      flush(stdout)
    end
  end
  println("")
  return cc
end

"""`borelcosets1( <lie>, <pr>, <g0> )`

returns a list of triples `[n0,jb,b]` where `n0` runs over a set of
representatives under the action of diagonal elements of order 2, 
`jb` is the Jordan normal form of `n0`, and `b` is the number of
Borel cosets fixed by `n0`.

For the convenience of the reader, we provide four dictionaries
which provide input data for this function corresponding to the
various cases considered in the paper: 

M. Geck,  Computing   Green  functions  in  small  characteristisc, 
J. Algebra 561 (2020), 163--199.                                      

The dictionaries are called `F4critical`, `E6critical`, `E7critical`, 
`E8critical`. For example (see Section 5.2 of the paper):

```julia-repl
julia> l=LieAlg(:f,4)
#I dim = 52
julia> ChevLie.F4critical
Dict{String,Array{Array{Int64,1},1}} with 4 entries:
"c3a1" => [[0,1,0,0], [0,0,0,1], [0,1,2,0]]
"f4a1" => [[1,0,0,0], [0,1,0,0], [0,1,1,0], [0,0,1,1]]
"f4a3" => [[1,1,0,0], [0,1,2,0], [0,1,2,2], [1,1,2,2]]
"f4a2" => [[1,1,0,0], [0,1,2,0], [0,0,0,1], [0,0,1,1]]

julia> borelcosets1(l,3,ChevLie.F4critical["f4a1"])
#I Torus orbits: 1
#I Prime = 3, u0 = Array{Int64,1}[[1, 1], [2, 1], [6, 1], [7, 1]]
#I Delta = [1, 2, 7] ..............
#I Number of cosets = 24
#I Number of cosets to consider = 7263016
#I Considering w of length 
0: 1 ++
1: 7 +
2: 13 +-
3: 19 --
4: 19 --
5: 19 --
6: 19 --
7: 19 --
8: 19 --
9: 19 --
10: 19 -
11: 19 -
12: 19 -
13: 19 -
#I borelcosets=19
1-element Array{Any,1}:
Any[Array{Int64,1}[[1, 1], [2, 1], [6, 1], [7, 1]], 
  Array{Int64,1}[[9, 5], [7, 1]], 19]
```
See also `borelcosets`.
"""
function borelcosets1(lie::LieAlg,pr::Int,g0,singular::Bool=true)
  R=Nemo.GF(pr)
  eins=R(1)
  str=structconsts(lie)
  r0=Int[]
  for r in g0 
    r8=[Int8(i) for i in r]
    push!(r0,findfirst(==(r8),lie.roots))
  end
  print("#I Torus orbits: ")
  if pr==2
    orbs=[[1 for i in 1:length(r0)]]
  else
    c=Array{Int,1}[]
    for tup in Iterators.product([[1,-1] for i in 1:length(r0)]...)
      push!(c,collect(tup))
    end
    orbs=Array{Int,1}[]
    while length(c)>0
      push!(orbs,c[1])
      orb=torusorbit(lie,[[r0[i],c[1][i]] for i in 1:length(r0)])
      o=[[p[2] for p in o1] for o1 in orb]
      f=[findfirst(i->c[i]==o1,1:length(c)) for o1 in o]
      sort!(f)
      deleteat!(c,f)
    end
  end
  println(length(orbs))
  flush(stdout)
  res=[]
  for c1 in orbs 
    n0=[[r0[i],c1[i]] for i in 1:length(r0)]
    m=Nemo.matrix(R,chevrootelt(lie,n0[1][1],n0[1][2]*eins))
    for i in 2:length(n0)
      m*=Nemo.matrix(R,chevrootelt(lie,n0[i][1],n0[i][2]*eins))
    end
    jb=jordanblocks(m)
    b=borelcosets(lie,pr,[[lie.roots[nn[1]],nn[2]] for nn in n0],-1,singular)
    println("#I borelcosets=",b)
    push!(res,[n0,jb,b])
  end
  return res
end

F4critical=Dict("f4a1"=>[[1,0,0,0],[0,1,0,0],[0,1,1,0],[0,0,1,1]],
"f4a2"=>[[1,1,0,0],[0,1,2,0],[0,0,0,1],[0,0,1,1]],
"f4a3"=>[[1,1,0,0],[0,1,2,0],[0,1,2,2],[1,1,2,2]],
"c3a1"=>[[0,1,0,0],[0,0,0,1],[0,1,2,0]])

E6critical=Dict("e6a3"=>[[1,0,0,0,0,0],[0,0,0,0,0,1],[0,0,1,0,0,0],
[0,0,0,0,1,0],[0,0,0,1,0,0],[1,1,1,1,1,1]])

E7critical=Dict("e7a3"=>[[1,0,0,0,0,0,0],[0,1,0,1,0,0,0],[0,0,1,1,0,0,0],
[0,1,0,1,1,0,0],[0,1,1,1,1,0,0],[0,0,0,0,1,1,0],[0,0,0,0,0,0,1]],
"e7a4"=>[[1,0,0,0,0,0,0],[0,1,1,1,0,0,0],[0,0,1,1,1,0,0],[0,1,0,1,1,0,0],
[0,0,0,1,1,1,0],[0,0,1,1,1,1,0],[0,0,0,0,0,1,1]],
"e7a5"=>[[1,0,1,1,0,0,0],[0,1,1,1,0,0,0],[0,0,1,1,1,0,0],[0,1,0,1,1,0,0],
[0,0,0,1,1,1,0],[0,0,0,0,1,1,1],[1,1,1,1,1,0,0],[1,1,1,1,1,1,0]],
"e6a3"=>[[1,0,1,0,0,0,0],[0,1,1,1,0,0,0],[0,0,1,1,1,0,0],[1,1,1,1,0,0,0],
[0,1,0,1,1,1,0],[0,0,0,1,1,1,1]])

E8critical=Dict("e8b6"=>[[1,1,1,1,0,0,0,0],[1,0,1,1,1,0,0,0],
[0,1,1,1,1,0,0,0],[0,0,1,1,1,1,0,0],[0,1,0,1,1,1,0,0],[0,0,0,1,1,1,1,0],
[0,0,0,0,0,0,1,1],[0,0,0,0,0,1,1,1]])

# delete element from a list:
# deleteat!(a, a .== 10); or deleteat!(a, findfirst(a .== 10))

"""`lietest()`

performs some tests with the programs in this `chevlie` module.
"""
function lietest()
  println("#I --->  TESTING MODULE CHEVLIE  <---")
  l=LieAlg(:d,5,[4,3,1,5,2],-1)
  a=allelms(l)
  r3=rep_minuscule(l,l.minuscule[3],true)
  r1=canchevbasis_min(l,l.minuscule[3])
  r2=rep_sc(l,true)
  println(canchevbasis(l,r2)==canchevbasis_sc(l))  
  println("")
  flush(stdout)
  l=LieAlg(:f,4,[2,3,4,1],-1)
  gr=[generic_gram_wdd(l,d) for d in weighted_dynkin_diagrams(l)[1]]
  display(gr[9])
  m=monomialelts(l,(l.e_i,l.f_i,l.h_i))
  println("Braid relation ",m[2]*m[3]*m[2]*m[2]==m[3]*m[2]*m[3]*m[2])
  checkrels(l,l.e_i,l.f_i,l.h_i)
  l=LieAlg(:b,2)
  R,x=Nemo.polynomial_ring(Nemo.ZZ,["x"*string(i) for i in 1:2])
  display(cross_regular1(l,x))
  l=LieAlg(:g,2)
  R,x=Nemo.finite_field(2,2,"x")
  c=[jordandecelt(a,2) for a in cross_regular(l,x)]
  display(chevrootelt(l,11,-19))
  b=canchevbasis_sc(l)
  c=structconsts(l)
  println("")
  flush(stdout)
end
#lietest()

function redirect_to_files(dofunc, outfile, errfile)
    open(outfile, "w") do out
        open(errfile, "w") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    dofunc()
                end
            end
        end
    end
end

#redirect_to_files("oute8z78" * ".log", "oute8z78" * ".err") do
#   l=LieAlg(:e,8)
#   include("e8grp2.jl")
#   b=borelcosets(l,2,z78,[e for e in els77 if length(e)<=24])
#end

# make this better, using sparse etc.
function chevrooteltprod(lie::LieAlg,n0,pr)
  R=Nemo.GF(pr)
  eins=R(1)
  m=Nemo.matrix(R,chevrootelt(lie,n0[1][1],n0[1][2]*eins))
  for i in 2:length(n0)
    m*=Nemo.matrix(R,chevrootelt(lie,n0[i][1],n0[i][2]*eins))
  end
  return jordanblocks(m)
end
  
function banner()
println("####################################################################")
println("##                                                                ##")
println("##   Welcome to  version 1.3  of the Julia module  `ChevLie`:     ##")
println("##      CONSTRUCTING  LIE  ALGEBRAS  AND  CHEVALLEY  GROUPS       ##")
println("##   Meinolf Geck,  University of Stuttgart, 22 February 2026     ##")
println("##   https://pnp.mathematik.uni-stuttgart.de/idsr/idsr1/geckmf/   ##")
println("##                                                                ##")
println("##   Type   lietest()   to run some tests.                        ##")
println("##   Type   ?LieAlg     for first help; all comments welcome!     ##")
println("##                                                                ##")
println("####################################################################")
end

banner()

end # module ChevLie

