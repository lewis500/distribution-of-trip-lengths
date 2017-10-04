####### introduction
# the Julia code to run the simulations and produce the plots in Lewis Lehe's paper Downtown Tolling and the Distribution of Trip Lengths, published in the journal Economics of Transportation
# if you have questions email Lewis Lehe at lewis500@berkeley.edu
# in addition to using the julia language, a lot of the syntax comes from the matplotlib module from python. Julia is able to call this function with similar commands to those use in python.

######BEGIN
###### Here we defined the module Simulation, which contains all the code required to run our simulation. The functions are placed in a module so that you don't have to recompile every time you edit something.
module Simulation
using PyCall
using PyPlot
using LaTeXStrings
using Distributions
using Roots:fzero
using Optim:optimize,minimizer
using DataFrames
using Cubature
using QuadGK:quadgk

#the World object that holds the parameters for the simulation. a world object is passed into all the other functions we use. For example, the p(k) function (defined like in the paper) needs to know k0 and pf (k_0 and p_f in the paper)
struct World
  λ::Float64
  pf::Float64
  k0::Float64
  z::MultivariateDistribution
end
#the generator function for World objects. All it does is take the means and covariance matrix used to produce the population, and creates a bivariate lognormal distribution for it as the z parameter in the World object
function World(;
  λ::Float64 = 20.,
  pf::Float64 = 2.2,
  k0::Float64 = 55.,
  μ::Array{Float64} = [1.0; 1.0],
  Σ::Array{Float64} = [.1 .1; .1 .1]
  )
  return World(λ,pf,k0,MvLogNormal(μ, Σ))
end

# Equilibrium is defined as type consisting of a density, a pace and a circulation
Equilibrium = Tuple{Float64, Float64, Float64}

# pk and qk are physical functions used in our calculations. pk is p(k) in the paper. qk is q(k) in the paper. these are the mfd functions
pk(w::World, k::Float64)::Float64 = w.pf*exp(.5(k/w.k0)^2)
qk(w::World, k::Float64)::Float64 = k/pk(w,k)

# X() is as defined as in the paper: the disutility of a car trip when pace is p, trip length is l and the given tolls
X(p::Float64,l::Float64;τt::Float64=0.,τa::Float64=0.)::Float64 = (p+τa)*l + τt

##### the following block gives all the functions required to calculate the equilbrium.
##### each takes a "World" (a set of parameters), a pace p and the tolls given by obvious keyword arguments

# qdp gives the cirulation depanded for a given p and set of tolls.
function qdp(w::World,p::Float64;τt::Float64=0.,τa::Float64=0.)
  function f(x)::Float64
    l::Float64 = x[2]/(1-x[2])
    ɛ::Float64 = (p+τa)*l + τt + x[1]/(1-x[1])
    return l*pdf(w.z,[ɛ;l])/((1-x[2])*(1-x[1]))^2
  end
  return w.λ*hcubature(f,[0.,0.],[1.,1.];abstol=1e-4)[1]
end

# adp gives the arrival flow demanded for a given p and set of tolls
function adp(w::World,p::Float64;τt::Float64=0.,τa::Float64=0.)::Float64
  function f(x)::Float64
      l::Float64 = x[2]/(1-x[2])
      cost::Float64 = (p+τa)*l + τt
      ɛ::Float64 = cost+ x[1]/(1-x[1])
      pdf(w.z,[ɛ;l])/((1-x[2])*(1-x[1]))^2
  end
  return w.λ*hcubature(f,[0.,0.],[1.,1.];abstol=1e-4)[1]
end

# TSS gives the total social surplus for a given p and set of tolls
function TSS(w::World,p::Float64;τt::Float64=0.,τa::Float64=0.)::Float64
  function f(x)::Float64
    l::Float64 = x[2]/(1-x[2])
    cost::Float64 = (p+τa)*l + τt
    ɛ::Float64 = cost + x[1]/(1-x[1])
    pdf(w.z,[ɛ;l])*(ɛ-l*p)/((1-x[2])*(1-x[1]))^2
  end
  return w.λ*hcubature(f,[0.,0.],[1.,1.];abstol=1e-3)[1]
end

# CS gives the consumer surplus for a given p and set of tolls
function CS(w::World,p::Float64;τt::Float64=0.,τa::Float64=0.)::Float64
  function f(x)::Float64
    l::Float64 = x[2]/(1-x[2])
    cost::Float64 = (p+τa)*l + τt
    ɛ::Float64 = cost + x[1]/(1-x[1])
    pdf(w.z,[ɛ;l])*(ɛ-cost)/((1-x[2])*(1-x[1]))^2
  end
  return w.λ*hcubature(f,[0.,0.],[1.,1.];abstol=1e-3)[1]
end

#### the functions chooseA and chooseT choose the welfare-maximizing access and trip toll levels
function chooseA(w::World)::Float64
  optimize(τa::Float64->begin
    p= getEq(w;τa=τa)[2]
    -TSS(w,p;τa=τa)
  end,0.,4.0,
  ;abs_tol=1e-4)|>minimizer
end

function chooseT(w::World)::Float64
  optimize(τt::Float64->begin
    -TSS(w,getEq(w;τt=τt)[2];τt=τt)
  end,1.,10.,
  ;abs_tol=1e-4)|>minimizer
end

##### the getEq function finds the equilibrium given tolls for a given "world"
function getEq(w::World;τt::Float64=0.,τa::Float64=0.)::Equilibrium
  ke::Float64 = fzero(
    k::Float64-> qk(w,k) - qdp(w,pk(w,k); τt=τt,τa=τa)
    ,w.k0;order=2)
  return (ke, pk(w,ke), qk(w,ke))
end

###### the plotSupplyAndDemand function takes a "World" and plots the q_d and p_s(q) functions in q/p space (the walters diagram)
function plotSupplyAndDemand(w::World)
  kRange = linspace(w.k0*.2,w.k0*1.6, 30)
  pRange = map(k->pk(w,k),kRange)
  qsRange = map(k->qk(w,k),kRange)
  k,p,q = getEq(w)
  close("all")
  fig,ax = subplots(figsize=(4,3))
  ax[:plot](qsRange,pRange,label=L"p_s/\tilde{p}_s", ls="-")
  ax[:plot](map(p->qdp(w,p),pRange),pRange,label=L"p_d(q)",ls="-.")
  τa = chooseA(w)
  ax[:plot](map(p->qdp(w,p;τa=τa,τt=0.),pRange),pRange,label=L"p_d(q;\tau_d^*)",ls=":")
  τt = chooseT(w)
  ax[:plot](map(p->qdp(w,p;τt=τt,τa=0.),pRange),pRange,label=L"p_d(q;\tau_a^*)",ls="--")
  ax[:set_xlim]((0,qk(w,w.k0)*1.2))
  ax[:set_ylim]((0,pRange[end]))
  ax[:set_xlabel](L"q")
  ax[:set_ylabel](L"p")
  ax[:legend](frameon=false)
  fig[:tight_layout]()
  plt[:show]()
  return fig
end

##### plotTripLengths takes a "World," find the optimal access and distance tolls and then plots the trip length distribution
function plotTripLengths(w)
  k,p,q = getEq(w)
  τa = chooseA(w)
  kd,pd,qd = getEq(w;τa=τa)
  τt = chooseT(w)
  kt,pt,qt = getEq(w;τt = τt)

  lRange = linspace(0,mean(w.z)[2]*2,100)

  function plotEq(w::World,ax;label="o",ls="-",τt=0.,τa=0.)
    k,p,q = getEq(w;τt=τt,τa=τa)
    mass = map(l->begin
      quadgk(x-> begin
        ɛ = X(p,l;τt=τt,τa=τa) + x/(1-x)
        pdf(w.z,[ɛ;l])
      end,0.,1.)[1]
    end,lRange)
    ax[:plot](lRange,mass,label=label,ls=ls)
  end

  fig,ax = subplots(figsize=(4,3))
  ax[:set_xlabel]("km")
  ax[:set_ylabel]("density")
  plotEq(w,ax;label=L"g(l;untolled)",ls="-")
  plotEq(w,ax;label=L"g(l;\tau_a^*)",ls=":",τt=τt)
  plotEq(w,ax;label=L"g(l;\tau_d^*)",ls="-.",τa=τa)
  ax[:legend](frameon=false)
  fig[:tight_layout]()
  fig
end

##### createTable creates the table used in the simulation by finding the optimal toll values
function createTable(w::World)
  k,p,q = getEq(w)
  a = adp(w,p)
  l = q/a
  tss = TSS(w,p)
  cs = CS(w,p)
  tr = tss - cs

  τa = chooseA(w)
  kd,pd,qd = getEq(w;τa=τa)
  ad = adp(w,pd;τa=τa)
  tssd = TSS(w,pd;τa=τa)
  csd = CS(w,pd;τa=τa)
  ld = qd/ad
  trd = tssd - csd

  τt = chooseT(w)
  kt,pt,qt = getEq(w;τt = τt)
  at = adp(w,pt;τt=τt)
  tsst = TSS(w,pt;τt=τt)
  cst = CS(w,pt;τt=τt)
  lt = qt/at
  trt = tsst - cst

  # a helper function to round all the results to a few decimal places
  rounder(arr::Array{Float64})::Array{Float64} = map(x-> round(x,1) , arr)
  return DataFrame(
    toll=["none","distance","access"],
    τ=[0,τa,τt]|>rounder,
    k=[k,kd,kt]|>rounder,
    l=[l,ld,lt]|>rounder,
    p=[p,pd,pt]|>rounder,
    q=[q,qd,qt]|>rounder,
    a=[a,ad,at]|>rounder,
    CS=[cs,csd,cst]|>rounder,
    TR=[tr,trd,trt]|>rounder,
    TSS=[tss,tssd,tsst]|>rounder
    )
end
#plot the distribution of trip lengths in the UE, with the TSS-maximizing access toll and with the TSS-maximizing distance toll
function plotDivision(w::World)
  n = 30
  xMax = 1.5*mean(w.z)[2]
  yMax = 1.7*mean(w.z)[1]
  X = linspace(0,xMax,n)
  Y = linspace(0,yMax,n)
  Z = zeros(Float64,(n,n))
  for i=1:n,j=1:n
    Z[j,i] = pdf(w.z,[Y[i];X[j]])
  end
  plt[:close]("all")
  fig,ax = subplots(figsize=(4,3))
  ax[:contourf](X,Y,Z,cmap=:Blues)
  k,p,q = getEq(w)
  ax[:plot](X, map(l->l*p, X),label=L"D_{untolled}",ls="-",c="k",lw=2)
  τa = chooseA(w)
  kd,pd,qd = getEq(w;τa=τa)
  ax[:plot](X, map(l->l*(pd + τa), X),label=L"D_{\tau_d^*}",ls=":",c="m",lw=2)
  τt = chooseT(w)
  kt,pt,qt = getEq(w;τt = τt)
  ax[:plot](X, map(l->l*pt + τt, X),label=L"D_{\tau_a^*}",ls="--",c="r",lw=2)
  ax[:set_xlim](0,xMax)
  ax[:set_ylim](0,yMax)
  ax[:legend]()
  ax[:set_xlabel](L"l")
  ax[:set_ylabel](L"\varepsilon")
  ax[:text](0.5, 5.0,L"R_1")
  ax[:text](4.0,19.0,L"R_2")
  fig[:tight_layout]()
  plt[:show]()
  return fig
end

#### plotAdvantage creates the figure showing the advantage of the distance toll over the access toll for various values of covariance between benefit and trip length
function plotAdvantage()
  n = 20
  cvRange = linspace(-.15,.15,n)
  res = map(cv->begin
    w = World(;μ=[2.4,1.0],Σ=[0.2 cv; cv 0.2], λ=20.,pf=2.2,k0=55.)
    τa = chooseA(w)
    kd,pd,qd = getEq(w;τa=τa)
    tssd = TSS(w,pd;τa=τa)
    τt = chooseT(w)
    kt,pt,qt = getEq(w;τt=τt)
    tsst = TSS(w,pt;τt=τt)
    tssd/tsst-1
  end,cvRange)
  fig,ax = subplots(figsize=(4,3))
  ax[:plot](cvRange,res*100)
  ax[:set_xlabel](L"\sigma^2_{\varepsilon,l}")
  ax[:set_ylabel]("% welfare gain from distance toll")
  fig[:tight_layout]()
  return fig
end
#end of the Simulation Module
end
###### in this part of the code you define the free parameters and run whichever functions you want for the plot and table, saving them to disk in the images folder

#here, w is a variable holding the parameters used in the paper. you can modify it with your own parameters.
w = Simulation.World(;μ=[2.4,1.0],Σ=[0.2 0.12; 0.12 0.2], λ=20.,pf=2.2,k0=55.)

Simulation.createTable(w)

fig = Simulation.plotTripLengths(w)
fig[:savefig]("img/tripLengths.pdf")

fig2 = Simulation.plotAdvantage()
fig2[:savefig]("img/advantage.pdf")

fig3 = Simulation.plotSupplyAndDemand(w)
fig3[:savefig]("img/supplyDemand.pdf")

fig4 = Simulation.plotDivision(w)
fig4[:savefig]("img/division.pdf")