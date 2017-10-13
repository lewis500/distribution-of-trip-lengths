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
  μ::Array{Float64}
  Σ::Array{Float64}
  z::MultivariateDistribution #this is produced from Σ and μ
end
#the generator function for World objects. All it does is take the means and covariance matrix used to produce the population, and creates a bivariate lognormal distribution for it as the z parameter in the World object
function World(;
  λ::Float64 = 20.,
  pf::Float64 = 2.2,
  k0::Float64 = 55.,
  μ::Array{Float64} = [1.0; 1.0],
  Σ::Array{Float64} = [.1 .1; .1 .1]
  )
  return World(λ,pf,k0,μ,Σ,MvLogNormal(μ, Σ))
end

# Equilibrium is defined as type consisting of a density, a pace and a circulation
Equilibrium = Tuple{Float64, Float64, Float64}

# pk and qk are physical functions used in our calculations. pk is p(k) in the paper. qk is q(k) in the paper. these are the mfd functions
pk(w::World, k::Float64)::Float64 = w.pf*exp(.5(k/w.k0)^2)
qk(w::World, k::Float64)::Float64 = k/pk(w,k)

# X() is as defined as in the paper: the disutility of a car trip when pace is p, trip length is l and the given tolls
X(p::Float64,l::Float64;τa::Float64=0.,τd::Float64=0.)::Float64 = (p+τd)*l + τa

##### the following block gives all the functions required to calculate the equilbrium.
##### each takes a "World" (a set of parameters), a pace p and the tolls given by obvious keyword arguments

# qdp gives the cirulation depanded for a given p and set of tolls.
function qdp(w::World,p::Float64;τa::Float64=0.,τd::Float64=0.)
  function f(x)::Float64
    l::Float64 = x[2]/(1-x[2])
    ɛ::Float64 = (p+τd)*l + τa + x[1]/(1-x[1])
    return l*pdf(w.z,[ɛ;l])/((1-x[2])*(1-x[1]))^2
  end
  return w.λ*hcubature(f,[0.,0.],[1.,1.];abstol=1e-4)[1]
end

# adp gives the arrival flow demanded for a given p and set of tolls
function adp(w::World,p::Float64;τa::Float64=0.,τd::Float64=0.)::Float64
  function f(x)::Float64
      l::Float64 = x[2]/(1-x[2])
      cost::Float64 = (p+τd)*l + τa
      ɛ::Float64 = cost+ x[1]/(1-x[1])
      pdf(w.z,[ɛ;l])/((1-x[2])*(1-x[1]))^2
  end
  return w.λ*hcubature(f,[0.,0.],[1.,1.];abstol=1e-4)[1]
end

# TSS gives the total social surplus for a given p and set of tolls
function TSS(w::World,p::Float64;τa::Float64=0.,τd::Float64=0.)::Float64
  function f(x)::Float64
    l::Float64 = x[2]/(1-x[2])
    cost::Float64 = (p+τd)*l + τa
    ɛ::Float64 = cost + x[1]/(1-x[1])
    pdf(w.z,[ɛ;l])*(ɛ-l*p)/((1-x[2])*(1-x[1]))^2
  end
  return w.λ*hcubature(f,[0.,0.],[1.,1.];abstol=1e-3)[1]
end

# CS gives the consumer surplus for a given p and set of tolls
function CS(w::World,p::Float64;τa::Float64=0.,τd::Float64=0.)::Float64
  function f(x)::Float64
    l::Float64 = x[2]/(1-x[2])
    cost::Float64 = (p+τd)*l + τa
    ɛ::Float64 = cost + x[1]/(1-x[1])
    pdf(w.z,[ɛ;l])*(ɛ-cost)/((1-x[2])*(1-x[1]))^2
  end
  return w.λ*hcubature(f,[0.,0.],[1.,1.];abstol=1e-3)[1]
end

#### the functions chooseA and chooseT choose the welfare-maximizing access and trip toll levels
function chooseA(w::World)::Float64
  optimize(τd::Float64->begin
    p= getEq(w;τd=τd)[2]
    -TSS(w,p;τd=τd)
  end,0.,4.0,
  ;abs_tol=1e-4)|>minimizer
end

function chooseT(w::World)::Float64
  optimize(τa::Float64->begin
    -TSS(w,getEq(w;τa=τa)[2];τa=τa)
  end,1.,10.,
  ;abs_tol=1e-4)|>minimizer
end

##### the getEq function finds the equilibrium given tolls for a given "world"
function getEq(w::World;τa::Float64=0.,τd::Float64=0.)::Equilibrium
  ke::Float64 = fzero(
    k::Float64-> qk(w,k) - qdp(w,pk(w,k); τa=τa,τd=τd)
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
  # supply curve
  ax[:plot](qsRange,pRange,label=L"p_s/\tilde{p}_s", ls="-",color="royalblue")
  # untolled
  ax[:plot](map(p->qdp(w,p),pRange),pRange,label=L"p_d(q;0,0)",ls="-.",color="coral")
  # distance toll
  τd = chooseA(w)
  ax[:plot](map(p->qdp(w,p;τd=τd,τa=0.),pRange),pRange,label=L"p_d(q;\tau_d^*,0)",ls=":",color="green")
  # access toll
  τa = chooseT(w)
  ax[:plot](map(p->qdp(w,p;τa=τa,τd=0.),pRange),pRange,label=L"p_d(q;0,\tau_a^*)",ls="--",color="red")
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
  τd = chooseA(w)
  kd,pd,qd = getEq(w;τd=τd)
  τa = chooseT(w)
  kt,pt,qt = getEq(w;τa = τa)

  lRange = linspace(0,mean(w.z)[2]*2,100)

  function plotEq(w::World,ax;label="o",ls="-",τa=0.,τd=0.,color="blue")
    k,p,q = getEq(w;τa=τa,τd=τd)
    mass = map(l->begin
      quadgk(x-> begin
        ɛ = X(p,l;τa=τa,τd=τd) + x/(1-x)
        pdf(w.z,[ɛ;l])
      end,0.,1.)[1]
    end,lRange)
    ax[:plot](lRange,mass,label=label,ls=ls,color=color)
  end

  fig,ax = subplots(figsize=(4,3))
  ax[:set_xlabel]("km")
  ax[:set_ylabel]("density")
  plotEq(w,ax;label=L"g(l;0,0)",ls="-.",color="coral")
  plotEq(w,ax;label=L"g(l;\tau_d^*,0)",τd=τd,ls=":",color="green")
  plotEq(w,ax;label=L"g(l;0,\tau_a^*)",τa=τa,ls="--",color="red")
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

  τd = chooseA(w)
  kd,pd,qd = getEq(w;τd=τd)
  ad = adp(w,pd;τd=τd)
  tssd = TSS(w,pd;τd=τd)
  csd = CS(w,pd;τd=τd)
  ld = qd/ad
  trd = tssd - csd

  τa = chooseT(w)
  kt,pt,qt = getEq(w;τa = τa)
  at = adp(w,pt;τa=τa)
  tsst = TSS(w,pt;τa=τa)
  cst = CS(w,pt;τa=τa)
  lt = qt/at
  trt = tsst - cst

  # a helper function to round all the results to a few decimal places
  rounder(arr::Array{Float64})::Array{Float64} = map(x-> round(x,1) , arr)
  return DataFrame(
    toll=["none","distance","access"],
    τ=[0,τd,τa]|>rounder,
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
  ax[:plot](X, map(l->l*p, X),label=L"D_{untolled}",ls="-.",c="coral",lw=2)
  τd = chooseA(w)
  kd,pd,qd = getEq(w;τd=τd)
  ax[:plot](X, map(l->l*(pd + τd), X),label=L"D_{\tau_d^*}",ls=":",c="g",lw=2)
  τa = chooseT(w)
  kt,pt,qt = getEq(w;τa = τa)
  ax[:plot](X, map(l->l*pt + τa, X),label=L"D_{\tau_a^*}",ls="--",c="r",lw=2)
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
function plotAdvantage(w::World)
  n = 20
  cvRange = linspace(-.15,.15,n)
  res = map(cv->begin
    w = World(;
      μ=w.μ,
      Σ=[w.Σ[1,1] cv; cv w.Σ[2,2]],
      λ=w.λ,pf=w.pf,k0=w.k0)
    τd = chooseA(w)
    kd,pd,qd = getEq(w;τd=τd)
    tssd = TSS(w,pd;τd=τd)
    τa = chooseT(w)
    kt,pt,qt = getEq(w;τa=τa)
    tsst = TSS(w,pt;τa=τa)
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

#here, world is a "World" variable holding the parameters used in the paper. you can modify it with your own parameters.
world = Simulation.World(;
  μ=[2.4,1.0],
  Σ=[0.2 0.12; 0.12 0.2],
  λ=20.,
  pf=2.2,
  k0=55.)

# print the table
Simulation.createTable(world)

# make the plots
fig = Simulation.plotTripLengths(world)
fig[:savefig]("img/tripLengths.pdf")

# fig2 = Simulation.plotAdvantage(world)
# fig2[:savefig]("img/advantage.pdf")

fig3 = Simulation.plotSupplyAndDemand(world)
fig3[:savefig]("img/supplyDemand.pdf")

fig4 = Simulation.plotDivision(world)
fig4[:savefig]("img/division.pdf")
