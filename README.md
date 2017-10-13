# Downtown Tolls and the Distribution of Trip Lengths

This repository contains the simulation code, Latex source and bibliography for Lewis Lehe's paper *Downtown Tolls and the Distribution of Trip Lengths*, published in **Economics of Transportation.**

The file simulations.jl, above, is written in the [Julia](https://julialang.org/) language, version 0.6.0, and is used to produce the plots and table used in the numerical simulation part of the paper. The code is documented via comments. Even if the user is not familiar with the Julia language, it should be obvious how to code the equivalent program in another language---especially Python or MatLab, to which Julia bears a strong resemblance visually. The language includes types, which should help make the code comprehensible even to the non-user.

### Abstract

> Currently, all downtown tolls are ``access tolls,'' meaning they charge for gross access to a zone, but tolls levied on distance-traveled are on the horizon. This paper shows how such tolls affect the distribution of trip lengths. A static model is presented in which travelers with potentially different trip lengths make a probabilistic choice of whether to enter a downtown zone governed by a Macroscopic Fundamental Diagram (MFD), with the choice probability declining as tolls and travel time rise. An application of Little's Law allows the model's equilibria to be derived in terms of a familiar supply/demand framework. Analysis proves and numerical simulation demonstrates that, if trip lengths and the value of a trip both vary across travelers, then access tolls inefficiently shift the distribution of car trip lengths toward long trips, whereas a distance toll can achieve the welfare-maximizing set of car trips. 

