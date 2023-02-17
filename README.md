# ChevLie.jl

This package contains  functions  for constructing  the   Lie algebra 
and the corresponding Chevalley groups associated with a root system.
Some basic functionality is described in the article 
https://doi.org/10.2140/jsag.2020.10.41.

If  you see anything to be improved  in this  package, please contact 
me  or make  an issue or a pull request in the GitHub repository.

### Installing

[For Julia novices]
To install this package, at the Julia command line:

  *  enter package mode with ]
  *  do the command
```
(@v1.8) pkg> add "https://github.com/geckmf/ChevLie.jl"
```
- exit package mode with backspace and then do
```
julia> using ChevLie
```
and you are set up. For first help, type "?LieAlg".

To update later to the latest version, do

```
(@v1.8) pkg> update ChevLie
```
This package requires julia 1.8 or later.

