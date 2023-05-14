# BabyMAKRO

***Disclaimer: Work-in-progress. Beware of errors!***

**Documentation:** 

1. The economic model and solution method is described detail in `BabyMAKRO.pdf`.
1. An overview of the economic model and solution method is given in `BabyMAKRO_slides.pdf`.

**Code structure**

1. Everything can be run from `0a - run all.ipynb`
1. The notebooks are numbered in the order they should be called
1. The main code is in `BabyMAKROModelClass` in `BabyMAKROMOdel.py`
1. The model blocks are in `blocks.py`
1. The steady state is found in `steady state.py`
1. The solver used is in `broyden_solver.py`

**Dependencies:**

`
pip install EconModel
pip install ConSav
`

The `EconModelClass` from [EconModel](https://github.com/NumEconCopenhagen/EconModel) is in particular used to interface with numba functions, and must be understood.