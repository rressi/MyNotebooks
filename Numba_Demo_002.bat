@echo off
ipython notebook Numba_Demo_002.ipynb
ipython nbconvert Numba_Demo_002.ipynb --to slides --post serve
pause