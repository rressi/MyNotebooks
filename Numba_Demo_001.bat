@echo off
ipython notebook Numba_Demo_001.ipynb
ipython nbconvert Numba_Demo_001.ipynb --to slides --post serve
pause