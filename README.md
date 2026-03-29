# Computational optimisation in Navier-Stokes and Helmholtz systems

This assignment implements two classical systems, which we use to optimise two different cases:
<uL>
  <li>Navier-Stokes: We implement a Karman vortex sheet and try to find the integration method that can remain stable at high Reynolds numbers.</li>
  <li>Helmholtz: We try to find the optimal wifi router position, simulating the system using a finite difference method and using a grid search to find the 
  optimal position.</li>
</uL>

### Requirements:
<uL>
  <li>Python 3.12.9+</li>
</uL>

### Dependencies:
<uL>
  <li>numpy 1.26.4</li>
  <li>matplotlib 3.9.0</li>
  <li>scipy 1.15.1</li>
  <li>numba 0.62.1</li>
  <li>tqdm 4.67.1</li>
  <li>joblib 1.5.2</li>
  <li>ngsolve 6.2.2602.post2</li>
</uL>

Install any missing modules using
```bash
pip install -r requirements.txt
```


### Figure reproduction:
The figures in the report are produced by the various python files as such:
<ul>
  <li>Figure 1, 2, 3: compare_re_100.py</li>
  <li>Figure 4: fd_wifi.py with res=60 (line 70)</li>
  <li>Figure 5: fd_wifi.py with f_hz=2.4e9 (line 75)</li>
  <li>Table 1: compare_re_100.py</li>
</ul>

To run the files use
```bash
python -m file_name.py
```
