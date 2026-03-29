import time
from solver_fd import FDSolver
from solver_fem import FEMSolver
from solver_lbm import LBMSolver
from visualization import print_summary

def test_solver_limits():
    re_test_list = [100, 250, 500, 750, 1000, 1500, 2000, 3000]

    test_steps_fd = 1000
    test_steps_fem = 1000
    test_steps_lbm = 10000 
    
    methods_info = {
        'FD': {'max_stable_Re': None, 'resolution': 'Nx=220'},
        'FEM': {'max_stable_Re': None, 'resolution': 'max_h=0.03'},
        'LBM': {'max_stable_Re': None, 'resolution': 'Ny=120'}
    }

    print("Start searching...")
  
    # FD
    print("\n>>> Start testing FD solver...")
    last_stable_re = None
    for re in re_test_list:
        print(f"\n--- Trying FD Re = {re} ---")
        solver = FDSolver(Re=re, Nx=220, p_iters=50)
        solver.run(test_steps_fd, report_interval=500)
        
        if len(solver.history) < test_steps_fd:
            print(f"[*] FD diverged/crashed at Re={re}")
            break
        else:
            print(f"[*] FD is stable at Re={re}")
            last_stable_re = re
            methods_info['FD']['max_stable_Re'] = last_stable_re

    # ---------------------------------------------------------
    # 2. Test Finite Element Method (FEM)
    # ---------------------------------------------------------
    print("\n>>> Start testing FEM solver")
    last_stable_re = None
    for re in re_test_list:
        print(f"\n--- Trying FEM Re = {re} ---")
        solver = FEMSolver(Re=re, max_h=0.03, dt=0.001)
        solver.run(test_steps_fem, report_interval=500)
        
        if len(solver.history) < test_steps_fem:
            print(f"[*] FEM diverged/crashed at Re={re}!")
            break
        else:
            print(f"[*] FEM is stable at Re={re}.")
            last_stable_re = re
            methods_info['FEM']['max_stable_Re'] = last_stable_re

    # ---------------------------------------------------------
    # 3. Test Lattice Boltzmann Method (LBM)
    # ---------------------------------------------------------
    print("\n>>> Start testing LBM solver")
    last_stable_re = None
    for re in re_test_list:
        print(f"\n--- Trying LBM Re = {re} ---")
        solver = LBMSolver(Re=re, Ny=120, U_max_lbm=0.05)
        solver.run(test_steps_lbm, report_interval=5000)
        
        if len(solver.history) < test_steps_lbm:
            print(f"[*] LBM diverged/crashed at Re={re}!")
            break
        else:
            print(f"[*] LBM is stable at Re={re}.")
            last_stable_re = re
            methods_info['LBM']['max_stable_Re'] = last_stable_re

    # ---------------------------------------------------------
    # Print final summary table
    # ---------------------------------------------------------
    print("\n=== Limit test complete ===")
    print_summary(methods_info)

if __name__ == "__main__":
    test_solver_limits()