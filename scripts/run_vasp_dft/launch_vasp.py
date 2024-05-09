import sys
import subprocess
import os

# Link to the path with ML relaxations
PATH = ""

MAX_CALCS = 200

def run_vasp(path):
    cmd = "mpirun -np 8 /opt/vasp.5.4.4.pl2/bin/vasp_std".split(" ")
    p = subprocess.Popen(cmd, cwd=path)


if __name__ == "__main__":
    run_name = "no_rot".lower()
    run_name = run_name.replace("_", "-")
    script_list = []
    for subdir, dirs, files in os.walk(
        f"{PATH}/vasp"
    ):
        for name in files:
            if name == "INCAR":
                script_list.append(os.path.join(subdir))
        for name in files:
            if name == "OUTCAR":
                script_list.remove(os.path.join(subdir))
    count = 0
    for script in script_list:
        if count == MAX_CALCS:
            break
        randomid = os.path.basename(script)
        randomid = randomid.replace("_", "-")

        # This is the launch command for VASP, you need VASP LICENSE to be able to run this
        # Code can be adapted for slurm or other job schedulers
        path = f"cd {script} && mpirun -np 8 /opt/vasp.5.4.4.pl2/bin/vasp_std"
        with open("run.sh", "w") as f:
            f.write(path)
        f.close()
        p = subprocess.Popen(["bash", "run.sh"])
        p.wait()
        count += 1

