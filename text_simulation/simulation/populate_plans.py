from text_simulation.simulation.generate import Simulation
import os
from pathlib import Path
import time

project_dir = Path(__file__).parent.parent.as_posix()

def main():
    sim = Simulation("experiment/domain")

    for time_step in sim:
        if time_step["type"] != "goal":
            continue
        print("Planning for time step", time_step["time"])
        problem_file = time_step['problem_path']
        plan_file = os.path.join(Path(problem_file).parent, "true_plan.pddl")
        
        start_time = time.time()
        os.system(f"sudo docker run --rm -v {project_dir}:/root/experiments lapkt/lapkt-public ./siw-then-bfsf " + \
            f"--domain /root/experiments/{sim.domain_path} " + \
            f"--problem /root/experiments/{time_step['problem_path']} " + \
            f"--output /root/experiments/{plan_file} " + \
            f"> /dev/null")
        duration = time.time() - start_time
        with open(os.path.join(Path(problem_file).parent, "true_plan.time"), "w") as f:
            f.write(str(duration))
        
    print("Finished planning")

if __name__ == "__main__":
    main()