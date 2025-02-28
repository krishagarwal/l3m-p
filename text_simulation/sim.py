from __future__ import annotations
import signal
import sys
from typing import IO
from contextlib import redirect_stdout
import os
from pathlib import Path
from difflib import ndiff

from .simulation.generate import Simulation

from .kg_planner.age import AgeGraphStore
from knowledge_representation.knowledge_loader import load_knowledge_from_yaml, populate_with_knowledge # type: ignore
from knowledge_representation._libknowledge_rep_wrapper_cpp import LongTermMemoryConduit # type: ignore

from .kg_planner.load_graph import load_graph
from .kg_planner.agent import KGAgent
from .kg_planner.utils import reset_database

AGENT_LABEL = "the_agent"
AGENT_IN_ROOM = "agent_in_room"

# set API key
openai_keys_file = os.path.join(os.getcwd(), "keys/openai_keys.txt")
with open(openai_keys_file, "r") as f:
	keys = f.read()
keys = keys.strip().split('\n')
os.environ["OPENAI_API_KEY"] = keys[0]

project_dir = Path(__file__).parent.parent.as_posix()

class KGSim:
	def __init__(self, sim: Simulation, agent: KGAgent, log_dir: str) -> None:
		self.sim = sim
		self.agent = agent
		self.log_dir = log_dir
		self.report: list[Result] = []
	
	def run(self):
		reset_database(
			dbname="knowledge_base_truth",
			user=self.agent.dbuser,
			password=self.agent.dbpass,
			host=self.agent.dbhost,
			port=self.agent.dbport,
			schema_file=self.agent.dbschema
		)
		all_knowledge = [load_knowledge_from_yaml(self.sim.initial_knowledge_path)]
		populate_with_knowledge(LongTermMemoryConduit("knowledge_base_truth", "localhost"), all_knowledge)
		load_graph("knowledge_base_truth", "knowledge_graph")
		truth_graph_store = AgeGraphStore(
			"knowledge_base_truth",
			"postgres",
			"password",
			"localhost",
			5432,
			"knowledge_graph",
			"entity",
		) # type: ignore

		def get_true_agent_loc(kg: AgeGraphStore) -> str:
			return kg.query(f"MATCH ({{name: '{AGENT_LABEL}'}})-[{AGENT_IN_ROOM}]->(V2) RETURN V2.name")[0][0][1:-1]

		true_agent_loc = get_true_agent_loc(truth_graph_store)
		previous_diff = []
		print(f"Initial State:\n{self.sim.initial_state}")
		self.agent.input_initial_state(self.sim.initial_state, self.sim.initial_knowledge_path, self.sim.predicate_names, self.sim.domain_path)
		
		for time_step in self.sim:
			if time_step["type"] == "state_change":
				print("\nTime: " + str(time_step["time"]))
				print(f"State change: {time_step['state_change']}")
				self.agent.input_state_change(time_step["state_change"])
				print("Completed state change")
			elif time_step["type"] == "goal":
				print("\nTime: " + str(time_step["time"]))
				print(f"Goal: {time_step['goal']}")
				predicted_plan = self.agent.answer_planning_query(time_step["goal"], truth_graph_store)
				true_agent_loc = get_true_agent_loc(truth_graph_store)
				print("Generated plan")

				if "true_plan_pddl" in time_step:
					true_plan = time_step["true_plan_pddl"].splitlines()
				else:
					plan_file = "true_plan.pddl"
					os.system(f"sudo docker run --rm -v {project_dir}:/root/experiments lapkt/lapkt-public ./siw-then-bfsf " + \
						f"--domain /root/experiments/{self.sim.domain_path} " + \
						f"--problem /root/experiments/{time_step['problem_path']} " + \
						f"--output /root/experiments/{plan_file} " + \
						f"> /dev/null")
					with open(os.path.join(project_dir, plan_file), "r") as f:
						true_plan = f.read().splitlines()
					os.remove(os.path.join(project_dir, plan_file))
				
				if true_plan != predicted_plan:
					print("Conflicting expected plan and predicted plan")
					self.report.append(Result(time_step["time"], "plan", time_step["type"], False))
					with open(os.path.join(self.log_dir, f"{time_step['time']:04d}_plan.diff"), "w") as f:
						f.write("\n".join(ndiff(true_plan, predicted_plan)))
				else:
					print("Plan is correct")
					self.report.append(Result(time_step["time"], "plan", time_step["type"], True))
			else:
				continue

			truth_graph_store._conn.close()

			reset_database(
				dbname="knowledge_base_truth",
				user=self.agent.dbuser,
				password=self.agent.dbpass,
				host=self.agent.dbhost,
				port=self.agent.dbport,
				schema_file=self.agent.dbschema
			)
			all_knowledge = [load_knowledge_from_yaml(time_step["knowledge_path"])]
			populate_with_knowledge(LongTermMemoryConduit("knowledge_base_truth", "localhost"), all_knowledge)
			load_graph("knowledge_base_truth", "knowledge_graph")
			truth_graph_store = AgeGraphStore(
				"knowledge_base_truth",
				"postgres",
				"password",
				"localhost",
				5432,
				"knowledge_graph",
				"entity",
			) # type: ignore
			bad_agent_loc = get_true_agent_loc(truth_graph_store)
			truth_graph_store.delete(AGENT_LABEL, AGENT_IN_ROOM, bad_agent_loc)
			truth_graph_store.upsert_triplet(AGENT_LABEL, AGENT_IN_ROOM, true_agent_loc)

			triplets_truth = truth_graph_store.query("MATCH (V)-[R]->(V2) RETURN V.name, type(R), V2.name", return_count=3)
			triplets_truth = [" -> ".join(item[1:-1] for item in row) for row in triplets_truth if all(isinstance(s, str) for s in row)]
			triplets_truth.sort()
			triplets_pred = self.agent.get_all_relations()
			triplets_pred.sort()

			if triplets_pred != triplets_truth:
				diff = list(item for item in ndiff(triplets_truth, triplets_pred) if item[0] != ' ')
				if previous_diff == diff:
					self.report.append(Result(time_step["time"], "state", time_step["type"], True))
					print("Conflicting expected and predicted state (same as last discrepancy)")
				elif set(diff).issubset(set(previous_diff)):
					self.report.append(Result(time_step["time"], "state", time_step["type"], True))
					print("Conflicting expected and predicted state (fixed part of last discrepancy)")
				else:
					self.report.append(Result(time_step["time"], "state", time_step["type"], False))
					print("Conflicting expected and predicted state")
					with open(os.path.join(self.log_dir, f"{time_step['time']:04d}_state.diff"), "w") as f:
						f.write("\n".join(diff))
					with open(os.path.join(self.log_dir, f"{time_step['time']:04d}_state.expected"), "w") as f:
						f.write("\n".join(triplets_truth))
					with open(os.path.join(self.log_dir, f"{time_step['time']:04d}_state.predicted"), "w") as f:
						f.write("\n".join(triplets_pred))
				previous_diff = diff
			else:
				self.report.append(Result(time_step["time"], "state", time_step["type"], True))
				previous_diff = []
				print("Update successful")
			print(f"Total token usage: {self.agent.total_prompt_tokens} prompt tokens | {self.agent.total_completion_tokens} completion tokens | {self.agent.total_llm_tokens} total tokens")
			
		print("\nAll updates/goals processed")
	
	def close(self):
		with open(os.path.join(self.log_dir, "report.txt"), "w") as f:
			f.write("\n".join(str(result) for result in self.report))
		self.agent.close()

class Result:
	def __init__(self, time: int, result_type: str, time_step_type: str, success: bool) -> None:
		self.time = time
		self.result_type = result_type
		self.time_step_type = time_step_type
		self.success = success
	
	def __str__(self) -> str:
		return f"{self.time:04d}|{self.result_type}|{self.time_step_type}|{'Success' if self.success else 'Fail'}"

	@staticmethod
	def from_str(line: str) -> Result:
		args = line.split('|')
		assert len(args) == 4
		return Result(int(args[0]), args[1], args[2], args[3] == "Success")

if __name__ == "__main__":
	class Logger(IO):
		def __init__(self, file_path):
			self.terminal = sys.stdout
			self.log = open(file_path, "w")
		def write(self, message):
			self.terminal.write(message)
			self.log.write(message)  
		def flush(self):
			self.terminal.flush()
			self.log.flush()
		def close(self):
			self.log.close()

	experiment_dir = "experiment"
	domain_path = f"{experiment_dir}/domain"
	run_dir = f"{experiment_dir}/run"
	log = Logger(f"{run_dir}/output.log")

	def cleanup(exit: bool = False):
		sim.close()
		log.close()
		if exit:
			sys.exit(0)
	
	signal.signal(signal.SIGINT, lambda sig, frame: cleanup(True))
	
	with redirect_stdout(log):
		sim = KGSim(Simulation(domain_path), KGAgent(run_dir, True, True, AGENT_LABEL), run_dir)
		try:
			sim.run()
		finally:
			cleanup()