from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action
import random
import util
import logging
import threading

LOGGER_LEVEL = logging.DEBUG
SIMULATE_TIMES = 20
MAX_RANDOM_MOVES = 50
class VoteRandom(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=logger_level)

    def get_policy(self, model: Model, agent_name: str) -> Action:
        successors = model.get_agent_successors(agent_name)
        successors = [succ for succ in successors if util.is_valid_action(model, succ, agent_name)]
        if len(successors) > 1:
            votes = self.voting(model, agent_name)
            best_action = ""
            least_vote = float('inf')
            for name, vote in votes.items():
                if vote < least_vote:
                    best_action = name
                    least_vote = vote
            return next(succ for succ in successors if succ.name == best_action)
        elif len(successors) == 1:
            return successors[0]
        else:
            return None
    
    def voting(self, model: Model, agent_name: str) -> dict[str, int]:
        vote = {}
        successors = model.get_agent_successors(agent_name)
        """
        re-show the regualr process of model with monte carlo search
        """
        agent_count = len(model.agents)
        for succ in successors:
            vote[succ.name] = 0
            start_move = succ.name
            moves = [0]
            lock = threading.Lock()
            threads = []
            for i in range(SIMULATE_TIMES):
                sim_model = util.generate_virtual_model(model, agent_name)
                sim_model = random.choice(sim_model)
                sim_model.do_action(succ)
                agent_index = sim_model.get_agent_index_by_name(agent_name)
                agent_index = (agent_index + 1) % agent_count
                t = threading.Thread(target=self.simulate, args=(sim_model, agent_index, moves, lock))
                threads.append(t)
            
            for t in threads:
                t.start()

            for t in threads:
                t.join()
            
            vote[start_move] = moves[0] / SIMULATE_TIMES
        return vote

    def simulate(self, model: Model, agent_index, moves, lock):
        this_moves = 0
        agent_count = len(model.agents)
        while not model.full_goal_complete() and this_moves <= MAX_RANDOM_MOVES:
            sim_agent = model.agents[agent_index].name
            sim_succ = model.get_agent_successors(sim_agent)
            action = random.choice(sim_succ) if sim_succ else None
            model.move(sim_agent.name, action)
            this_moves += 1
            agent_index = (agent_index + 1) % agent_count
        with lock:
            moves[0] += this_moves