from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action
import random
import util
import logging

LOGGER_LEVEL = logging.DEBUG
SIMULATE_TIMES = 100
MAX_RANDOM_MOVES = 100
class VoteRandom(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=logger_level)

    def get_policy(self, model: Model, agent_name: str) -> Action:
        successors = model.get_agent_successors(agent_name)
        if len(successors) > 1:
            votes = self.voting(model, agent_name)
            best_action = ""
            least_vote = float('inf')
            for name, vote in votes.items():
                if vote < least_vote:
                    best_action = name
                    least_vote = vote
            action = next(succ for succ in successors if succ.name == best_action)
            return action if util.is_valid_action(model.ontic_functions, action) else None
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
            moves = 0
            for i in range(SIMULATE_TIMES):
                sim_model = util.generate_virtual_model(model, agent_name)
                agent_index = sim_model.get_agent_index_by_name(agent_name)
                this_moves = 0
                while not sim_model.full_goal_complete() and this_moves <= MAX_RANDOM_MOVES:
                    sim_agent = sim_model.agents[agent_index].name
                    sim_model.observe_and_update_agent(sim_agent)
                    sim_succ = sim_model.get_agent_successors(sim_agent)
                    action = random.choice(sim_succ) if sim_succ else None
                    if util.is_valid_action(sim_model.ontic_functions, action):
                        sim_model.do_action(sim_agent, action)
                    else:
                        sim_model.do_action(sim_agent, None)
                    moves += 1
                    this_moves += 1
                    agent_index = (agent_index + 1) % agent_count
            vote[start_move] = moves / SIMULATE_TIMES
        return vote