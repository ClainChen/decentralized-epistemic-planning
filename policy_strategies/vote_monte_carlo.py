from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Agent, Action, ProblemType
import util
import logging
from policy_strategies import random as ps_random
import math

LOGGER_LEVEL = logging.DEBUG
EXPLORATION_RATE = 1
ITERATION_LIMIT = 20
STEP_LIMIT = 20
SIMULATE_MULTIPLER = 3

class VoteMonteCarlo(AbstractPolicyStrategy):
    """
    This MonteCarlo will self simulate multiple times to format a voting result.
    
    Strategy based on monte carlo\n
    1. Selection
    2. Expansion
    3. Simulation
    4. Backpropagation
    """
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=logger_level)
        self.simulation_strategy = ps_random.Random(handler, logger_level=LOGGER_LEVEL)
        self.root: Node = None

    def get_policy(self, model: Model, agent_name: str) -> Action:
        successors = model.get_agent_successors(agent_name)
        print(f"Successors for {agent_name}: {[succ.name for succ in successors]}")
        if len(successors) > 1:
            vote = self.voting(model, agent_name)
            succ_name = ""
            min_score = float("inf")
            for name, votes in vote.items():
                if votes[1] > 0:
                    score = votes[0] / votes[1]
                    if score < min_score:
                        min_score = score
                        succ_name = name
            action = next(succ for succ in successors if succ.name == succ_name)
            return action if util.is_valid_action(model.ontic_functions, action) else None
        elif len(successors) == 1:
            return successors[0]
        else:
            return None

    def voting(self, model: Model, agent_name: str) -> dict[str, int]:
        vote = {}
        successors = model.get_agent_successors(agent_name)
        for succ in successors:
            vote[succ.name] = [0, 0]
        """
        re-show the regualr process of model with monte carlo search
        """
        sim_times = len(successors) * SIMULATE_MULTIPLER
        agent_count = len(model.agents)
        for i in range(sim_times):
            sim_model = util.generate_virtual_model(model, agent_name)
            agent_index = sim_model.get_agent_index_by_name(agent_name)
            start_move = ""
            moves = 0
            # self.logger.debug(f"{sim_model}")
            while not sim_model.full_goal_complete():
                sim_agent = sim_model.agents[agent_index].name
                sim_model.observe_and_update_agent(sim_agent)
                sim_succ = sim_model.get_agent_successors(sim_agent)
                if len(sim_succ) > 1:
                    action = self.search(sim_model, sim_agent)
                elif len(sim_succ) == 1:
                    action = sim_succ[0]
                else:
                    action = None
                if moves == 0:
                    start_move = action.name
                if util.is_valid_action(sim_model.ontic_functions, action):
                    sim_model.do_action(sim_agent, action)
                else:
                    sim_model.do_action(sim_agent, None)
                moves += 1
                agent_index = (agent_index + 1) % agent_count
            if start_move != "":
                vote[start_move][0] += moves
                vote[start_move][1] += 1
            # self.logger.debug(f"{agent_name}: {start_move}")
            print(f"Simulate Time: {i}, {vote}")
        return vote

    def search(self, model: Model, agent_name: str):
        self.root = Node(model, agent_name)

        for i in range(ITERATION_LIMIT):
            node = self.select_node(self.root, agent_name)
            reward = self.simulate(node.model, agent_name)
            self.back_propagate(node, reward)
        
        # for node in self.root.children:
            # print(f"{node.last_action.name}, visits: {node.visits}, value: {node.value}")
        action = self.get_best_action(self.root)
        # print(f"Best action: {action.name}")
        return self.get_best_action(self.root)
    
    def select_node(self, node: 'Node', agent_name: str):
        while not node.model.agent_goal_complete(agent_name):
            if not node.is_fully_expanded():
                return self.expand(node, agent_name)
            else:
                node = node.select_child()
        return node

    def expand(self, node: 'Node', agent_name: str):
        action = node.untried_action.pop()
        return node.add_child(agent_name, action, self.simulation_strategy)
    
    def simulate(self, model: Model, agent_name: str):
        simulate_model = model.copy()
        current_agent_index = model.get_agent_index_by_name(agent_name)
        agents_count = len(model.agents)

        reward = 1
        steps = 0
        while ((
            (simulate_model.problem_type == ProblemType.COOPERATIVE
                and not simulate_model.full_goal_complete()
            )
            or 
            (simulate_model.problem_type == ProblemType.NEUTRAL
                and not simulate_model.agent_goal_complete(current_agent_name)
            ))
            and steps < STEP_LIMIT):
            current_agent_name = simulate_model.agents[current_agent_index].name
            simulate_model.observe_and_update_agent(current_agent_name)
            action = self.simulation_strategy.get_policy(simulate_model, current_agent_name)
            actions = simulate_model.get_agent_successors(current_agent_name)
            simulate_model.do_action(current_agent_name, action)
            # TODO: 这里也许要添加一个intention prediction的步骤
            current_agent_index = (current_agent_index + 1) % agents_count
            steps += 1
        reward = reward / max(math.ceil(steps / agents_count), 1)
        return reward
            
    def back_propagate(self, node: 'Node', reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def get_best_action(self, node: 'Node'):
        return max(node.children, key=lambda node: node.get_uct_score()).last_action

class Node:
    def __init__(self, model: Model, agent_name: Agent, action: Action = None, parent: 'Node' = None):
        self.model: Model = model.copy()
        self.parent: Node = parent
        self.children: list[Node] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.last_action = action
        self.untried_action: list[Action] = self.model.get_agent_successors(agent_name)

    def show_info(self):
        return f"last_action: {self.last_action.name if self.last_action is not None else None}, visits: {self.visits}, value: {self.value}"

    def is_fully_expanded(self) -> bool:
        return len(self.untried_action) == 0

    def get_uct_score(self) -> float:
        if self.visits == 0:
            return float('inf')
        else:
            return (self.value / self.visits) + EXPLORATION_RATE * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self) -> 'Node':
        selected_child =  max(self.children, key=lambda node: node.get_uct_score())
        return selected_child
    
    def add_child(self, agent_name: str, action: Action, strategy: AbstractPolicyStrategy):
        new_model = self.model.copy()

        agent_index = new_model.get_agent_index_by_name(agent_name)
        agent_count = len(new_model.agents)
        next_agent_index = (agent_index + 1) % agent_count

        new_model.do_action(agent_name, action)
        while next_agent_index != agent_index:
            next_agent_name = new_model.agents[next_agent_index].name
            next_action = strategy.get_policy(new_model, next_agent_name)
            new_model.do_action(next_agent_name, next_action)
            next_agent_index = (next_agent_index + 1) % agent_count

        child_node = Node(new_model, new_model.agents[agent_index].name, action=action, parent=self)
        self.children.append(child_node)
        return child_node