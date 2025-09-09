from epistemic_handler.epistemic_class import *
import logging
import itertools
import util
import time
from pathlib import Path
from tqdm import tqdm
from string import Template

PROBLEM_BUILDER_LOG_LEVEL = logging.DEBUG
TAP = "        "

class ProblemBuilder:
    def __init__(self, base_model, handlers, log_level=PROBLEM_BUILDER_LOG_LEVEL):
        self.logger = util.setup_logger(__name__, handlers, logger_level=log_level)
        self.base_model: Model = base_model
    
    def get_all_init_ontic_world(self):
        groups = []
        for schema in self.base_model.function_schemas:
            if isinstance(schema.range, tuple):
                start, end = schema.range
                value_range = list(range(start, end + 1))
            else:
                value_range = schema.range
            
            params = [self.base_model.get_all_entity_name_by_type(type) for type in schema.require_parameters.values()]
            # start grouping
            for param in itertools.product(*params):
                group = []
                for value in value_range:
                    function = Function()
                    function.name = schema.name
                    function.value = value
                    function.parameters = dict(zip(schema.require_parameters.keys(), param))
                    group.append(function)
                groups.append(group)
            
        # generate all possible function groups
        all_possible_init = [list(lst) for lst in itertools.product(*groups)]
        all_possible_init = [world for world in all_possible_init if self.base_model.rules.check_functions(world)]
        return all_possible_init

    def get_all_poss_goals(self, agent_name = "") -> list[dict[str, list[Condition]]]:
        """
        if input agent_name, then the generated goals for agent_name will only have it's own goals
        """

        def get_cross_subsets(nested_list):
            # 为每个子列表生成选择（包括空选择）
            choices = [[[]] + [[item] for item in sublist] for sublist in nested_list]
            
            # 生成所有组合，展平并过滤空列表
            return [list(itertools.chain.from_iterable(combo)) 
                    for combo in itertools.product(*choices) 
                    if any(combo)]
        
        # Generate all possible belief sequences
        belief_sequences = self.base_model.possible_belief_sequences

        groups: list[Condition] = []
        for sg in self.base_model.S_G:
            for bs in belief_sequences:
                if bs[0] == agent_name:
                    continue
                bg = Condition()
                bg.belief_sequence = bs
                bg.condition_function_name = sg.condition_function_name
                bg.condition_function_parameters = sg.condition_function_parameters
                bg.condition_operator = sg.condition_operator
                bg.value = sg.value
                groups.append(bg)
        
        s = {}
        for a in self.base_model.agents:
            s[a.name] = []
        for g in groups:
            s[g.belief_sequence[0]].append(g)
        
        for name, goals in s.items():
            s[name] = [list(group) for key, group in 
                                     itertools.groupby(sorted(goals, 
                                                              key=lambda x: f"{x.belief_sequence}{x.header()}"), 
                                                       key=lambda x: f"{x.belief_sequence}{x.header()}")]
        
        for name, goals in s.items():
            s[name] = get_cross_subsets(goals)
        s[agent_name] = [self.base_model.get_agent_by_name(agent_name).own_goals]

        key = list(s.keys())
        value = list(s.values())

        agent_goal_sets = []
        for comb in product(*value):
            agent_goal_sets.append(dict(zip(key, comb)))

        agent_goal_sets.sort(key=lambda x: [j.plain_text() for i in x.values() for j in i ])
        agent_goal_sets.sort(key=lambda x: sum([len(value) for value in x.values()]))
        # for se in agent_goal_sets:
        #     print(se)

        valid = 0
        invalid_jump = 0
        total = len(agent_goal_sets)

        invalid_goal_sets: list[set] = []
        start_time = time.perf_counter()
        
        results = []
        print(f"Total goal settings: {len(agent_goal_sets)}, now begin to test each setting")
        with tqdm(range(total), desc="progress") as pbar:
            max_action_length = -1
            for i in pbar:
                agent_goal_set = agent_goal_sets[i]
                goal_set = set()
                for goals in agent_goal_set.values():
                    for goal in goals:
                        goal_set.add(goal)
                jump= False
                for sett in invalid_goal_sets:
                    if sett.issubset(goal_set): 
                        jump = True
                        invalid_jump += 1
                        break
                if not jump:
                    test_model = self.base_model.copy()
                    for agent in test_model.agents:
                        agent.own_goals = agent_goal_set[agent.name]
                    num_actions = util.check_bfs(test_model, max(12, max_action_length * 2))
                    max_action_length = max(num_actions, max_action_length)
                    if num_actions >= 0 :
                        results.append(agent_goal_set)
                        valid += 1
                    else:
                        invalid_goal_sets.append(goal_set)
                pbar.set_postfix({"Valid Count": f"{valid}/{total}", "Skip invalid test count": invalid_jump})
        
        # for sett in results:
        #     result = "\n"
        #     for agent, goals in sett.items():
        #         result += f"{agent}:\n"
        #         for goal in goals:
        #             result += f"{goal}\n"
        #         result += f"-----\n"
        #     self.logger.debug(result)
        
        return results, -1 if agent_name == "" else (time.perf_counter() - start_time) / max(1, valid)

    def get_all_poss_problem(self, worlds: list[list[Function]], goal_sets: list[dict[str, list[Condition]]]) -> list[Model]:
        # The difference between the problems is that their initial states and agent's goals are different
        # so that are what we needs to change.

        # filter the acceptable goals first
        # now assume if a goal is valid in a world settings, then it is valid in all world settings.
        
        result = []
        for world in worlds:
            for goal_set in goal_sets:
                new_model = self.base_model.copy()
                new_model.ontic_functions = world
                for agent in new_model.agents:
                    agent.own_goals = goal_set[agent.name]
                result.append(new_model)
        
        return result
    
    @util.record_time
    def generate_all_problem_pddl_files(self):
        worlds = self.get_all_init_ontic_world()
        goal_sets, _ = self.get_all_poss_goals()
        problems = self.get_all_poss_problem(worlds, goal_sets)
        self.logger.info(f"Total possible problem num: {len(problems)}")
        print(f"共计 {len(problems)} 个可能的模型")

        # create init template
        init_template_path = Path(util.INIT_TEMPLATE_PATH)
        with open(init_template_path, 'r') as template_file:
            template_content = template_file.read()
        init_template = Template(template_content)
        init_func_template = Template(TAP+"(assign (${func_name} ${func_params}) ${func_value})\n")
        init_range_template = Template(TAP+"(${func_name} ${range_type} ${value_range})\n")
        init_goal_set_template = Template(TAP+"(${func_name} ${goal_type} ${value_range})\n")

        # create agent template
        agt_template_path = Path(util.AGENT_TEMPLATE_PATH)
        with open(agt_template_path, 'r') as template_file:
            template_content = template_file.read()
        agt_template = Template(template_content)
        goal_template = Template(TAP + "(= (@ep (${belief_sequence}) (= (${condition_func_name} ${condition_func_params}) ${condition_func_value})) ep.true)\n")

        # create share info for init files
        domain_name = self.base_model.domain_name
        agents = " ".join([agt.name for agt in self.base_model.agents])
        entities = ""
        objects = defaultdict(list)
        for entity in self.base_model.entities:
            if entity.type != "agent":
                objects[entity.type].append(entity.name)
        for type, objs in objects.items():
            entities += TAP + " ".join(objs) + f" - {type}\n"
        entities = entities.rstrip()
        
        ranges = ""
        viewed_ranges = set()
        for r in self.base_model.ontic_functions:
            if r.name not in viewed_ranges:
                viewed_ranges.add(r.name)
                this_range = f"[{r.range[0]}, {r.range[1]}]" if isinstance(r.range, tuple) else str(r.range)
                ranges += init_range_template.substitute(
                    func_name=r.name,
                    range_type=r.type.__str__(),
                    value_range = this_range
                )
        ranges = ranges.rstrip()

        goal_sets = ""
        for goal_set in self.base_model.S_G:
            if goal_set.type == GoalValueType.RANGE:
                value_range = f"[]{goal_set.min}, {goal_set.max}"
            elif goal_set.type == GoalValueType.ENUMERATE:
                value_range = f"{goal_set.enumerates}"
            else:
                value_range = "[]"
            goal_sets += init_goal_set_template.substitute(
                func_name = goal_set.function_name,
                goal_type = goal_set.type.__str__(),
                value_range = value_range
            )
        goal_sets = goal_sets.rstrip()

        max_belief_depth = self.base_model.max_belief_depth

        num = 1
        for problem in problems:
            # create the problem folder
            new_problem_folder_path = f"{util.MODEL_FOLDER_PATH}{problem.domain_name}/auto_generate/problem_{num}/"
            new_problem_folder_path = Path(new_problem_folder_path)
            
            if not new_problem_folder_path.exists():
                new_problem_folder_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created problem folder: {new_problem_folder_path}")
            else:
                self.logger.info(f"Problem folder already exists: {new_problem_folder_path}")
            
            # generate init pddl file
            problem_name = f"problem_{num}"
            init_functions = ""
            for func in problem.ontic_functions:
                func_name = func.name
                func_params = " ".join(func.parameters.values())
                func_value = func.value
                init_functions += init_func_template.substitute(
                    func_name=func_name,
                    func_params=func_params,
                    func_value=func_value
                )
            init_functions = init_functions.rstrip()
            init_pddl_content = init_template.substitute(
                problem_name = problem_name,
                domain_name = domain_name,
                agents = agents,
                objects = entities,
                init_functions = init_functions,
                ranges = ranges,
                goal_sets = goal_sets,
                max_belief_depth = max_belief_depth
            )


            init_file_name = util.INIT_FILE_NAME
            init_file_path = new_problem_folder_path / init_file_name
            with open(init_file_path, "w") as init_file:
                init_file.write(init_pddl_content)
            
            # generate agent pddl file
            for agt in problem.agents:
                agent_name = agt.name
                goals = ""
                for goal in agt.own_goals:
                    
                    belief_sequence = f"\"{"b " + " b ".join([f"[{b}]" for b in goal.belief_sequence])}\""
                    goals += goal_template.substitute(
                        belief_sequence = belief_sequence,
                        condition_func_name = goal.condition_function_name,
                        condition_func_params = " ".join(goal.condition_function_parameters.values()),
                        condition_func_value = goal.value
                    )
                goals = goals.rstrip()
                
                agt_pddl_content = agt_template.substitute(
                    problem_name = problem_name,
                    domain_name = domain_name,
                    agent_name = agent_name,
                    goals = goals
                )

                agt_file_name = agt.name + util.AGENT_FILE_NAME
                agt_file_path = new_problem_folder_path / agt_file_name
                with open(agt_file_path, 'w') as agt_file:
                    agt_file.write(agt_pddl_content)
            num += 1

    def generate_problem_pddl_file_from_model(self, model: Model):
        pass







