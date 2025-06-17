# import pddl_model
import logging
import typing

from .util import setup_logger

from .util import Condition, EP_formula, EPFType, GLOBAL_PERSPECTIVE_INDEX
from .util import eq_ternay_dict, Ternary, ConditionOperatorType
from .util import EpistemicGroupType, EpistemicType, ep_type_dict
from .util import evaluation, compareTernary, bool2Ternary_dict, format_JPstr2PerspectiveKey, str_replace_last
from .util import special_value

LOGGER_NAME = "epistemic_model"
LOGGER_LEVEL = logging.INFO
# LOGGER_LEVEL = logging.DEBUG
# from util import ActionList2DictKey,GLOBAL_PERSPECTIVE_INDEX, ROOT_NODE_ACTION
# from util import raiseNotDefined,eval_var_from_str,Queue
# PRE_INIT_PDICT_KEY = ActionList2DictKey([])

EQ_PREFIX = "((?:\$|\+|\-) [a-z]* \[[a-z0-9,]*\] )"


class EpistemicModel:

    def __init__(self, handlers, entities, functions, function_schemas, external):
        self.logger = setup_logger(LOGGER_NAME, handlers, logger_level=LOGGER_LEVEL)
        self.logger.info("initialize epistemic model")
        # self.logger = setup_logger(LOGGER_NAME,handlers,logger_level=LOGGER_LEVEL) 
        self.entities = entities
        self.functions = functions
        self.function_schemas = function_schemas
        self.external = external
        self.goal_p_keys = None
        self.pre_p_keys = None
        self.all_p_keys = list()
        self.common_iteration_list = list()

    def epistemicConditionsHandler(self, epistemic_condition_dict: typing.Dict[str, Condition], path, p_dict):
        self.logger.debug('epistemicConditionHandler')
        # self.logger.debug('prefix: [%s]',prefix)
        # action_list = [a for s,a in path]
        state_sequence = [s for s, a in path]
        # self.logger.debug(action_list)
        # old_actions_str = ActionList2DictKey(action_list=action_list[:-1])
        # actions_str = ActionList2DictKey(action_list=action_list)
        # self.logger.debug("actions_str [%s], old_actions_str [%s]",actions_str,old_actions_str)
        result_dict = dict()

        for key, condition in epistemic_condition_dict.items():
            self.logger.debug("%s: %s", key, condition)
            ep_formula: EP_formula = condition.condition_formula
            operator = condition.condition_operator
            target_value = condition.target_value
            if ep_formula.epf_type == EPFType.EP:
                eq_str = ep_formula.ep_query
                condition = ep_formula.ep_varphi
                output = self.eval_eq(eq_str, condition, GLOBAL_PERSPECTIVE_INDEX, state_sequence, p_dict)
                if operator == ConditionOperatorType.EQUAL:
                    result = output == target_value
                elif operator == ConditionOperatorType.NOT_EQUAL:
                    result = not output == target_value
                result_dict[key] = result

        for key, condition in epistemic_condition_dict.items():
            ep_formula: EP_formula = condition.condition_formula
            operator = condition.condition_operator
            target_value = condition.target_value
            if target_value == None:
                target_value = state_sequence[-1][condition.target_variable]
            if ep_formula.epf_type == EPFType.JP:
                jp_str = ep_formula.ep_query
                variable_name = ep_formula.ep_variable
                p_key = format_JPstr2PerspectiveKey(jp_str)
                if p_key in p_dict.keys():
                    local_perspective = p_dict[p_key]
                    local_state = local_perspective[-1]


                else:
                    self.generate_ps_from_jp_query(jp_str, GLOBAL_PERSPECTIVE_INDEX, state_sequence, p_dict)
                    if not p_key in p_dict.keys():
                        raise ValueError("p_key is not generated correctly", p_key, condition, p_dict.keys())
                    local_perspective = p_dict[p_key]
                    local_state = local_perspective[-1]
                    # p = self.generate_os(state_sequence,GLOBAL_PERSPECTIVE_INDEX)
                    # p_dict[p_key] = p
                if not variable_name in local_state.keys():
                    raise ValueError("variable_name is not in the local state", variable_name, local_state)
                else:
                    value2 = local_state[variable_name]
                    result_dict[key] = evaluation(self.logger, operator, value2, target_value) == Ternary.TRUE
        # self.logger.debug(result_dict)
        # self.logger.debug(p_dict)
        return result_dict, p_dict

    def epistemicEffectHandler(self, epistemic_effect_dict: typing.Dict[str, EP_formula], path, p_dict):
        result_dict = dict()
        state_sequence = [s for s, a in path]

        for jp_name, jp_item in epistemic_effect_dict.items():
            if not jp_item.epf_type == EPFType.JP:
                raise ValueError("action effect should only be a JP", jp_item)
            jp_str = jp_item.ep_query
            variable_name = jp_item.ep_variable
            p_key = format_JPstr2PerspectiveKey(jp_str)
            if p_key in p_dict.keys():
                local_perspective = p_dict[p_key]
                local_state = local_perspective[-1]
            else:
                self.generate_ps_from_jp_query(jp_str, GLOBAL_PERSPECTIVE_INDEX, state_sequence, p_dict)
                if not p_key in p_dict.keys():
                    raise ValueError("p_key is not generated correctly", p_key, p_dict.keys())
                local_perspective = p_dict[p_key]
                local_state = local_perspective[-1]
                # p = self.generate_os(state_sequence,GLOBAL_PERSPECTIVE_INDEX)
                # p_dict[p_key] = p
            if not variable_name in local_state.keys():
                raise ValueError("variable_name is not in the local state", variable_name, local_state)
            else:
                value2 = local_state[variable_name]
                result_dict[jp_name] = value2
        return result_dict, p_dict

    def generate_ps_from_jp_query(self, jp_str, parent_prefix, state_sequence, input_ps):
        query_content_list = jp_str.split(" ")
        self.logger.debug(query_content_list)
        if not len(query_content_list) >= 2:
            raise ValueError("eq_query is not in the correct format, it should contains at least two elements",
                             query_content_list)
        jp_prefix = query_content_list[0]
        if not jp_prefix in ep_type_dict.keys():
            raise ValueError("jp_prefix is not in the correct format", jp_prefix)
        jp_group_type, jp_type = ep_type_dict[jp_prefix]
        agent_id_str = query_content_list[1]
        rest_jp_str = jp_str[len(jp_prefix) + len(agent_id_str) + 2:]
        if jp_group_type == EpistemicGroupType.Individual:
            if not agent_id_str.startswith("[") or not agent_id_str.endswith("]"):
                raise ValueError("agent_id is not in the correct format (should cover with [])", jp_str)
            agent_id_str = agent_id_str[1:-1]
            if "," in agent_id_str:
                raise ValueError("agent_id is not in the correct format (Should only contain one agent based on b/s/k)",
                                 agent_id_str, jp_str)
            agent_index = agent_id_str
            # new_p = self.generate_os(state_sequence,agent_index)
            self.handle_ps_from_jp_single(jp_type, agent_index, rest_jp_str, parent_prefix, state_sequence, input_ps)

            return
        elif jp_group_type == EpistemicGroupType.Uniform:
            pass
        elif jp_group_type == EpistemicGroupType.Distribution:
            pass
        elif jp_group_type == EpistemicGroupType.Common:
            pass
        else:
            raise ValueError("Wrong JP group type")

    def generate_ps(self, os_key_str, os, parent_ps, input_ps):
        ps_key_str = str_replace_last(os_key_str, "o ", "f ")
        if ps_key_str in input_ps.keys():
            return input_ps[ps_key_str], ps_key_str

        if ps_key_str in input_ps.keys():
            return
        ps = []
        for i in range(len(os)):
            temp_ps = self.generate_p(os[:i + 1], parent_ps[:i + 1])
            ps.append(temp_ps)
        input_ps[ps_key_str] = ps
        return ps, ps_key_str

    def generate_p(self, partial_os, partial_ps):
        new_state = partial_os[-1].copy()
        for key, value in new_state.items():
            if value == special_value.UNSEEN:
                # new_state[key] = self.(agent_id,key,partial_os,partial_ps)

                ts = self.identify_last_seen_timestamp(key, partial_os)
                new_state[key] = self.retrieval_function(partial_ps, ts, key)
                if key == 'dir b':
                    self.logger.debug("partial_os: %s", partial_os)
                    self.logger.debug("partial_ps: %s", partial_ps)
                    self.logger.debug("ts: %s", ts)
                    # self.logger.debug("new_state: %s",new_state)
                    self.logger.debug("key: %s", key)
                    self.logger.debug("value: %s", new_state[key])
                    # self.logger.debug(partial_ps)
                    # self.logger.debug(ts)
                    # self.logger.debug(new_state[key])
        return new_state

    def retrieval_function(self, partial_ps, ts, variable_name):
        temp_ts = ts
        if temp_ts < 0:
            return special_value.HAVENT_SEEN
        while temp_ts >= 0:
            if variable_name in partial_ps[temp_ts].keys():
                if partial_ps[temp_ts][variable_name] == special_value.HAVENT_SEEN:
                    temp_ts += -1
                elif partial_ps[temp_ts][variable_name] == special_value.UNSEEN:
                    raise ValueError("variable is not seen by the agent, should not happen", variable_name, partial_ps,
                                     temp_ts)
                else:
                    return partial_ps[temp_ts][variable_name]
            else:
                raise ValueError("variable is not in the observation list", variable_name, partial_ps, ts)

        temp_ts = ts + 1
        while temp_ts < len(partial_ps):
            if variable_name in partial_ps[temp_ts].keys():
                if partial_ps[temp_ts][variable_name] == special_value.HAVENT_SEEN:
                    temp_ts += 1
                elif partial_ps[temp_ts][variable_name] == special_value.UNSEEN:
                    raise ValueError("variable is not seen by the agent, should not happen", variable_name, partial_ps,
                                     temp_ts)
                else:
                    return partial_ps[temp_ts][variable_name]
            else:
                raise ValueError("variable is not in the observation list", variable_name, partial_ps, ts)
        return special_value.HAVENT_SEEN

    def identify_last_seen_timestamp(self, variable_name, partial_os):
        length = len(partial_os)
        for i in range(length):
            ts = length - i - 1
            if variable_name in partial_os[ts].keys():
                if not partial_os[ts][variable_name] == special_value.UNSEEN:
                    return ts
            else:
                raise ValueError("variable is not in the observation list", variable_name, ts)
        return -1

    def handle_ps_from_jp_single(self, jp_type, agent_index, rest_eq_str, parent_prefix, state_sequence, input_ps):
        # os_key_str = parent_prefix + "o ["+agent_index + "] "
        # if os_key_str in input_ps.keys():
        #     os = input_ps[os_key_str]
        # else:
        #     
        #     input_ps[os_key_str] = os
        os, os_key_str = self.generate_os(agent_index, parent_prefix, state_sequence, input_ps)

        if jp_type == EpistemicType.Knowledge or jp_type == EpistemicType.Seeing:
            if rest_eq_str == '':
                return
            else:
                self.generate_ps_from_jp_query(
                    jp_str=rest_eq_str,
                    parent_prefix=os_key_str,
                    state_sequence=os,
                    input_ps=input_ps)
                return

        elif jp_type == EpistemicType.Belief:
            ps, ps_key_str = self.generate_ps(os_key_str, os, state_sequence, input_ps)
            if rest_eq_str == "":
                return
            else:
                self.generate_ps_from_jp_query(
                    jp_str=rest_eq_str,
                    parent_prefix=ps_key_str,
                    state_sequence=ps,
                    input_ps=input_ps)
            # need to generate ps based on os
            pass
        else:
            raise ValueError("jp_type is not defined yet", jp_type)

    def eval_eq(self, eq_query_str, condition, parent_prefix, state_sequence, input_ps):
        eq_query_content_list = eq_query_str.split(" ")
        self.logger.debug(eq_query_content_list)
        if not len(eq_query_content_list) >= 3:
            raise ValueError("eq_query is not in the correct format, it should contains at least three elements",
                             eq_query_str)

        eq_ternary_type_str = eq_query_content_list[0]
        if not eq_ternary_type_str in eq_ternay_dict.keys():
            raise ValueError("eq_ternary_type is not in the correct format", eq_ternary_type_str)
        eq_ternary_type = eq_ternay_dict[eq_ternary_type_str]
        ep_prefix_str = eq_query_content_list[1]
        if not ep_prefix_str in ep_type_dict.keys():
            raise ValueError("ep_type is not in the correct format", ep_prefix_str)
        ep_group_type, ep_type = ep_type_dict[ep_prefix_str]
        agent_id_str = eq_query_content_list[2]
        rest_eq_str = eq_query_str[len(eq_ternary_type_str) + len(ep_prefix_str) + len(agent_id_str) + 3:]
        if ep_group_type == EpistemicGroupType.Individual:
            if not agent_id_str.startswith("[") or not agent_id_str.endswith("]"):
                raise ValueError("agent_id is not in the correct format (should cover with [])", eq_query_str)
            agent_id_str = agent_id_str[1:-1]
            if "," in agent_id_str:
                raise ValueError("agent_id is not in the correct format (Should only contain one agent based on b/s/k)",
                                 agent_id_str, eq_query_str)
            agent_index = agent_id_str

            result = self.eval_eq_single_agent(ep_type, agent_index, rest_eq_str, condition, parent_prefix,
                                               state_sequence, input_ps)
            self.logger.debug(result)
            return compareTernary(eq_ternary_type, result)
            # return self.eval_eq_in_ps(eq_query_str,new_prefix, parent_prefix, state_sequence)
        elif ep_group_type == EpistemicGroupType.Uniform:
            pass
        elif ep_group_type == EpistemicGroupType.Distribution:
            pass
        elif ep_group_type == EpistemicGroupType.Common:
            pass
        else:
            raise ValueError("Wrong ep group type")

    def eval_eq_single_agent(self, ep_type, agent_index, rest_eq_str, condition: Condition, parent_prefix,
                             state_sequence, input_ps):

        # os_key_str = parent_prefix + "o ["+agent_index + "] "
        # if os_key_str in input_ps.keys():
        #     os = input_ps[os_key_str]
        # else:
        #     os = self.generate_os(state_sequence,agent_index)
        #     input_ps[os_key_str] = os
        os, os_key_str = self.generate_os(agent_index, parent_prefix, state_sequence, input_ps)

        if ep_type == EpistemicType.Knowledge:
            if rest_eq_str == '':
                # this is the last level, we need to evaluate the condition
                operator = condition.condition_operator
                variable1_name = condition.condition_variable
                if variable1_name in os[-1].keys():
                    value1 = os[-1][variable1_name]
                else:
                    raise ValueError("variable is not in the observation list", variable1_name, os[-1])
                variable2_name = condition.target_variable
                if variable2_name is None:
                    value2 = condition.target_value
                else:
                    value2 = os[-1][variable2_name]
                return evaluation(self.logger, operator, value1, value2)
            else:
                # it means there is still nesting query
                return self.eval_eq(rest_eq_str, condition, os_key_str, os, input_ps)
        elif ep_type == EpistemicType.Seeing:
            if rest_eq_str == '':
                # this is the last level, we need to evaluate the condition
                operator = condition.condition_operator
                variable1_name = condition.condition_variable
                return bool2Ternary_dict[variable1_name in os[-1].keys()]
            else:
                # it means there is still nesting query
                return bool2Ternary_dict[
                    self.eval_eq(rest_eq_str, condition, os_key_str, os, input_ps) != Ternary.UNKNOWN]

        elif ep_type == EpistemicType.Belief:
            ps, ps_key_str = self.generate_ps(os_key_str, os, state_sequence, input_ps)
            if rest_eq_str == '':
                # this is the last level, we need to evaluate the condition
                operator = condition.condition_operator
                variable1_name = condition.condition_variable
                if variable1_name in ps[-1].keys():
                    value1 = ps[-1][variable1_name]
                else:
                    raise ValueError("variable is not in the perspective list", variable1_name, os[-1])
                variable2_name = condition.target_variable
                if variable2_name == None:
                    value2 = condition.target_value
                else:
                    value2 = ps[-1][variable2_name]
                return evaluation(self.logger, operator, value1, value2)
            else:
                return self.eval_eq(rest_eq_str, condition, ps_key_str, ps, input_ps)
        else:
            raise ValueError("ep_type is not defined yet", ep_type)

    def generate_os(self, agent_index, parent_prefix, ps, input_ps):

        os_key_str = parent_prefix + "o [" + agent_index + "] "
        if os_key_str in input_ps.keys():
            os = input_ps[os_key_str]
        else:
            os = list()
            for state in ps:
                new_state = self.get1o(state, agent_index)
                os.append(new_state)
            input_ps[os_key_str] = os
        return os, os_key_str

    def get1o(self, parent_state, agt_id):
        new_state = dict()
        for var_index, value in parent_state.items():
            if self.external.checkVisibility(parent_state, agt_id, var_index, self.entities, self.functions,
                                             self.function_schemas):
                new_state.update({var_index: value})
            else:
                new_state.update({var_index: special_value.UNSEEN})

        return new_state
