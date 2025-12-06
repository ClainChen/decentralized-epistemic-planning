from abstracts import AbstractRules
from dep.epistemic_class import Model, Function, Condition
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG


class LightBulbRules(AbstractRules):
    def check_functions(self, functions: list[Function]):
        """
        the button state needs to consistent with the light state in a certain logic
        (light_state ?l) = (button_light_state ?b ?bs) where (connected ?b) = ?l and (button_state ?b) = ?bs
        """
        button_light_state = {}
        connected = {}
        light_state = {}
        button_state = {}

        for func in functions:
            if func.name == 'button_light_state':
                button = func.parameters['?b']
                if button not in button_light_state:
                    button_light_state[button] = dict()
                button_light_state[button][func.parameters['?bs']] = func.value
            elif func.name == 'connected':
                connected[func.parameters['?b']] = func.value
            elif func.name == 'light_state':
                light_state[func.parameters['?l']] = func.value
            elif func.name == 'button_state':
                button_state[func.parameters['?b']] = func.value

        for v in button_light_state.values():
            vs = list(v.values())
            if len(vs) != len(set(vs)):
                return False

        vs = list(connected.values())
        if len(vs) != len(set(vs)):
            return False

        for b, l in connected.items():
            ls = light_state[l]
            bs = button_state[b]
            bls = button_light_state[b][bs]
            if ls != bls:
                return False

        return True
