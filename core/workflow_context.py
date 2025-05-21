import re

class WorkflowContext:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.results = {}

    def resolve(self, value):
        """
        Replace ${parameters.key} in strings with actual parameter values.
        """
        if not isinstance(value, str):
            return value

        def _repl(match):
            key = match.group(1)
            if key in self.parameters:
                return str(self.parameters[key])
            raise KeyError(f"Parameter '{key}' not found")

        return re.sub(r"\$\{parameters\.([^}]+)\}", _repl, value)
