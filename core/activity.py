from abc import ABC, abstractmethod

class Activity(ABC):
    def __init__(self, definition: dict):
        self.definition = definition

    @abstractmethod
    def execute(self, context):
        """
        Perform the activity using `context`.
        """
        pass
