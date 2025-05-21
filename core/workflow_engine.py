import json
from core.workflow_context import WorkflowContext
from core.activity_factory import ActivityFactory

def load_workflow(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

class WorkflowEngine:
    def __init__(self, workflow_def: dict):
        self.workflow = workflow_def
        self.mode = workflow_def.get("type", "sequential")

    def run(self, parameters: dict) -> dict:
        ctx = WorkflowContext(parameters)

        if self.mode != "sequential":
            raise NotImplementedError(f"Workflow type '{self.mode}' not supported")

        for act_def in self.workflow.get('activities', []):
            activity = ActivityFactory.create(act_def)
            activity.execute(ctx)

        return ctx.results
