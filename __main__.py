#!/usr/bin/env python3
import os
from core.workflow_engine import load_workflow, WorkflowEngine

if __name__ == "__main__":
    # ensure output folder exists
    os.makedirs("output", exist_ok=True)

    # load and run
    wf = load_workflow("workflow.json")
    results = WorkflowEngine(wf).run(wf["parameters"])
    print("Workflow completed. Results:")
    print(results)
