import pandas as pd
from core.activity import Activity

class ReadFileActivity(Activity):
    def execute(self, context):
        raw_path = self.definition.get('path') or context.parameters.get('input_path')
        path = context.resolve(raw_path)
        df = pd.read_csv(path)
        context.results[self.definition.get('name')] = df
