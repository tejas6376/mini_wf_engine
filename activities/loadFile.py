from core.activity import Activity

class LoadFileActivity(Activity):
    def execute(self, context):
        raw_in  = self.definition.get('input_key')
        raw_out = self.definition.get('path') or context.parameters.get('output_path')
        input_key  = context.resolve(raw_in)
        out_path   = context.resolve(raw_out)

        df = context.results.get(input_key)
        if df is None:
            raise ValueError(f"No data for key '{input_key}'")

        df.to_csv(out_path, index=False)
        context.results[self.definition.get('name')] = {'output_path': out_path}
