from core.activity import Activity

class TransformActivity(Activity):
    def execute(self, context):
        raw_in  = self.definition.get('input_key')
        raw_out = self.definition.get('output_key') or self.definition.get('name')
        input_key  = context.resolve(raw_in)
        output_key = context.resolve(raw_out)

        df = context.results.get(input_key)
        if df is None:
            raise ValueError(f"No data for key '{input_key}'")

        df_transformed = df.iloc[:-1]  # drop last row
        context.results[output_key] = df_transformed
