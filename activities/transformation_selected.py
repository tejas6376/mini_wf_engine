import pandas as pd
from core.activity import Activity

class TransformationActivityOLD(Activity):
    def execute(self, context):
        raw_in  = self.definition.get('input_key')
        raw_out = self.definition.get('output_key') or self.definition.get('name')
        input_key  = context.resolve(raw_in)
        output_key = context.resolve(raw_out)
        transform_type = self.definition.get("transformation_type")

        df = context.results.get(input_key)
        if df is None:
            raise ValueError(f"No input found for key: {input_key}")

        transform_func = self.TRANSFORMATIONS.get(transform_type)
        if not transform_func:
            raise ValueError(f"Unsupported transformation_type: {transform_type}")

        df_transformed = transform_func(df)
        #context.set(output_key, df_transformed)
        context.results[output_key] = df_transformed

    @staticmethod
    def filter_rows(df):
        return df[df[df.columns[0]] > 0]  # Example: filter where first column > 0

    @staticmethod
    def select_columns(df):
        return df.iloc[:, :2]  # Example: select first 2 columns

    @staticmethod
    def sort_rows(df):
        return df.sort_values(by=df.columns[0])

    @staticmethod
    def create_new_columns(df):
        df['new_col'] = df.select_dtypes(include='number').sum(axis=1)
        return df

    @staticmethod
    def aggregate(df):
        return df.groupby(df.columns[0]).sum()

    @staticmethod
    def group_by(df):
        return df.groupby(df.columns[0]).mean()

    @staticmethod
    def normalize(df):
        numeric = df.select_dtypes(include='number')
        df[numeric.columns] = (numeric - numeric.min()) / (numeric.max() - numeric.min())
        return df

    @staticmethod
    def standardize(df):
        numeric = df.select_dtypes(include='number')
        df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
        return df

    @staticmethod
    def binning(df):
        df['binned'] = pd.cut(df[df.columns[0]], bins=3)
        return df

    @staticmethod
    def one_hot_encoding(df):
        return pd.get_dummies(df)

    @staticmethod
    def label_encoding(df):
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].astype('category').cat.codes
        return df

    @staticmethod
    def smooth(df):
        return df.rolling(window=2).mean().fillna(method='bfill')

    @staticmethod
    def impute(df):
        return df.fillna(df.mean(numeric_only=True))

    @staticmethod
    def handle_outliers(df):
        numeric = df.select_dtypes(include='number')
        for col in numeric.columns:
            upper = df[col].quantile(0.95)
            lower = df[col].quantile(0.05)
            df[col] = df[col].clip(lower, upper)
        return df

    @staticmethod
    def log_transform(df):
        return df.applymap(lambda x: pd.np.log(x + 1) if pd.api.types.is_numeric_dtype(type(x)) else x)

    @staticmethod
    def sqrt_transform(df):
        return df.applymap(lambda x: pd.np.sqrt(x) if isinstance(x, (int, float)) and x >= 0 else x)

    @staticmethod
    def power_transform(df):
        return df.applymap(lambda x: x ** 0.5 if isinstance(x, (int, float)) else x)

    @staticmethod
    def reciprocal(df):
        return df.applymap(lambda x: 1 / x if isinstance(x, (int, float)) and x != 0 else x)

    @staticmethod
    def square(df):
        return df.applymap(lambda x: x ** 2 if isinstance(x, (int, float)) else x)

    @staticmethod
    def generalize(df):
        df['general'] = 'grouped'
        return df

    @staticmethod
    def construct_attribute(df):
        df['constructed'] = 1
        return df

    @staticmethod
    def pivot(df):
        return df.pivot_table(index=df.columns[0], aggfunc='mean')

    @staticmethod
    def unpivot(df):
        return df.melt()

    @staticmethod
    def remove_duplicates(df):
        return df.drop_duplicates()

    @staticmethod
    def convert_dtype(df):
        return df.convert_dtypes()

    TRANSFORMATIONS = {
        "filter_rows": filter_rows.__func__,
        "select_columns": select_columns.__func__,
        "sort_rows": sort_rows.__func__,
        "create_new_columns": create_new_columns.__func__,
        "aggregate": aggregate.__func__,
        "group_by": group_by.__func__,
        "normalize": normalize.__func__,
        "standardize": standardize.__func__,
        "binning": binning.__func__,
        "one_hot_encoding": one_hot_encoding.__func__,
        "label_encoding": label_encoding.__func__,
        "smooth": smooth.__func__,
        "impute": impute.__func__,
        "handle_outliers": handle_outliers.__func__,
        "log_transform": log_transform.__func__,
        "sqrt_transform": sqrt_transform.__func__,
        "power_transform": power_transform.__func__,
        "reciprocal": reciprocal.__func__,
        "square": square.__func__,
        "generalize": generalize.__func__,
        "construct_attribute": construct_attribute.__func__,
        "pivot": pivot.__func__,
        "unpivot": unpivot.__func__,
        "remove_duplicates": remove_duplicates.__func__,
        "convert_dtype": convert_dtype.__func__,
    }