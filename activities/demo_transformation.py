import pandas as pd
import numpy as np
from core.activity import Activity

class TransformationActivity(Activity):
    def execute(self, context):
        raw_in  = self.definition.get('input_key')
        raw_out = self.definition.get('output_key') or self.definition.get('name')
        input_key  = context.resolve(raw_in)
        output_key = context.resolve(raw_out)
        transform_type = self.definition.get("transformation_type")
        transform_params = self.definition.get("transformation_params", {})
        
        df = context.results.get(input_key)
        if df is None:
            raise ValueError(f"No input found for key: {input_key}")

        transform_func = self.TRANSFORMATIONS.get(transform_type)
        if not transform_func:
            raise ValueError(f"Unsupported transformation_type: {transform_type}")
        
        print("Input DataFrame columns:", df.columns.tolist())
        print("Transformation type:", transform_type)
        print("Transformation params:", transform_params)
        df_transformed = transform_func(df, **transform_params)

        #context.set(output_key, df_transformed)
        context.results[output_key] = df_transformed

    # 1. Head
    @staticmethod
    def head(df, n=5, columns=None):
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame")
            return df[columns].head(n)
        else:
            return df.head(n)

    # 2. Shape
    @staticmethod
    def shape(df):
        return pd.DataFrame({'rows':[df.shape[0]], 'columns':[df.shape[1]]})

    # 3. Describe
    @staticmethod
    def describe(df, columns=None):
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame")
            return df[columns].describe()
        else:
            return df.describe()

    # 4. Info (returns string, so wrap it)
    @staticmethod
    def info(df):
        from io import StringIO
        buf = StringIO()
        df.info(buf=buf)
        s = buf.getvalue()
        return pd.DataFrame({'info': [s]})

    # 5. Null Summary
    @staticmethod
    def null_summary(df, columns=None):
        cols = columns if columns else df.columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        return df[cols].isnull().sum().to_frame(name='null_count')

    # 6. Duplicate Rows
    @staticmethod
    def duplicate_rows(df):
        return df[df.duplicated()]

    # 7. Value Counts
    @staticmethod
    def value_counts(df, column):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        return df[column].value_counts().to_frame(name='counts')

    # 8. Unique Values
    @staticmethod
    def unique_values(df, column):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        return pd.Series(df[column].unique(), name='unique_values')

    # 9. Sample
    @staticmethod
    def sample(df, n=5, columns=None, random_state=None):
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame")
            return df[columns].sample(n=n, random_state=random_state)
        else:
            return df.sample(n=n, random_state=random_state)

    # 10. Column Stats (mean, median, std etc.)
    @staticmethod
    def column_stats(df, columns=None):
        cols = columns if columns else df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        stats = df[cols].agg(['mean','median','std','min','max','count']).transpose()
        return stats

    # 11. Impute
    @staticmethod
    def impute(df, columns=None, method="mean"):
        cols = columns if columns else df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")

        for col in cols:
            if method == "mean":
                val = df[col].mean()
            elif method == "median":
                val = df[col].median()
            elif method == "mode":
                val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
            else:
                raise ValueError(f"Unsupported impute method: {method}")
            df[col].fillna(val, inplace=True)
        return df

    # 12. Drop Nulls
    @staticmethod
    def drop_nulls(df, columns=None):
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame")
            return df.dropna(subset=columns)
        else:
            return df.dropna()

    # 13. Drop Duplicates
    @staticmethod
    def drop_duplicates(df, subset=None):
        if subset:
            missing = [c for c in subset if c not in df.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame")
        return df.drop_duplicates(subset=subset)

    # 14. Drop Columns
    @staticmethod
    def drop_columns(df, columns):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        return df.drop(columns=columns)

    # 15. Rename Columns
    @staticmethod
    def rename_columns(df, columns_map):
        if not isinstance(columns_map, dict):
            raise ValueError("rename_columns expects a dict for columns_map")
        missing = [c for c in columns_map.keys() if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        return df.rename(columns=columns_map)

    # 16. Filter Rows
    @staticmethod
    def filter_rows(df, column=None, threshold=None, operator='>'):
        if column is None or threshold is None:
            raise ValueError("filter_rows requires 'column' and 'threshold' parameters")
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        ops = {
            '>': df[column] > threshold,
            '>=': df[column] >= threshold,
            '<': df[column] < threshold,
            '<=': df[column] <= threshold,
            '==': df[column] == threshold,
            '!=': df[column] != threshold,
        }
        if operator not in ops:
            raise ValueError(f"Unsupported operator: {operator}")
        return df[ops[operator]]

    # 17. Sort Rows
    @staticmethod
    def sort_rows(df, columns=None, ascending=True):
        cols = columns or [df.columns[0]]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        return df.sort_values(by=cols, ascending=ascending)

    # 18. Select Columns
    @staticmethod
    def select_columns(df, columns):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        return df[columns]

    # 19. Create New Column
    @staticmethod
    def create_new_column(df, new_column_name):
        if new_column_name in df.columns:
            raise ValueError(f"Column '{new_column_name}' already exists")
        df[new_column_name] = pd.NA
        return df

    # 20. Binning
    @staticmethod
    def binning(df, column=None, bins=3, new_column='binned'):
        col = column or df.select_dtypes(include=np.number).columns[0]
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric for binning")
        df[new_column] = pd.cut(df[col], bins=bins)
        return df

    # 21. Label Encoding
    @staticmethod
    def label_encoding(df, columns=None):
        cols = columns or df.select_dtypes(include='object').columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        for col in cols:
            df[col] = df[col].astype('category').cat.codes
        return df

    # 22. One Hot Encoding
    @staticmethod
    def one_hot_encoding(df, columns=None):
        cols = columns or df.select_dtypes(include='object').columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        return pd.get_dummies(df, columns=cols)

    # 23. Standardize
    @staticmethod
    def standardize(df, columns=None):
        cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        for col in cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                raise ValueError(f"Standard deviation is zero for column '{col}'")
            df[col] = (df[col] - mean) / std
        return df

    # 24. Normalize
    @staticmethod
    def normalize(df, columns=None):
        cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        for col in cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val == min_val:
                raise ValueError(f"Column '{col}' has constant value, cannot normalize")
            df[col] = (df[col] - min_val) / (max_val - min_val)
        return df

    # 25. Log Transform
    @staticmethod
    def log_transform(df, columns=None):
        cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        for col in cols:
            if (df[col] < 0).any():
                raise ValueError(f"Column '{col}' contains negative values, cannot apply log transform")
            df[col] = np.log1p(df[col])
        return df

    # 26. Sqrt Transform
    @staticmethod
    def sqrt_transform(df, columns=None):
        cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        for col in cols:
            if (df[col] < 0).any():
                raise ValueError(f"Column '{col}' contains negative values, cannot apply sqrt transform")
            df[col] = np.sqrt(df[col])
        return df

    # 27. Power Transform
    @staticmethod
    def power_transform(df, columns=None, power=0.5):
        cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        for col in cols:
            df[col] = df[col] ** power
        return df

    # 28. Reciprocal
    @staticmethod
    def reciprocal(df, columns=None):
        cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        for col in cols:
            if (df[col] == 0).any():
                raise ValueError(f"Column '{col}' contains zeros, cannot apply reciprocal transform")
            df[col] = 1 / df[col]
        return df

    # 29. Group By
    @staticmethod
    def group_by(df, by=None, aggfunc='mean'):
        group_col = by or df.columns[0]
        if group_col not in df.columns:
            raise ValueError(f"Group by column '{group_col}' not found")
        return df.groupby(group_col).agg(aggfunc)

    # 30. Aggregate
    @staticmethod
    def aggregate(df, by=None, aggfunc='sum'):
        group_col = by or df.columns[0]
        if group_col not in df.columns:
            raise ValueError(f"Group by column '{group_col}' not found")
        return df.groupby(group_col).agg(aggfunc)
    
    # 31. Correlation Matrix
    @staticmethod
    def correlation_matrix(df, columns=None):
        """
        Calculate the correlation matrix for given columns or all numeric columns if none specified.
        Returns a DataFrame showing pairwise correlation coefficients.
        """
        # If columns not provided, use all numeric columns
        cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        
        # Check if all columns exist in df
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        
        # Calculate and return correlation matrix
        return df[cols].corr()
    
    # 32. Handle Outliers (simple IQR method)
    @staticmethod
    def handle_outliers(df, columns=None, method='clip'):
        cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")

        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            if method == 'clip':
                df[col] = df[col].clip(lower, upper)
            elif method == 'remove':
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            else:
                raise ValueError("method must be either 'clip' or 'remove'")
        return df

    # 33. Smooth (rolling mean)
    @staticmethod
    def smooth(df, columns=None, window=3):
        cols = columns or df.select_dtypes(include=np.number).columns.tolist()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame")
        for col in cols:
            df[col] = df[col].rolling(window=window, min_periods=1).mean()
        return df

    # 34. custom_code (executes Python code string on the DataFrame):
    @staticmethod
    def custom_code(df, code=None):
        """
        Allows executing a custom transformation on the DataFrame.
        The code string should use 'df' as the DataFrame variable and return a modified df.
        Example: "df['new_col'] = df['A'] + df['B']"
        """
        if not code:
            raise ValueError("No transformation code provided in 'code' parameter")
        # Prepare local scope with df, and execute the code
        local_vars = {'df': df.copy()}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            raise ValueError(f"Error executing custom code: {e}")

        # Make sure 'df' is updated and returned
        if 'df' not in local_vars or not isinstance(local_vars['df'], pd.DataFrame):
            raise ValueError("Custom code must modify and return the variable 'df'")

        return local_vars['df']

    # TRANSFORMATIONS dict mapping string to methods
    TRANSFORMATIONS = {
        'head': head.__func__,
        'shape': shape.__func__,
        'describe': describe.__func__,
        'info': info.__func__,
        'null_summary': null_summary.__func__,
        'duplicate_rows': duplicate_rows.__func__,
        'value_counts': value_counts.__func__,
        'unique_values': unique_values.__func__,
        'sample': sample.__func__,
        'column_stats': column_stats.__func__,
        'impute': impute.__func__,
        'drop_nulls': drop_nulls.__func__,
        'drop_duplicates': drop_duplicates.__func__,
        'drop_columns': drop_columns.__func__,
        'rename_columns': rename_columns.__func__,
        'filter_rows': filter_rows.__func__,
        'sort_rows': sort_rows.__func__,
        'select_columns': select_columns.__func__,
        'create_new_column': create_new_column.__func__,
        'binning': binning.__func__,
        'label_encoding': label_encoding.__func__,
        'one_hot_encoding': one_hot_encoding.__func__,
        'standardize': standardize.__func__,
        'normalize': normalize.__func__,
        'log_transform': log_transform.__func__,
        'sqrt_transform': sqrt_transform.__func__,
        'power_transform': power_transform.__func__,
        'reciprocal': reciprocal.__func__,
        'group_by': group_by.__func__,
        'aggregate': aggregate.__func__,
        'correlation_matrix': correlation_matrix.__func__,
        'handle_outliers': handle_outliers.__func__,
        'smooth': smooth.__func__,
        'custom_code': custom_code.__func__
    }
