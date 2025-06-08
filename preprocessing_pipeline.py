import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_pipeline():
    # 1. Kolumny
    columns_to_drop = [
        'patient_id', 'dataset', 'acquisition_times', 'mastectomy_post_nac', 'days_to_follow_up', 'days_to_recurrence',
        'days_to_metastasis', 'days_to_death', 'oncotype_score', 'nottingham_grade', 'patient_size', 'view',
        'bilateral_mri', 'num_phases', 'fat_suppressed', 'field_strength', 'image_rows', 'image_columns',
        'num_slices', 'pixel_spacing', 'slice_thickness', 'site', 'manufacturer', 'scanner_model', 'high_bit',
        'window_center', 'window_width', 'echo_time', 'repetition_time', 'acquisition_date',
        'tcia_series_uid', 'nac_agent'
    ]

    columns_fill_minus1 = ['anti_her2_neu_therapy', 'endocrine_therapy', 'mammaprint', 'er', 'pr', 'hr', 'her2']
    columns_fill_0 = ['bilateral_breast_cancer', 'multifocal_cancer']
    columns_fill_median = ['age', 'weight']
    columns_fill_unknown = ['ethnicity', 'bmi_group', 'tumor_subtype', 'menopause']

    categorical_cols = [
        'fill_minus1__mammaprint', 'fill_minus1__anti_her2_neu_therapy', 'fill_minus1__endocrine_therapy',
        'fill_unknown__bmi_group', 'fill_unknown__menopause', 'fill_unknown__tumor_subtype',
        'fill_minus1__her2', 'fill_minus1__hr', 'fill_minus1__er', 'fill_minus1__pr',
        'fill_unknown__ethnicity'
    ]

    # 2. Dropper
    dropper = FunctionTransformer(lambda X: X.drop(columns=columns_to_drop))

    # 3. ColumnTransformer do imputacji
    column_imputer = ColumnTransformer(transformers=[
        ("fill_minus1", SimpleImputer(strategy="constant", fill_value=-1), columns_fill_minus1),
        ("fill_0", SimpleImputer(strategy="constant", fill_value=0), columns_fill_0),
        ("fill_median", SimpleImputer(strategy="median"), columns_fill_median),
        ("fill_unknown", SimpleImputer(strategy="constant", fill_value="unknown"), columns_fill_unknown),
    ], remainder='passthrough')

    # 4. Konwersja kolumn na string (kategorie)
    def cast_categories(X):
        df = pd.DataFrame(X)
        for col in df.columns:
            try:
                df[col] = df[col].astype(str)
            except Exception:
                pass
        return df

    cast_categoricals = FunctionTransformer(cast_categories, feature_names_out='one-to-one')

    # 5. OneHotEncoder
    onehot = ColumnTransformer(transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ], remainder='passthrough')

    # 6. Pipeline
    pipeline = Pipeline(steps=[
        ("drop_columns", dropper),
        ("impute_columns", column_imputer),
        ("cast_categoricals", cast_categoricals),
        ("encode_categoricals", onehot)
    ])

    # Zwracaj DataFrame z nazwami kolumn
    pipeline.set_output(transform="pandas")

    return pipeline
