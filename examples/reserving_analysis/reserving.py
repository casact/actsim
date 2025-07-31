import chainladder as cl
import pandas as pd

file_path = 'sample_file_path'

df_cat_claim = pd.read_csv(
    file_path
)

df_cat_claim_filtered = df_cat_claim[(df_cat_claim['accident_year']<2026) & (df_cat_claim['development_date']<'2026-10-01')]

# Initialize the Loss Development Triangle
tri_cat_claim = cl.Triangle(
    data=df_cat_claim_filtered ,
    origin="incurred_date",
    development="development_date",
    columns=["incurred_loss"],
    cumulative=True,
)

tri_cat_claim_OQDQ = tri_cat_claim['incurred_loss'].grain('OQDQ')
tri_cat_claim_OQDQ.link_ratio