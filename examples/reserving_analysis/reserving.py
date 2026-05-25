import chainladder as cl
import pandas as pd
from actdecor import summarize_with_llm, completion

file_path = 'examples/reserving_analysis/claim_development_random.csv'

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

######################################
####### AI Augmentation with ActDecor
######################################
@summarize_with_llm(model="meta/llama-3.3-70b-instruct", completion_params={"temperature": 0}, instructions = "Help me select the optimal number quarters of average for age 3 - 6, keep it concise")
def get_link_ratio(triangle):
    return triangle.link_ratio

tri_cat_claim_link_ratio = get_link_ratio(tri_cat_claim_OQDQ)
print(tri_cat_claim_link_ratio.value)
print(tri_cat_claim_link_ratio.commentary)
print(tri_cat_claim_link_ratio.info)

response = completion(
    model="nvidia_nim/meta/llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Help me select the optimal number years of average for age 3 - 6"}],
    temperature=0,
)