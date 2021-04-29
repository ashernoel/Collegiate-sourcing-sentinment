import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go

# import our CSV and clean up the date
df = pd.read_csv("feedback.csv")
df["date"] = df.Timestamp.str[:1].astype(int) + df.Timestamp.str[2:4].astype(int)/30 - 2.3
df_results = df["date"].copy().to_frame()

# import transformer
classifier = pipeline('sentiment-analysis')

# convert the list of dictionaries to a list of scores
scores1 = []
dic_life1 = classifier(df["sourcing"].values.tolist())
for dic in dic_life1:
    scores1.append(dic["score"])

scores2 = []
dic_life2 = classifier(df["life"].values.tolist())
for dic in dic_life2:
    scores2.append(dic["score"])

# add the list of scores to the dataframe
sourcing = pd.Series(scores1)
life = pd.Series(scores2)
df_results["sourcing"] = sourcing.values
df_results["life"] = life.values

# create mean sentiments for each date
test1 = df_results.groupby("date", as_index=False)["sourcing"].mean()
test2 = df_results.groupby("date", as_index=False)["life"].mean()

# visualize the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=test1["date"], y=test1["sourcing"],
                    mode='lines+markers',
                    name='sourcing vibe'))
fig.add_trace(go.Scatter(x=test2["date"], y=test2["life"],
                    mode='lines+markers',
                    name='life vibe'))

fig.update_layout(
    showlegend=True,
    title='Average Sentiment of Sourcers',
    xaxis_title='Months into Sourcing',
    yaxis_title='Sentiment (1 max)'
)
fig.show()



