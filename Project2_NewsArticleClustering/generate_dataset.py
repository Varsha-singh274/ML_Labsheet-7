"""
generate_dataset.py
--------------------
Generates a synthetic news-article dataset for Project 2 (News Article
Clustering System).

The lab sheet did not supply a real dataset, and network access to fetch a
public corpus (e.g. the 20 Newsgroups dataset) is not available from this
environment, so this script builds a synthetic corpus of short news articles
across five realistic topic categories: Sports, Technology, Politics,
Business, and Health. Each category has its own vocabulary and sentence
templates, so the resulting documents have genuine, learnable textual
structure for TF-IDF to pick up - the same kind of signal a real news corpus
would have, just generated rather than scraped.

The "true" category label is saved as well (True_Category), purely so this
assignment's cluster results can be checked for accuracy - a real deployment
would not have this column, since the whole point of clustering is to work
without labels.
"""

import random
import pandas as pd

random.seed(42)

TOPICS = {
    "Sports": {
        "subjects": ["the national cricket team", "the city football club", "the Olympic sprinter",
                     "the tennis champion", "the basketball league", "the marathon runner",
                     "the hockey squad", "the chess grandmaster"],
        "actions": ["won the championship after a thrilling final", "suffered a surprise defeat in the semi-final",
                    "set a new national record", "signed a new coach ahead of the season",
                    "announced their squad for the upcoming tournament", "celebrated a historic victory",
                    "faced a tough injury setback before the match", "trained intensively for the finals"],
        "extras": ["Fans celebrated across the city.", "The coach praised the team's discipline.",
                   "Ticket sales broke previous records.", "Analysts called it the match of the season.",
                   "The stadium was packed to capacity.", "Sponsors are reportedly pleased with the exposure."],
    },
    "Technology": {
        "subjects": ["the leading smartphone manufacturer", "the AI research lab", "the cloud computing startup",
                     "the social media platform", "the chip manufacturer", "the electric vehicle company",
                     "the cybersecurity firm", "the software giant"],
        "actions": ["unveiled a new generation of processors", "launched an updated artificial intelligence model",
                    "reported a major data breach affecting millions of users",
                    "announced record quarterly revenue driven by cloud services",
                    "released a software update fixing critical security flaws",
                    "showcased a prototype at the annual technology conference",
                    "partnered with a university on machine learning research",
                    "faced regulatory scrutiny over data privacy practices"],
        "extras": ["Shares rose sharply after the announcement.", "Developers welcomed the new features.",
                   "Analysts remain divided on its long-term impact.", "Users reported mixed experiences online.",
                   "The company promised further updates soon.", "Competitors are expected to respond quickly."],
    },
    "Politics": {
        "subjects": ["the prime minister", "the opposition leader", "the finance minister",
                     "the state government", "the newly elected mayor", "the ruling party",
                     "the parliamentary committee", "the foreign affairs secretary"],
        "actions": ["announced a new economic policy during the parliamentary session",
                    "criticized the government's handling of the recent crisis",
                    "held talks with international leaders on trade relations",
                    "proposed sweeping reforms to the education system",
                    "faced backlash over the controversial new bill",
                    "called for an emergency session to discuss the budget",
                    "signed a landmark agreement with a neighboring country",
                    "addressed concerns over rising inflation in a public statement"],
        "extras": ["The announcement sparked widespread debate.", "Opposition parties demanded clarifications.",
                   "Citizens expressed mixed reactions on social media.", "The decision is expected to take effect next month.",
                   "International observers are closely watching the situation.", "Analysts predict significant policy shifts."],
    },
    "Business": {
        "subjects": ["the central bank", "the leading retail chain", "the automobile manufacturer",
                     "the national stock exchange", "the oil and gas company", "the airline industry",
                     "the real estate sector", "the banking conglomerate"],
        "actions": ["raised interest rates to curb rising inflation", "reported stronger than expected quarterly earnings",
                    "announced a major merger with a rival company", "cut thousands of jobs amid restructuring",
                    "saw its stock price plunge after disappointing results", "expanded operations into new overseas markets",
                    "faced supply chain disruptions affecting production", "unveiled an ambitious five-year growth strategy"],
        "extras": ["Investors reacted cautiously to the news.", "The move is seen as a sign of market confidence.",
                   "Analysts have revised their forecasts accordingly.", "Trading volumes surged following the announcement.",
                   "Industry experts called it a bold strategic shift.", "Markets remain volatile amid the uncertainty."],
    },
    "Health": {
        "subjects": ["the health ministry", "leading hospital researchers", "the pharmaceutical company",
                     "the World Health Organization", "a team of medical scientists", "the public health department",
                     "a group of nutrition experts", "the vaccine research institute"],
        "actions": ["announced promising results from a new clinical trial", "issued new guidelines for seasonal illness prevention",
                    "warned of a rise in cases during the current season",
                    "launched a nationwide vaccination awareness campaign",
                    "published a study linking diet to long-term health outcomes",
                    "recommended increased screening for early detection",
                    "received approval for a new treatment method",
                    "called for greater investment in mental health services"],
        "extras": ["Experts urged the public to stay informed.", "The findings were published in a leading medical journal.",
                   "Health officials called the results encouraging.", "Doctors recommend consulting a physician for more details.",
                   "The campaign will run throughout the coming months.", "Specialists say more research is still needed."],
    },
}

N_PER_TOPIC = 30
rows = []
article_id = 1

for topic, vocab in TOPICS.items():
    for _ in range(N_PER_TOPIC):
        subject = random.choice(vocab["subjects"])
        action = random.choice(vocab["actions"])
        extra1 = random.choice(vocab["extras"])
        extra2 = random.choice(vocab["extras"])
        sentence1 = f"{subject.capitalize()} {action}."
        text = f"{sentence1} {extra1} {extra2}"
        rows.append({
            "ArticleID": article_id,
            "Text": text,
            "True_Category": topic,  # kept only for accuracy-checking in this assignment
        })
        article_id += 1

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("dataset/news_articles.csv", index=False)
print(f"Saved dataset/news_articles.csv with {len(df)} articles across {len(TOPICS)} topics.")
print(df.head())
