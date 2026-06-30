"""Draft an NNS broadcast package with DeepSeek -> output/news_package.json
(anchor_name, kicker, headline, screen_label, ticker[], script). Needs $DEEPSEEK_API_KEY.
  python3 newscast/draft_news.py ["optional topic hint"]
"""
import os, sys, json, urllib.request
key = os.environ["DEEPSEEK_API_KEY"]
hint = sys.argv[1] if len(sys.argv) > 1 else "one plausible serious tech/world story (invent realistic specifics)"
sysmsg = "You are a TV news script writer for NNS (National News Service)."
usr = (f"Write a single-anchor broadcast read, ~30 seconds spoken (~80-90 words) about {hint}. "
       "Structure: 'Good evening, I'm <anchor name>.' then the lead story, then a brief sign-off ending '...NNS.' "
       "Performable: natural cadence, a comma/pause or two, no stage directions. "
       "Return STRICT JSON only with keys: anchor_name (first last), kicker (1-2 words ALL CAPS e.g. BREAKING/TECH/WORLD), "
       "headline (<=6 words Title Case), screen_label (2-3 words), subhead (one short sentence <=9 words summarizing the story for the on-screen graphic), "
       "ticker (array of 6 short unrelated one-line headlines), "
       "script (the spoken read as one string).")
body = json.dumps({"model": "deepseek-v4-flash",
                   "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": usr}],
                   "temperature": 0.8, "response_format": {"type": "json_object"}}).encode()
req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions", data=body,
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
content = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
open("output/news_package.json", "w").write(content)
print(content)
