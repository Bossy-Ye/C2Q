import openai
import os

api_key = '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'
organization = 'org-iBkqZ9GhbImMXfIuggzUg3au'
project = 'proj_cJRHOhDUpozcROd59n9ZTXZL'


from openai import OpenAI
client = OpenAI(api_key=api_key,
                organization=organization,
                project=project)
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)