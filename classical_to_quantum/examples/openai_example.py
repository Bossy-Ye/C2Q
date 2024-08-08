from openai import OpenAI
import os

client = OpenAI(
    api_key='sk-None-ZcD0TBz08o7Okrj7gLZqT3BlbkFJFbyKVH3g1hHOIPJXgOuP'
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)