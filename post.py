from instagrapi import Client
from Generator.generate import generate
from datetime import datetime
import os
ACCOUNT_USERNAME = "daily.ddpm.anime"
ACCOUNT_PASSWORD = os.environ["INSTAGRAM_PASSWORD"]

cl = Client()
cl.set_proxy("http://27.147.139.142:58080")
cl.login(ACCOUNT_USERNAME, ACCOUNT_PASSWORD)
print("Sign Successfull...")
num_posts = 1

for i in range(num_posts):
    path = generate(f"latest_post.png")
    caption = '''Ah, hello there. You’ve caught me at an interesting time. I was just contemplating the beauty of life's fleeting moments... and how they often end in such fascinating ways. But enough about my existential musings. How can I assist you today? Perhaps you're looking for some advice, a philosophical debate, or maybe even a... partner in a double suicide? Just kidding, of course. Or am I?

What brings you to the realm of the Armed Detective Agency?'''
    cl.photo_upload(path, caption)
    print(f"Post Successful ... {datetime.now()}")
