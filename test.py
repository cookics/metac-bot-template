import os
import dotenv

dotenv.load_dotenv()

print(f"ASKNEWS_CLIENT_ID: {os.getenv('ASKNEWS_CLIENT_ID')}")
print(f"ASKNEWS_SECRET: {os.getenv('ASKNEWS_SECRET')}")
