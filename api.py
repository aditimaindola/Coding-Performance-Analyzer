import requests

def fetch_user_submissions(username):
    url = f"https://codeforces.com/api/user.status?handle={username}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to fetch data")

    data = response.json()

    if data["status"] != "OK":
        raise Exception("Invalid username or API error")

    return data["result"]