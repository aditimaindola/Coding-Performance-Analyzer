import requests

username = "tourist"

url = f"https://codeforces.com/api/user.status?handle={username}"
response = requests.get(url)
data = response.json()

submissions = data["result"]

print("Total Submissions:", len(submissions))

# Print first 5 structured entries
for sub in submissions[:5]:
    problem = sub.get("problem", {})
    verdict = sub.get("verdict", "")

    name = problem.get("name")
    rating = problem.get("rating", "No rating")
    tags = problem.get("tags", [])

    print("Problem:", name)
    print("Rating:", rating)
    print("Tags:", tags)
    print("Verdict:", verdict)
    print("-" * 40)