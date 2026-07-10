import requests

headers = {
    "User-Agent": "FinSightAI-DevUser student_dev@finsightai.local"
}

url = "https://efts.sec.gov/LATEST/search-index"
params = {
    "q": '"10-K"',
    "startdt": "2023-01-01",
    "enddt": "2023-12-31",
    "forms": "10-K",
    "size": 100
}

response = requests.get(url, headers=headers, params=params)
if response.status_code == 200:
    hits = response.json().get("hits", {}).get("hits", [])
    target_sics = {"6022", "6141", "6159", "6311", "6321", "6411", "7372"}
    found = []
    for hit in hits:
        source = hit.get("_source", {})
        sics = source.get("sics", [])
        ciks = source.get("ciks", [])
        names = source.get("display_names", [])
        if sics and ciks and names:
            # Check if any of the sics matches our target
            matched = [s for s in sics if s in target_sics]
            if matched:
                found.append((names[0], ciks[0], matched[0]))
    print(f"Searched 100 filings. Found {len(found)} companies matching target SIC codes:")
    for name, cik, sic in found[:10]:
        print(f" - {name} | CIK: {cik} | SIC: {sic}")
else:
    print(f"Error: {response.status_code}")
