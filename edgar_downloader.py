import os
import re
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "CheckitRAG dg4489@nyu.edu"}
BASE_SAVE = "/Users/dedeepyaguntaka/Library/CloudStorage/GoogleDrive-dg4489@nyu.edu/My Drive/Checkit_Analytics_RAG"
FAILED_LOG = "/Users/dedeepyaguntaka/Documents/Checkit/checkit-rag/failed_downloads.txt"
TICKERS = ["AAPL", "MSFT", "META", "NVDA", "TSLA", "JPM", "GS", "SBUX", "WMT", "MCD",
           "NKE", "DIS", "CRM", "GOOGL", "AMZN", "NFLX", "ORCL", "MS", "BAC", "V"]
FORM_TYPES = ["8-K", "10-K", "10-Q"]
N_FILINGS = 20
MIN_DATE = "2020-01-01"

MONTH_TO_QUARTER = {
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3,
    10: 4, 11: 4, 12: 4,
}

saved_count = 0
skipped_exists_count = 0
failed_count = 0
failed_entries = []


def sleep():
    time.sleep(0.5)


def date_to_quarter(date_str):
    try:
        parts = date_str.split("-")
        year = int(parts[0])
        month = int(parts[1])
        return year, MONTH_TO_QUARTER[month]
    except Exception:
        return 0, 0


def build_filename(ticker, form_type, date_str):
    year, q = date_to_quarter(date_str)
    return f"{ticker}_{form_type}_{year}_Q{q}_{date_str}.txt"


def filing_exists(ticker, form_type, date_str):
    folder = os.path.join(BASE_SAVE, ticker, form_type)
    fname = build_filename(ticker, form_type, date_str)
    return os.path.exists(os.path.join(folder, fname))


def get_cik(ticker):
    url = (f"https://www.sec.gov/cgi-bin/browse-edgar"
           f"?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=include&count=10")
    resp = requests.get(url, headers=HEADERS)
    sleep()
    soup = BeautifulSoup(resp.text, "html.parser")
    match = re.search(r"CIK=(\d+)", resp.url)
    if match:
        return match.group(1).lstrip("0") or "0"
    match = re.search(r"CIK\s*=\s*(\d+)", resp.text)
    if match:
        return match.group(1).lstrip("0") or "0"
    info = soup.find("span", class_="companyInfo")
    if info:
        m = re.search(r"CIK (\d+)", info.get_text())
        if m:
            return m.group(1).lstrip("0") or "0"
    raise ValueError(f"Could not find CIK for {ticker}")


def get_submissions(cik):
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    resp = requests.get(url, headers=HEADERS)
    sleep()
    resp.raise_for_status()
    return resp.json()


def get_document_list(cik, accession_number):
    acc_clean = accession_number.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/"
    resp = requests.get(index_url, headers=HEADERS)
    sleep()
    if resp.status_code != 200:
        raise ValueError(f"Filing index returned {resp.status_code}")
    soup = BeautifulSoup(resp.text, "html.parser")
    docs = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".htm") or href.endswith(".html") or href.endswith(".txt"):
            full_url = "https://www.sec.gov" + href if href.startswith("/") else href
            docs.append(full_url)
    return docs


def fetch_text(url):
    resp = requests.get(url, headers=HEADERS)
    sleep()
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "html" in content_type or url.endswith(".htm") or url.endswith(".html"):
        soup = BeautifulSoup(resp.content, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    return resp.text


def pick_8k_exhibit(docs):
    candidates = []
    for url in docs:
        fname = url.split("/")[-1].lower()
        if "index" in fname or fname.startswith("0000"):
            continue
        candidates.append(url)
    if not candidates:
        return docs[0] if docs else None
    best_url = None
    best_size = -1
    for url in candidates:
        try:
            head = requests.head(url, headers=HEADERS)
            sleep()
            size = int(head.headers.get("Content-Length", 0))
            if size > best_size:
                best_size = size
                best_url = url
        except Exception:
            pass
    return best_url or candidates[0]


def pick_primary_doc(docs, form_type):
    for url in docs:
        fname = url.split("/")[-1].lower()
        if "index" in fname:
            continue
        if fname.endswith(".htm") or fname.endswith(".html") or fname.endswith(".txt"):
            return url
    return docs[0] if docs else None


def save_filing(ticker, form_type, date_str, text):
    folder = os.path.join(BASE_SAVE, ticker, form_type)
    os.makedirs(folder, exist_ok=True)
    fname = build_filename(ticker, form_type, date_str)
    fpath = os.path.join(folder, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)
    return fpath


def process_ticker(ticker):
    global saved_count, skipped_exists_count, failed_count

    try:
        cik_raw = get_cik(ticker)
        cik = int(cik_raw)
    except Exception as e:
        print(f"  ✗ {ticker} — could not get CIK: {e}")
        failed_entries.append(f"{ticker} — CIK lookup failed: {e}")
        failed_count += 1
        return

    try:
        data = get_submissions(cik)
    except Exception as e:
        print(f"  ✗ {ticker} — could not fetch submissions: {e}")
        failed_entries.append(f"{ticker} — submissions fetch failed: {e}")
        failed_count += 1
        return

    recent = data.get("filings", {}).get("recent", {})
    form_types_list = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])

    for form_type in FORM_TYPES:
        matching = []
        for i, ft in enumerate(form_types_list):
            if ft != form_type:
                continue
            d = dates[i]
            if d < MIN_DATE:
                continue
            matching.append((d, accessions[i]))
            if len(matching) == N_FILINGS:
                break

        if not matching:
            continue

        for date_str, accession in matching:
            year, q = date_to_quarter(date_str)
            label = f"Q{q} {year}"

            if filing_exists(ticker, form_type, date_str):
                print(f"  → {ticker} {form_type} {label} ({date_str}) skipped — already exists")
                skipped_exists_count += 1
                continue

            try:
                docs = get_document_list(cik, accession)
                if not docs:
                    raise ValueError("no documents found in filing index")

                if form_type == "8-K":
                    doc_url = pick_8k_exhibit(docs)
                else:
                    doc_url = pick_primary_doc(docs, form_type)

                if not doc_url:
                    raise ValueError("could not select document")

                text = fetch_text(doc_url)
                if not text.strip():
                    raise ValueError("fetched text is empty")

                save_filing(ticker, form_type, date_str, text)
                print(f"  ✓ {ticker} {form_type} {label} ({date_str}) saved")
                saved_count += 1

            except Exception as e:
                reason = type(e).__name__ + (f": {e}" if str(e) else "")
                print(f"  ✗ {ticker} {form_type} {label} ({date_str}) failed: {reason}")
                failed_entries.append(f"{ticker} {form_type} {label} ({date_str}) — {reason}")
                failed_count += 1


def main():
    print(f"Starting EDGAR downloader for {len(TICKERS)} tickers...")
    print(f"Saving to: {BASE_SAVE}")
    print(f"Fetching up to {N_FILINGS} filings per type, from {MIN_DATE} onwards\n")

    for ticker in TICKERS:
        print(f"[{ticker}]")
        process_ticker(ticker)

    # Write failed log
    with open(FAILED_LOG, "w", encoding="utf-8") as f:
        if failed_entries:
            f.write("\n".join(failed_entries) + "\n")
        else:
            f.write("No failures.\n")

    print("\n=== Done ===")
    print(f"Downloaded:       {saved_count} new files")
    print(f"Skipped (exists): {skipped_exists_count} files")
    print(f"Failed:           {failed_count} files")
    if failed_count > 0:
        print(f"(see failed_downloads.txt for details)")


if __name__ == "__main__":
    main()
