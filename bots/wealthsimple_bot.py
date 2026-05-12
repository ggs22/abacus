#%%
import re
import time
import shutil
from pathlib import Path
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from utils.utils import print_step

ROOT_DIR     = Path(__file__).resolve().parents[1]
DRIVER_PATH  = ROOT_DIR / "bots" / "chrome_driver" / "chromedriver"
PROFILE_DIR  = ROOT_DIR / "bots" / "chrome_profile"
DOWNLOAD_DIR = ROOT_DIR / "bots" / "downloads"
ACCOUNTS_DIR = ROOT_DIR / "accounting" / "accounts"

DOWNLOAD_DIR.mkdir(exist_ok=True)

# WealthSimple UI label → local account folder name
ACCOUNT_LABEL_MAP: dict[str, str] = {
    "Chequing":                 "wealthsimpleop",
    "TFSA":                     "wealthsimpletfsa",
    "FHSA":                     "wealthsimplefhsa",
    "RRSP":                     "wealthsimplerrsp",
    "Corporate chequing":       "ordialwealthsimpleop",
    "Corporate investing":      "ordialwealthsimpleci",
    "Portfolio line of credit": "wealthsimpleplc",
}

_MONTH_NUMS: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

NUM_STEPS = 3


# ── local state helpers ───────────────────────────────────────────────────────

def _build_account_id_map() -> dict[str, Path]:
    """Map WS account-ID prefix (e.g. 'HQ64PZ866CAD') → local account folder."""
    mapping: dict[str, Path] = {}
    for folder in ACCOUNTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        for sub in ("pdf_data", "csv_data"):
            for f in (folder / sub).glob("*CAD_*"):
                acct_id = f.name.split("_")[0]
                if acct_id and acct_id not in mapping:
                    mapping[acct_id] = folder
    return mapping


ACCOUNT_ID_MAP: dict[str, Path] = _build_account_id_map()
ACCOUNT_ID_MAP.setdefault("WK7FFRH08CAD", ACCOUNTS_DIR / "ordialwealthsimpleop")


def _is_statement_file(f: Path) -> bool:
    """True for WealthSimple statement exports; False for bare balance-snapshot CSVs."""
    stem = f.stem
    if f.suffix.lower() == ".pdf":
        return True
    # Old WS export format: exactly YYYY-MM
    if re.fullmatch(r"\d{4}-\d{2}", stem):
        return True
    # New WS export format: descriptive name containing the account ID
    return "CAD" in stem


def get_latest_local_date(folder_name: str) -> str | None:
    """Return the most recent YYYY-MM from statement filenames in pdf_data/ and csv_data/."""
    folder = ACCOUNTS_DIR / folder_name
    dates: list[str] = []
    for sub in ("pdf_data", "csv_data"):
        d = folder / sub
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() not in (".pdf", ".csv"):
                continue
            if not _is_statement_file(f):
                continue
            m = re.search(r"(\d{4}-\d{2})(?:-\d{2})?", f.stem)
            if m:
                dates.append(m.group(1))
    return max(dates) if dates else None


def _parse_stmt_period(aria_label: str) -> str | None:
    """
    Extract YYYY-MM statement period from an aria-label like:
      'Open Chequing March Monthly Statement Chequing April 8, 2026'
    The statement month is the first month name in the title;
    the year comes from the trailing publication date.
    """
    pub_m = re.search(r"(\w+) \d{1,2}, (\d{4})$", aria_label)
    if not pub_m:
        return None
    pub_month_num = _MONTH_NUMS.get(pub_m.group(1), 0)
    pub_year = int(pub_m.group(2))

    title_part = aria_label[: pub_m.start()]
    for month_name, month_num in _MONTH_NUMS.items():
        if month_name in title_part:
            # December statement published in January → previous year
            stmt_year = pub_year if month_num <= pub_month_num else pub_year - 1
            return f"{stmt_year}-{month_num:02d}"
    return None


# ── download helpers ──────────────────────────────────────────────────────────

def _snapshot_downloads() -> set[Path]:
    return {f for f in DOWNLOAD_DIR.iterdir()
            if f.suffix not in (".crdownload", ".tmp")}


def _wait_for_new_file(before: set[Path], timeout: int = 30) -> Path | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        after = {f for f in DOWNLOAD_DIR.iterdir()
                 if f.suffix not in (".crdownload", ".tmp")}
        new = after - before
        if new:
            return max(new, key=lambda f: f.stat().st_mtime)
        time.sleep(0.5)
    return None


def _route_file(file: Path, ws_label: str):
    suffix = file.suffix.lower()
    sub = "pdf_data" if suffix == ".pdf" else "csv_data"

    acct_id = file.name.split("_")[0]
    folder = ACCOUNT_ID_MAP.get(acct_id)
    if folder is None:
        folder_name = ACCOUNT_LABEL_MAP.get(ws_label)
        folder = ACCOUNTS_DIR / folder_name if folder_name else None

    if folder is None:
        dest = DOWNLOAD_DIR / "unrouted" / file.name
        dest.parent.mkdir(exist_ok=True)
        shutil.move(str(file), str(dest))
        print(f"    [?] unrouted → bots/downloads/unrouted/{file.name}")
        return

    dest_dir = folder / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.name

    if dest.exists():
        print(f"    [skip] {file.name} already in {folder.name}/{sub}")
        file.unlink()
    else:
        ACCOUNT_ID_MAP.setdefault(acct_id, folder)
        shutil.move(str(file), str(dest))
        print(f"    [saved] {file.name} → {folder.name}/{sub}")


def _click_and_collect(btn, ws_label: str):
    """Click a download button, wait for the file, and route it."""
    original_windows = set(driver.window_handles)
    before = _snapshot_downloads()
    btn.click()

    # Handle case where click opens a new tab (e.g. PDF viewer)
    time.sleep(1.5)
    new_windows = set(driver.window_handles) - original_windows
    if new_windows:
        driver.switch_to.window(new_windows.pop())
        time.sleep(1)
        driver.close()
        driver.switch_to.window(list(original_windows)[0])

    downloaded = _wait_for_new_file(before, timeout=30)
    if downloaded:
        _route_file(downloaded, ws_label)
    else:
        print(f"    [!] download timed out")


# ── page interaction ──────────────────────────────────────────────────────────

def _expand_filter(label_text: str):
    """Expand a filter accordion by its label text; return the region element."""
    btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((
            By.XPATH,
            f"//button[@aria-controls and .//p[normalize-space()='{label_text}']]",
        ))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", btn)
    if btn.get_attribute("aria-expanded") == "false":
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(0.5)
    region = driver.find_element(By.ID, btn.get_attribute("aria-controls"))
    WebDriverWait(driver, 5).until(EC.visibility_of(region))
    return region


def _set_checkbox(region, label_text: str, checked: bool):
    label = region.find_element(
        By.XPATH, f".//label[.//p[normalize-space()='{label_text}']]"
    )
    inp = label.find_element(By.XPATH, ".//input[@type='checkbox']")
    if inp.is_selected() != checked:
        label.click()
        time.sleep(0.4)


def _find_doc_entries() -> list[dict]:
    """
    Return all visible document entries.
    Each entry: {"stmt_period": str|None, "aria_label": str,
                 "open_btn": WebElement, "csv_btn": WebElement|None}
    """
    open_btns = driver.find_elements(
        By.XPATH, "//button[starts-with(@aria-label, 'Open ')]"
    )
    entries = []
    for btn in open_btns:
        aria_label = btn.get_attribute("aria-label") or ""
        stmt_period = _parse_stmt_period(aria_label)

        # CSV button lives in the same row container as the Open button
        csv_btn = None
        try:
            row = btn.find_element(
                By.XPATH,
                "./ancestor::div[.//button[normalize-space()='Download CSV']][1]",
            )
            csv_btn = row.find_element(
                By.XPATH, ".//button[normalize-space()='Download CSV']"
            )
        except Exception:
            pass

        entries.append({
            "stmt_period": stmt_period,
            "aria_label": aria_label,
            "open_btn": btn,
            "csv_btn": csv_btn,
        })
    return entries


def _click_load_more() -> bool:
    btns = driver.find_elements(By.XPATH, "//button[normalize-space()='Load more']")
    if btns:
        driver.execute_script("arguments[0].scrollIntoView(true);", btns[0])
        btns[0].click()
        time.sleep(1.5)
        return True
    return False


# ── per-account logic ─────────────────────────────────────────────────────────

def process_account(ws_label: str, folder_name: str, download_csv: bool = True):
    latest = get_latest_local_date(folder_name)
    print(f"\n── {ws_label} (local latest: {latest or 'none'}) ──")

    driver.get("https://my.wealthsimple.com/app/docs")
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "main")))
    time.sleep(2)

    account_region = _expand_filter("Account")
    _set_checkbox(account_region, ws_label, True)
    doc_type_region = _expand_filter("Document type")
    _set_checkbox(doc_type_region, "Performance statements", True)
    time.sleep(1)

    # Collect entries to download, loading more pages until we reach known content
    to_download: list[dict] = []
    seen_labels: set[str] = set()

    while True:
        entries = _find_doc_entries()
        new_entries = [e for e in entries if e["aria_label"] not in seen_labels]
        seen_labels.update(e["aria_label"] for e in new_entries)

        stop = False
        for entry in new_entries:
            sp = entry["stmt_period"]
            if latest and sp and sp < latest:
                stop = True
                break
            to_download.append(entry)

        if stop or not _click_load_more():
            break

    if not to_download:
        print("  nothing new")
        return

    print(f"  {len(to_download)} new statement(s) to download")
    for entry in to_download:
        print(f"  → {entry['aria_label']}")
        if download_csv and entry["csv_btn"] is not None:
            _click_and_collect(entry["csv_btn"], ws_label)
        _click_and_collect(entry["open_btn"], ws_label)


def process_chequing():
    process_account("Chequing", "wealthsimpleop", download_csv=True)


def process_corporate_chequing():
    process_account("Corporate chequing", "ordialwealthsimpleop", download_csv=True)


def process_tfsa():
    process_account("TFSA", "wealthsimpletfsa", download_csv=False)


def process_fhsa():
    process_account("FHSA", "wealthsimplefhsa", download_csv=False)


def process_rrsp():
    process_account("RRSP", "wealthsimplerrsp", download_csv=False)


def process_ordialwealthsimpleop():
    process_account("Corporate chequing", "ordialwealthsimpleop", download_csv=True)


def process_corporate_investing():
    process_account("Corporate investing", "ordialwealthsimpleci", download_csv=False)


def process_plc():
    process_account("Portfolio line of credit", "wealthsimpleplc", download_csv=False)


# ── main steps ────────────────────────────────────────────────────────────────

def wealthsimple_login():
    print_step("Wealthsimple login", 1, NUM_STEPS,
               f"Waiting for password field — {driver.current_url}")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "password")))
    input("Enter your password in the browser, then press Enter to continue...")


def download_all_accounts():
    print_step("Downloading documents", 2, NUM_STEPS, "")
    process_chequing()
    process_corporate_chequing()
    process_ordialwealthsimpleop()
    process_tfsa()
    process_fhsa()
    process_rrsp()
    process_corporate_investing()
    process_plc()
    print_step("Done", 3, NUM_STEPS, "")


def quit_browser():
    print_step("Quitting", 1, 1, driver.current_url)
    driver.quit()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    options = Options()
    options.add_argument(f"--user-data-dir={PROFILE_DIR}")
    options.add_experimental_option("prefs", {
        "download.default_directory":         str(DOWNLOAD_DIR),
        "download.prompt_for_download":       False,
        "plugins.always_open_pdf_externally": True,
    })
    service = Service(str(DRIVER_PATH))
    driver = webdriver.Chrome(service=service, options=options)

    driver.get("https://my.wealthsimple.com/app/login")
    wealthsimple_login()
    download_all_accounts()

    quit_browser()
