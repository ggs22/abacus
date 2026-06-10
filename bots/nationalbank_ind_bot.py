#%%
import re
import time
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth

from utils.utils import print_step

ROOT_DIR     = Path(__file__).resolve().parents[1]
DRIVER_PATH  = ROOT_DIR / "bots" / "chrome_driver" / "chromedriver"
PROFILE_DIR  = ROOT_DIR / "bots" / "chrome_profile"
DOWNLOAD_DIR = ROOT_DIR / "bots" / "downloads"
ACCOUNTS_DIR = ROOT_DIR / "backend" / "accounts"

DOWNLOAD_DIR.mkdir(exist_ok=True)

NUM_STEPS = 3

# BNC portal account IDs from the accountSelect dropdown on /documents
_PDF_ACCOUNT_IDS: dict[str, str] = {
    "bnc_op": "653818cc14a4175c2360a37c",
    "bnc_mc": "654e647625572c351a4d4550",
    "bnc_cc": "653a565c984bb1155acebe15",
}

# BNC UI label → local folder (for the /accounts CSV download flow)
_CSV_ACCOUNT_LABELS: dict[str, str] = {
    "bnc_op": "Compte Chèques",
    "bnc_mc": "Mastercard Platine",
    "bnc_cc": "Marge de crédit",
}


# ── human-like interaction helpers ───────────────────────────────────────────

def _pause(seconds: float):
    """Sleep for seconds ± 40% to avoid mechanical timing patterns."""
    time.sleep(random.uniform(seconds * 0.6, seconds * 1.4))


def _type(element, text: str):
    """Type text character by character with random inter-key delays."""
    for ch in text:
        element.send_keys(ch)
        time.sleep(random.uniform(0.04, 0.12))


# ── local state helpers ───────────────────────────────────────────────────────

def get_latest_local_date(folder_name: str, subdir: str) -> date | None:
    """Return the latest date found in statement filenames in the given subfolder."""
    folder = ACCOUNTS_DIR / folder_name / subdir
    if not folder.exists():
        return None
    dates: list[date] = []
    for f in folder.iterdir():
        if f.suffix.lower() not in (".csv", ".pdf"):
            continue
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", f.stem)
        if m:
            dates.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue
        # Existing YYYY-MM or YYYY_MM files → treat as last day of that month
        m = re.search(r"(\d{4})[-_](\d{2})", f.stem)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            next_month = date(y + 1, 1, 1) if mo == 12 else date(y, mo + 1, 1)
            dates.append(next_month - timedelta(days=1))
    return max(dates) if dates else None


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


def _save_file(file: Path, folder_name: str, subdir: str, dest_name: str):
    dest_dir = ACCOUNTS_DIR / folder_name / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dest_name
    if dest.exists():
        print(f"    [skip] {dest_name} already in {folder_name}/{subdir}")
        file.unlink()
    else:
        shutil.move(str(file), str(dest))
        print(f"    [saved] {dest_name} → {folder_name}/{subdir}")


# ── CSV flow (/accounts) ──────────────────────────────────────────────────────

def _download_csv(label: str, folder_name: str):
    latest = get_latest_local_date(folder_name, "csv_data")
    start  = (latest + timedelta(days=1)) if latest else (date.today() - timedelta(days=365))
    end    = date.today()

    if start >= end:
        print(f"  [{label}] CSV already up to date")
        return

    print(f"\n── CSV: {label} ({start} → {end}) ──")

    driver.get("https://app.bnc.ca/accounts")
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "accounts-table"))
    )
    _pause(1)

    driver.find_element(
        By.XPATH, f"//button[@aria-label='Télécharger les relevés du {label}']"
    ).click()
    _pause(1)

    Select(WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "filter-form__period"))
    )).select_by_value("PERSONALIZED")
    _pause(0.5)

    start_input = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "filter-form__startDate"))
    )
    start_input.clear()
    _type(start_input, start.strftime("%d/%m/%Y"))

    end_input = driver.find_element(By.ID, "filter-form__endDate")
    end_input.clear()
    _type(end_input, end.strftime("%d/%m/%Y"))

    driver.find_element(By.ID, "filter-form__display").click()
    _pause(1.5)

    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (By.XPATH, "//button[@data-test='link' and normalize-space()='Télécharger']")
        )
    ).click()
    _pause(0.5)

    driver.find_element(By.CSS_SELECTOR, "label[for='radio-csv']").click()
    _pause(0.3)

    before = _snapshot_downloads()
    driver.find_element(
        By.XPATH, "//button[@data-test='button' and .//span[normalize-space()='Télécharger']]"
    ).click()

    downloaded = _wait_for_new_file(before, timeout=30)
    if downloaded:
        _save_file(downloaded, folder_name, "csv_data", f"{start.isoformat()}.csv")
    else:
        print(f"  [!] CSV download timed out for {label}")


# ── PDF flow (/documents) ─────────────────────────────────────────────────────

def _download_pdfs(account_id: str, folder_name: str):
    latest = get_latest_local_date(folder_name, "pdf_data")
    print(f"\n── PDFs: {folder_name} (local latest: {latest or 'none'}) ──")

    driver.get("https://app.bnc.ca/documents")
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "accountSelect"))
    )
    _pause(1)

    Select(driver.find_element(By.ID, "accountSelect")).select_by_value(account_id)
    _pause(0.3)
    Select(driver.find_element(By.ID, "documentTypeSelect")).select_by_value("STATEMENT")
    _pause(0.3)

    driver.find_element(
        By.XPATH, "//button[@aria-label='Cliquez pour rechercher ce type de document.']"
    ).click()
    _pause(2)

    result_items = driver.find_elements(
        By.XPATH, "//li[.//button[@aria-label]]"
    )

    if not result_items:
        print(f"  nothing found")
        return

    for item in result_items:
        btn = item.find_element(By.XPATH, ".//button[@aria-label]")
        aria = btn.get_attribute("aria-label") or ""

        m = re.search(r"(\d{4}-\d{2}-\d{2})", aria)
        stmt_date = m.group(1) if m else None

        if latest and stmt_date and stmt_date <= latest.isoformat():
            continue

        print(f"  → {aria}")
        original_windows = set(driver.window_handles)
        before = _snapshot_downloads()
        btn.click()
        _pause(1.5)

        new_windows = set(driver.window_handles) - original_windows
        if new_windows:
            driver.switch_to.window(new_windows.pop())
            _pause(1)
            driver.close()
            driver.switch_to.window(list(original_windows)[0])

        downloaded = _wait_for_new_file(before, timeout=30)
        if downloaded:
            dest_name = f"{stmt_date}_Relevé.pdf" if stmt_date else downloaded.name
            _save_file(downloaded, folder_name, "pdf_data", dest_name)
        else:
            print(f"  [!] PDF download timed out: {aria}")


# ── per-account functions ─────────────────────────────────────────────────────

def process_chequing():
    _download_csv(_CSV_ACCOUNT_LABELS["bnc_op"], "bnc_op")
    _download_pdfs(_PDF_ACCOUNT_IDS["bnc_op"], "bnc_op")


def process_mastercard():
    _download_csv(_CSV_ACCOUNT_LABELS["bnc_mc"], "bnc_mc")
    _download_pdfs(_PDF_ACCOUNT_IDS["bnc_mc"], "bnc_mc")


def process_credit_line():
    _download_csv(_CSV_ACCOUNT_LABELS["bnc_cc"], "bnc_cc")
    _download_pdfs(_PDF_ACCOUNT_IDS["bnc_cc"], "bnc_cc")


# ── main steps ────────────────────────────────────────────────────────────────

def nationalbank_login():
    print_step("National Bank login", 1, NUM_STEPS,
               f"Waiting for 2FA — {driver.current_url}")
    input("Complete login and 2FA in the browser, then press Enter to continue...")


def download_all_accounts():
    print_step("Downloading documents", 2, NUM_STEPS, "")
    process_chequing()
    process_mastercard()
    process_credit_line()
    print_step("Done", 3, NUM_STEPS, "")


def quit_browser():
    print_step("Quitting", 1, 1, driver.current_url)
    driver.quit()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1440,900")
    options.add_experimental_option("prefs", {
        "download.default_directory":         str(DOWNLOAD_DIR),
        "download.prompt_for_download":       False,
        "plugins.always_open_pdf_externally": True,
    })
    driver = uc.Chrome(options=options,
                       user_data_dir=str(PROFILE_DIR),
                       driver_executable_path=str(DRIVER_PATH))
    stealth(driver,
            languages=["fr-CA", "fr", "en-CA", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True)

    driver.get("https://www.bnc.ca/")
    WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-text='Particuliers']"))
    ).click()

    nationalbank_login()
    download_all_accounts()

    quit_browser()
