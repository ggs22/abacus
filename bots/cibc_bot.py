#%%
import re
import time
import shutil
from datetime import date
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc

from utils.utils import print_step

ROOT_DIR     = Path(__file__).resolve().parents[1]
DRIVER_PATH  = ROOT_DIR / "bots" / "chrome_driver" / "chromedriver"
PROFILE_DIR  = ROOT_DIR / "bots" / "chrome_profile"
DOWNLOAD_DIR = ROOT_DIR / "bots" / "downloads"
ACCOUNT_DIR  = ROOT_DIR / "accounting" / "accounts" / "cibc"

DOWNLOAD_DIR.mkdir(exist_ok=True)

LOGIN_URL      = "https://www.cibconline.cibc.com/ebm-resources/public/auth-gateway/main/client/index.html?locale=en&auth_context=login"
DOWNLOAD_URL   = "https://www.cibconline.cibc.com/ebm-resources/public/banking/cibc/client/web/index.html#/accounts/download"
STATEMENTS_URL = "https://www.cibconline.cibc.com/ebm-resources/public/banking/cibc/client/web/index.html#/accounts/online-statements/525c847b0b14c4209344567983d679c376441e1e64e9e4a6e0d0138689de364d/details"

_MONTH_NUMS: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

NUM_STEPS = 3


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _parse_stmt_end_date(aria_label: str) -> str | None:
    """Extract YYYY-MM-DD end date from 'Month DD to Month DD, YYYY. Download this PDF.'"""
    m = re.search(r"to (\w+) (\d{1,2}), (\d{4})", aria_label)
    if not m:
        return None
    month_num = _MONTH_NUMS.get(m.group(1))
    if month_num is None:
        return None
    return f"{m.group(3)}-{month_num:02d}-{int(m.group(2)):02d}"


def _get_existing_pdf_dates() -> set[str]:
    """Return set of end-date strings (YYYY-MM-DD) already in pdf_data/."""
    dates: set[str] = set()
    for f in (ACCOUNT_DIR / "pdf_data").glob("onlineStatement_*.pdf"):
        m = re.search(r"onlineStatement_(\d{4}-\d{2}-\d{2})\.pdf", f.name)
        if m:
            dates.add(m.group(1))
    return dates


# ── main steps ────────────────────────────────────────────────────────────────

def cibc_login():
    print_step("CIBC login", 1, NUM_STEPS,
               f"Complete login in the browser, then press Enter...")
    input()


def download_csv():
    print_step("Downloading CSV transactions", 2, NUM_STEPS, "")
    driver.get(DOWNLOAD_URL)

    label = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located(
            (By.XPATH, "//label[normalize-space()='All since last download']")
        )
    )
    inp = driver.find_element(By.ID, label.get_attribute("for"))
    if not inp.is_selected():
        label.click()
        time.sleep(0.4)

    btn = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//ui-button[@role='button' and .//div[normalize-space()='Download Transactions']]")
        )
    )
    before = _snapshot_downloads()
    driver.execute_script("arguments[0].click();", btn)

    downloaded = _wait_for_new_file(before, timeout=30)
    if downloaded:
        dest_dir = ACCOUNT_DIR / "csv_data"
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = date.today().strftime("%Y_%m_%d") + ".csv"
        dest = dest_dir / filename
        if dest.exists():
            print(f"  [skip] {filename} already exists")
            downloaded.unlink()
        else:
            shutil.move(str(downloaded), str(dest))
            print(f"  [saved] {filename} → cibc/csv_data/")
    else:
        print("  [!] CSV download timed out")


def download_pdfs():
    print_step("Downloading PDF statements", 3, NUM_STEPS, "")
    driver.get(STATEMENTS_URL)

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "monthly-statements"))
    )
    time.sleep(1)

    existing = _get_existing_pdf_dates()

    pdf_btns = driver.find_elements(
        By.XPATH, "//ui-button[contains(@aria-label, 'Download this PDF.')]"
    )

    to_download = []
    for btn in pdf_btns:
        aria_label = btn.get_attribute("aria-label") or ""
        end_date = _parse_stmt_end_date(aria_label)
        if end_date and end_date not in existing:
            to_download.append((btn, end_date, aria_label))

    if not to_download:
        print("  nothing new")
        return

    print(f"  {len(to_download)} new statement(s) to download")
    for btn, end_date, aria_label in to_download:
        print(f"  → {aria_label}")
        before = _snapshot_downloads()
        driver.execute_script("arguments[0].click();", btn)

        downloaded = _wait_for_new_file(before, timeout=30)
        if downloaded:
            dest_dir = ACCOUNT_DIR / "pdf_data"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"onlineStatement_{end_date}.pdf"
            if dest.exists():
                print(f"    [skip] onlineStatement_{end_date}.pdf already exists")
                downloaded.unlink()
            else:
                shutil.move(str(downloaded), str(dest))
                print(f"    [saved] onlineStatement_{end_date}.pdf → cibc/pdf_data/")
        else:
            print(f"    [!] download timed out for {aria_label}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    options = uc.ChromeOptions()
    options.add_experimental_option("prefs", {
        "download.default_directory":         str(DOWNLOAD_DIR),
        "download.prompt_for_download":       False,
        "plugins.always_open_pdf_externally": True,
    })
    driver = uc.Chrome(version_main=147, user_data_dir=str(PROFILE_DIR), options=options)

    driver.get(LOGIN_URL)
    cibc_login()
    download_csv()
    download_pdfs()

    driver.quit()
