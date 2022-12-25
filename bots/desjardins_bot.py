#%%
"""
This scripts automate connection to the web portal (AccesD) of Desjardins for individual users
and automatically download transcripts and/or conciliation files for divers accounts.
GGS - ggs8922@gmail.com
2021-08-07
"""
import datetime
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from getpass import getpass

import Accounts
from utils.utils import print_step, get_project_root


def quit_browser():
    print_step("Quitting...", 1, 1, f"{driver.current_url}")
    driver.quit()


def accessd_login():
    # Step 1 - location confirmation
    print_step("login", 1, 4, f"location (Qc) confirmation - {driver.current_url}")
    try:
        container = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "modale-langue")))
        buttons = container.find_elements_by_tag_name("button")
        for lb in buttons:
            # print(b.text)
            if lb.text == "Confirmer":
                lb.click()
                break
    except Warning:
        raise RuntimeWarning("No confirmation windows popped...")

    # Step 2 - connection
    print_step("login", 2, 4, f"connection - {driver.current_url}")
    try:
        container = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "corps")))
        links = container.find_elements_by_tag_name("a")
        for llink in links:
            # print(l.text)
            if llink.text == "Se connecter\nà AccèsD ou AccèsD Affaires.":
                # print("CLICKING!")
                llink.click()
                break
    except Exception:
        raise RuntimeError("No connection button found!")

    try:
        # Step 3 - News confirmation (temporary 2021-09-06)
        print_step("login", 3, 4, f"News confirmation - {driver.current_url}")
        lcont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "/html/body/app-root/div/div/app-lightbox-onboarding/div/dsd-lightbox")))

        # see https://stackoverflow.com/questions/37384458/how-to-handle-elements-inside-shadow-dom-from-selenium
        sr1 = driver.execute_script("return arguments[0].shadowRoot", lcont)

        for lb in sr1.find_elements_by_tag_name("button"):
            if lb.text == "J\'ai compris":
                lb.click()
                break
    except Warning:
        raise RuntimeWarning("No confirmation dialog seen, it might be time to update code...")

    # Step 4 - Validation
    print_step("login", 4, 4, f"Validation - {driver.current_url}")
    container = driver.find_element_by_id("codeUtilisateur")
    str1 = "030203264532"
    container.send_keys(str1)

    container = driver.find_element_by_id("motDePasse")
    container.send_keys(getpass("Enter pw:", stream=None))

    lcont = driver.find_elements_by_tag_name("button")
    for lb in lcont:
        if lb.text == "Valider":
            lb.click()
            break

    # if a brought to the security question page, must wait for user input
    if driver.current_url == "https://accweb.mouv.desjardins.com/identifiantunique/defi":
        input("Press a key after having entered the answer to the security question on Desjardin's web page...")


def download_transcript(year: int, month: int):
    # Step 1 - Access transcripts files
    print_step("Download transcripts", 1, 3, f"Access transcripts files - {driver.current_url}")
    b = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "btnRelevesDocuments")))
    b.click()

    conts = WebDriverWait(driver, 5).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "col-sm-12")))
    found = False
    for cont in conts:
        for link in cont.find_elements_by_tag_name("a"):
            if link.text == "Comptes":
                found = True
                link.click()
                break
        if found:
            break

    # Step 2 - Download transcripts files
    print_step("Download transcripts", 2, 3, f"Download transcripts files - {driver.current_url}")

    # switch to embedded iframe
    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "modaleIFrame")))
    driver.switch_to.frame(cont)

    # click radio-button for account choice (Desjardins Operations)
    but = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.NAME, "chRadioChoixFolio")))
    but.click()

    # select year and month
    cbox = Select(WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "listePeriodeFormat0"))))
    cbox.select_by_visible_text(str(year))

    cbox = Select(WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "listePeriodeFormatMois00"))))
    cbox.select_by_value(f"0{month}.PDF")

    # select file format
    but = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='radio'][value='fichier']")))
    but.click()

    cbox = Select(WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "chListeFormatPDF"))))
    cbox.select_by_value("CSV")

    # download file
    but = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='submit'][name='Valider']")))
    but.click()

    # reset frame
    driver.switch_to.default_content()

    # Step 3 - rename downloaded transcripts
    print_step("Download transcripts", 3, 3, "rename transcripts files")

    # Move file from Downloads dir to Desjardins data dir in Abacus repo.
    home_dir = "/home/ggsanchez"

    source_dir = os.path.join(home_dir, "Downloads/")
    default_download_name = "releve.csv"
    source_path = os.path.join(source_dir, default_download_name)

    dest_dir = os.path.join(home_dir, "repos/abacus/data/desjardins_csv_files")
    dest_path = os.path.join(dest_dir, f"releve_{year}-{month}.csv")

    if os.path.exists(dest_path):
        print(f"remove existing file... ({dest_path})")
        os.remove(dest_path)
    print(f"Moving dowloaded file... ({source_path} to {dest_path})")
    os.rename(source_path, dest_path)


def download_conciliation():
    # Step 1 - Access transcripts files
    print_step("Download conciliation", 1, 3, f"Access conciliation files - {driver.current_url}")

    b = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "btnRelevesDocuments")))
    b.click()

    link = WebDriverWait(driver, 5).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "a[class='lien-action'][href='#'][onclick='javascript:creerModaleConciliationBancaire();']")))
    link.click()

    # Step 2 - Download transcripts files
    print_step("Download conciliation", 2, 3, f"Download conciliation files - {driver.current_url}")
    # switch to embedded iframe
    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "modaleIFrame")))
    driver.switch_to.frame(cont)

    checkbox = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,
                                        "input[type='checkbox'][name='chListeCompte[0].chChkbCompte'][value='on']")))
    checkbox.click()

    checkbox = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,
                                        "input[type='checkbox'][name='chListeCompte[1].chChkbCompte'][value='on']")))
    checkbox.click()

    # set custom time period from the 1st of the current month to current date
    # we set this time period to be all transactions since the begining of the current month since
    # it match the transaction numbers nicely with the official transcripts
    rbutton = WebDriverWait(driver, 5).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input[type='radio'][name='chPeriode'][value='PI']")))
    rbutton.click()

    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input[type='text'][name='chDateJourMin'][id='idJour']")))
    cont.send_keys("1")

    day = datetime.datetime.today().day
    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input[type='text'][name='chDateJourMax'][id='idJour']")))
    cont.send_keys(str(day))

    month = datetime.datetime.today().month
    cbox = Select(WebDriverWait(driver, 5).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "select[name='chDateMoisMin'][id='idMois']"))))
    cbox.select_by_value(f"{month:02}")

    cbox = Select(WebDriverWait(driver, 5).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "select[name='chDateMoisMax'][id='idMois']"))))
    cbox.select_by_value(f"{month:02}")

    year = datetime.datetime.today().year
    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input[type='text'][name='chDateAnneeMin'][id='idAnnee']")))
    cont.send_keys(f"{year}")

    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input[type='text'][name='chDateAnneeMax'][id='idAnnee']")))
    cont.send_keys(f"{year}")

    rbutton = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='radio'][name='chFormat'][value='CSV']")))
    rbutton.click()

    val_button = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='submit'][name='Valider'][class='btn btn-primary']")))
    val_button.click()

    # Step 3 - Download transcripts files
    print_step("Download conciliation", 3, 3, f"Rename conciliation files - {driver.current_url}")
    # Move file from Downloads dir to Desjardins data dir in Abacus repo.
    home_dir = "/home/ggsanchez"
    source_dir = os.path.join(home_dir, "Downloads/")
    dest_dir = os.path.join(home_dir, "repos/abacus/data/desjardins_csv_files")
    default_download_name = "releve.csv"
    source_path = os.path.join(source_dir, default_download_name)
    dest_path = os.path.join(dest_dir, f"conciliation_{year}-{month}.csv")
    if os.path.exists(dest_path):
        print(f"remove existing file... ({dest_path})")
        os.remove(dest_path)
    print(f"Moving dowloaded file... ({source_path} to {dest_path})")
    os.rename(source_path, dest_path)


def access_credit_card_management_page():
    # Step 1 - drop down menu
    print_step("Access credit card management", 1, 3, f"Selecting options drop down menu - {driver.current_url}")
    drop_down_menu = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-container='#produitFinancement0']")))
    drop_down_menu.click()

    # Step 2 - option selection
    print_step("Access credit card management", 2, 3, f"Selecting card management option - {driver.current_url}")
    link = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[class='lien-action'][data-mw-action-clic='options - gerer la carte']")))
    driver.get(link.get_attribute("href"))

    # Step 3 - link selection
    print_step("Access credit card management", 3, 3, f"Selecting card management link - {driver.current_url}")
    link = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-mw-action-clic='gerer la carte credit - gestion du compte'][class='dsd-b2']")))
    driver.get(link.get_attribute("href"))


def download_visa_pp_transcript(year: int, month: int):
    """Implement this"""
    # TODO access transcript file from credit card manage


def download_visa_pp_conciliation():
    # Step 1 - link selection
    print_step("Download visa conciliation", 1, 5, f"Selecting transcripts link in menu - {driver.current_url}")
    menu = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "menuGauche")))
    for elem in menu.find_elements_by_tag_name("a"):
        if elem.text == "Relevé de compte":
            elem.click()
            break

    # Step 2 - conciliation option selection
    print_step("Download visa conciliation", 2, 5, f"Selecting conciliation link in menu - {driver.current_url}")
    menu = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "menuGauche")))
    for elem in menu.find_elements_by_tag_name("a"):
        if elem.text == "Conciliation / Téléchargement":
            elem.click()
            break

    # Step 3 - set time periode
    print_step("Download visa conciliation", 3, 5, f"setting conciliation period - {driver.current_url}")
    # default selection on web page is good for now

    # Step 4 - select file format
    print_step("Download visa conciliation", 4, 5, f"selecting CSV format - {driver.current_url}")
    but = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='radio'][name='choixFormat'][value='2']")))
    but.click()

    cbox = WebDriverWait(driver, 5).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "select[name='formatTelechargement']")))
    cbox = Select(cbox)
    cbox.select_by_value("ASCII")

    # Step 4 - Download conciliation files
    print_step("Download visa conciliation", 4, 5, f"downloading files - {driver.current_url}")
    but = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='button'][onclick='boutonValider()']")))
    but.click()

    if driver.current_url == "https://www.scd-desjardins.com/GCE/SAInfoCpte":
        # Step 5 - Moving files
        print_step("Download visa conciliation", 5, 5, f"saving text to csv - {driver.current_url}")
        table = driver.find_element_by_tag_name("pre")

        dest_dir = os.path.join(get_project_root(), "data/desjardins_ppcard_csv_files/")
        dest_path = os.path.join(dest_dir, "conciliation.csv")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        with open(dest_path, "w") as f:
            f.write(table.text)
    else:
        # Step 5 - Moving files
        print_step("Download visa conciliation", 5, 5, f"moving files - {driver.current_url}")


if __name__ == "__main__":
    # Selenium needs the google web driver
    PATH = os.path.join(get_project_root(), "bots/web_driver/chromedriver")

    # Set options and create a "driver" object, which basically is the browser
    options = Options()
    # options.add_argument("--headless")  # make the actual browser invisible
    driver = webdriver.Chrome(PATH, options=options)

    driver.get("https://www.desjardins.com/")

    print("Welcome!")
    accessd_login()
    # download_conciliation()
    download_visa_pp_transcript(year=2021, month=9)
    # access_credit_card_management_page()
    # download_visa_pp_conciliation()

    input("Press any key to quit...")
    quit_browser()
