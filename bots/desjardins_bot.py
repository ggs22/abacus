#%%
"""
This scripts automate connection to the web portal (AccesD) of Desjardins for individual users
and automatically download transcripts and/or conciliation files for divers accounts.
GGS - ggs8922@gmail.com
2021-08-07
"""
import datetime
import os
import colorama

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from getpass import getpass

import Accounts
from helpers.helpers import print_step, get_project_root

# Selenium needs the google web driver
PATH = os.path.join(get_project_root(), 'bots/web_driver/chromedriver')

# Set options and create a 'driver' object, which basically is the browser
options = Options()
# options.add_argument('--headless')  # make the actual browser invisible
driver = webdriver.Chrome(PATH, options=options)

driver.get('https://www.desjardins.com/')


def quit_browser():
    # step 7 - Quit browser
    print_step('Quitting...', 1, 1, f'{driver.current_url}')
    driver.quit()


def accessd_login():
    # Step 1 - location confirmation
    print_step('login', 1, 4, f'location (Qc) confirmation - {driver.current_url}')
    try:
        container = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'modale-langue')))
        buttons = container.find_elements_by_tag_name('button')
        for lb in buttons:
            # print(b.text)
            if lb.text == 'Confirmer':
                lb.click()
                break
    except Warning:
        raise RuntimeWarning('No confirmation windows popped...')

    # Step 2 - connection
    print_step('login', 2, 4, f'connection - {driver.current_url}')
    try:
        container = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'corps')))
        links = container.find_elements_by_tag_name('a')
        for llink in links:
            # print(l.text)
            if llink.text == 'Se connecter\nà AccèsD ou AccèsD Affaires.':
                # print("CLICKING!")
                llink.click()
                break
    except Exception:
        raise RuntimeError('No connection button found!')

    # Step 3 - News confirmation (temporary 2021-09-06)
    print_step('login', 3, 4, f'News confirmation - {driver.current_url}')
    lcont = driver.find_element_by_xpath('/html/body/app-root/div/div/app-lightbox-onboarding/div/dsd-lightbox')

    # see https://stackoverflow.com/questions/37384458/how-to-handle-elements-inside-shadow-dom-from-selenium
    sr1 = driver.execute_script('return arguments[0].shadowRoot', lcont)

    for lb in sr1.find_elements_by_tag_name('button'):
        if lb.text == 'J\'ai compris':
            lb.click()
            break

    # Step 4 - Validation
    print_step('login', 4, 4, f'Validation - {driver.current_url}')
    container = driver.find_element_by_id('codeUtilisateur')
    str1 = '030203264532'
    container.send_keys(str1)

    container = driver.find_element_by_id('motDePasse')
    container.send_keys(getpass('Enter pw:', stream=None))

    lcont = driver.find_elements_by_tag_name('button')
    for lb in lcont:
        if lb.text == 'Valider':
            lb.click()
            break


def download_transcript(year: int, month: int):
    # Step 1 - Access transcripts files
    print_step('Download transcripts', 1, 3, f'Access transcripts files - {driver.current_url}')
    b = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'btnRelevesDocuments')))
    b.click()

    conts = WebDriverWait(driver, 5).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'col-sm-12')))
    found = False
    for cont in conts:
        for link in cont.find_elements_by_tag_name('a'):
            if link.text == 'Comptes':
                found = True
                link.click()
                break
        if found:
            break

    # Step 2 - Download transcripts files
    print_step('Download transcripts', 2, 3, f'Download transcripts files - {driver.current_url}')

    # switch to embedded iframe
    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'modaleIFrame')))
    driver.switch_to.frame(cont)

    # click radio-button for account choice (Desjardins Operations)
    but = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.NAME, 'chRadioChoixFolio')))
    but.click()

    # select year and month
    cbox = Select(WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, 'listePeriodeFormat0'))))
    cbox.select_by_visible_text(str(year))

    cbox = Select(WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, 'listePeriodeFormatMois00'))))
    cbox.select_by_value(f'0{month}.PDF')

    # select file format
    but = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='radio'][value='fichier']")))
    but.click()

    cbox = Select(WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, 'chListeFormatPDF'))))
    cbox.select_by_value('CSV')

    # download file
    but = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='submit'][name='Valider']")))
    but.click()

    # reset frame
    driver.switch_to.default_content()

    # Step 3 - rename downloaded transcripts
    print_step('Download transcripts', 3, 3, 'rename transcripts files')

    # Move file from Downloads dir to Desjardins data dir in Abacus repo.
    home_dir = '/home/ggsanchez'

    source_dir = os.path.join(home_dir, 'Downloads/')
    default_download_name = 'releve.csv'
    source_path = os.path.join(source_dir, default_download_name)

    dest_dir = os.path.join(home_dir, 'repos/abacus/data/desjardins_csv_files')
    dest_path = os.path.join(dest_dir, f'releve_{year}-{month}.csv')

    if os.path.exists(dest_path):
        print(f'remove existing file... ({dest_path})')
        os.remove(dest_path)
    print(f'Moving dowloaded file... ({source_path} to {dest_path})')
    os.rename(source_path, dest_path)


def download_conciliation():
    # Step 1 - Access transcripts files
    print_step('Download conciliation', 1, 3, f'Access conciliation files - {driver.current_url}')

    b = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'btnRelevesDocuments')))
    b.click()

    conts = driver.find_elements_by_css_selector("div[class='col-sm-12 col-md-12']")
    for cont in conts:
        if cont.text == 'Conciliation bancaire':
            print('found!')
            cont.click()
            break

    # Step 2 - Download transcripts files
    print_step('Download conciliation', 2, 3, f'Download conciliation files - {driver.current_url}')
    # switch to embedded iframe
    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'modaleIFrame')))
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
    rbutton = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='radio'][name='chPeriode'][value='PI']")))
    rbutton.click()

    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'][name='chDateJourMin'][id='idJour']")))
    cont.send_keys('1')

    day = datetime.datetime.today().day
    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'][name='chDateJourMax'][id='idJour']")))
    cont.send_keys(str(day))

    month = datetime.datetime.today().month
    cbox = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "select[name='chDateMoisMin'][id='idMois']")))
    cbox.select_by_value(f'{month:02}')

    cbox = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "select[name='chDateMoisMax'][id='idMois']")))
    cbox.select_by_value(f'{month:02}')

    year = datetime.datetime.today().year
    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'][name='chDateAnneeMin'][id='idAnnee']")))
    cont.send_keys(f'{year}')
    cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'][name='chDateAnneeMax'][id='idAnnee']")))
    cont.send_keys(f'{year}')

    rbutton = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='radio'][name='chFormat'][value='CSV']")))
    rbutton.click()

    val_button = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='submit'][name='Valider'][class='btn btn-primary']")))
    val_button.click()

    # Step 3 - Download transcripts files
    print_step('Download conciliation', 3, 3, f'Rename conciliation files - {driver.current_url}')
    # Move file from Downloads dir to Desjardins data dir in Abacus repo.
    home_dir = '/home/ggsanchez'
    source_dir = os.path.join(home_dir, 'Downloads/')
    dest_dir = os.path.join(home_dir, 'repos/abacus/data/desjardins_csv_files')
    default_download_name = 'releve.csv'
    source_path = os.path.join(source_dir, default_download_name)
    dest_path = os.path.join(dest_dir, f'conciliation_{year}-{month}.csv')
    if os.path.exists(dest_path):
        print(f'remove existing file... ({dest_path})')
        os.remove(dest_path)
    print(f'Moving dowloaded file... ({source_path} to {dest_path})')
    os.rename(source_path, dest_path)


# "span[class='pull-left'][aria-hidden='true']"

cont = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'][name='chDateJourMin'][id='idJour']")))


def access_credit_card_management_page():
    drop_down_menu = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-container='#produitFinancement0']")))
    drop_down_menu.click()

    link = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[class='lien-action'][data-mw-action-clic='options - gerer la carte']")))
    driver.get(link.get_attribute('href'))

    link = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-mw-action-clic='gerer la carte credit - gestion du compte'][class='dsd-b2']")))
    driver.get(link.get_attribute('href'))


def download_visa_pp_transcript(year: int, month: int):
    """Implement this"""
    # TODO access transcript file from credit card manage


def download_visa_pp_conciliation():
    """Implement this"""
    # TODO access conciliation file from credit card manage


if __name__ == '__main__':
    print('Welcome!')
    accessd_login()
    access_credit_card_management_page()

    # input("Press any key to quit...")
    # quit_browser()
