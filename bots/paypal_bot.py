#%%
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.chrome.options import Options
from getpass import getpass

from utils.utils import print_step, get_project_root, get_css_selector_from_html


def quit_browser():
    print_step("Quitting...", 1, 1, f"{driver.current_url}")
    driver.quit()


def paypal_login():
    # Step 1 - Enter username
    print_step("Payppal login", 1, 2, f"Entering username - {driver.current_url}")
    textbox = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "input[id='email'][name='login_email'][type='email'][class='hasHelp  validateEmpty   ']"
                          "[required='required'][value=''][autocomplete='username']"
                          "[placeholder='Email or mobile number'][aria-describedby='emailErrorMessage']")))

    textbox.send_keys('ggs8922@gmail.com')

    but = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "button[class='button actionContinue scTrack:unifiedlogin-login-click-next']"
                          "[type='submit'][id='btnNext'][name='btnNext'][value='Next'][pa-marked='1']")))
    but.click()

    # Step 2 - Enter pw
    print_step("Payppal login", 1, 4, f"Enter password - {driver.current_url}")
    textbox = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "input[id='password'][name='login_password'][type='password']"
                          "[class='hasHelp  validateEmpty   pin-password'][required='required'][value='']"
                          "[placeholder='Password'][aria-describedby='passwordErrorMessage']")))
    textbox.send_keys(getpass("Enter your Paypal password:"))

    but = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "button[class='button actionContinue scTrack:unifiedlogin-login-submit']"
                          "[type='submit'][id='btnLogin'][name='btnLogin'][value='Login'][pa-marked='1']")))
    but.click()


def download_custom_report():
    # step 1 - Go to reports page
    print_step("Paypal report download", 1, 4, f"Acessing reports page - {driver.current_url}")
    driver.get("https://business.paypal.com/merchantdata/consumerHome")

    # step 2 - Set report options
    print_step("Paypal report download", 2, 4, f"Setting report options - {driver.current_url}")

    but = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "button[class='btn btn-default dropdown-toggle dropdown'][type='button']"
                          "[data-toggle='dropdown'][aria-haspopup='true'][aria-expanded='true']"
                          "[value='BALANCE_IMPACTING']")))
    but.click()
    choice = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "a[data-value=''][tabindex='0'][class='hanldeDropdownSelect'][title='All transactions']")))
    choice.click()

    but = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "button[type='button'][class='react-dropdown-trigger btn btn-default dropdown-toggle']"
                          "[aria-expanded='false'][data-pp-compdatepicker-id='.0.0.0']"
                          "[style='padding-top: 19px; height: 44px; text-shadow: none;']")))
    but.click()
    choice = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "a[href='#'][class='dropdown-option'][title='Since last download']"
                          "[data-id='Since last download'][data-pp-compdatepicker-id='.0.0.1.0:0.0']")))
    choice.click()

    but = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "button[class='btn btn-default dropdown-toggle dropdown'][type='button']"
                          "[data-toggle='dropdown'][aria-haspopup='true'][aria-expanded='true'][value='CSV']")))
    but.click()
    choice = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "a[data-value='CSV'][tabindex='0'][class='hanldeDropdownSelect']")))
    choice.click()

    # step 3 - Create report
    print_step("Paypal report download", 3, 4, f"Create report - {driver.current_url}")
    but = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.CSS_SELECTOR, "button[id='dlogSubmit'][type='button'][class='btn btn-primary dlogSubmit']")))
    but.click()

    # step 4 - Download report
    input("Press key once report is available for download")
    print_step("Paypal report download", 4, 4, f"Download report - {driver.current_url}")
    but = WebDriverWait(driver, 5).until(ec.presence_of_element_located(
        (By.ID, "download_0")))
    but.click()


if __name__ == "__main__":
    # Selenium needs the google web driver
    PATH = os.path.join(get_project_root(), "bots/web_driver/chromedriver")

    # Set options and create a "driver" object, which basically is the browser
    options = Options()
    # options.add_argument("--headless")  # make the actual browser invisible
    driver = webdriver.Chrome(PATH, options=options)

    driver.get("https://www.paypal.com/ca/signin")

    print("Welcome!")
    paypal_login()
    download_custom_report()

    input("Press any key to quit...")
    quit_browser()
