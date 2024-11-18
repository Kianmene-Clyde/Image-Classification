import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

DRIVERPATH = "D:/chromedriver-win64/chromedriver.exe"
SEARCH_QUERY = "sens interdit"  # Mots-clés de recherche

service = Service(DRIVERPATH)
driver = webdriver.Chrome(service=service)

try:
    driver.get('https://www.google.com/imghp')
    wait = WebDriverWait(driver, 20)
    search_box = wait.until(EC.element_to_be_clickable((By.NAME, 'q')))

    search_box.send_keys(SEARCH_QUERY)
    search_box.send_keys(Keys.RETURN)

    def scroll_to_bottom():
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    scroll_to_bottom()

    images = driver.find_elements(By.CSS_SELECTOR, 'img.rg_i')
    os.makedirs('images', exist_ok=True)

    for index, image in enumerate(images):
        try:
            image_url = image.get_attribute('src')
            if image_url is None:
                image_url = image.get_attribute('data-src')
            if image_url is not None:
                driver.get(image_url)
                driver.save_screenshot(f'images/{SEARCH_QUERY}{index}.png')
        except Exception as e:
            print(f"Erreur pour l'image {index}: {e}")

finally:
    driver.quit()
