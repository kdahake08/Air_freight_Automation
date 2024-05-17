import sys
import time
import random
import pytz
import os
from datetime import datetime, timedelta
import numpy as np
from database import *
from io import StringIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException
from sqlalchemy.exc import IntegrityError
pd.set_option('future.no_silent_downcasting', True)

log_file_path = r'C:\Users\Karti\Downloads\TGL_Airfreight_Automation\dist\Output_Files\app.log'

logging.basicConfig(filename=log_file_path, filemode ='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("selenium").setLevel(logging.ERROR)
logging.getLogger("selenium.webdriver").setLevel(logging.ERROR)
logging.getLogger("webdriver").setLevel(logging.ERROR)

#Read the master files into variables
input_sheet_path = r'C:\Users\Karti\Downloads\TGL_Airfreight_Automation\dist\input_config_sheet.csv'
chrome_driver_path = r'C:\Users\Kartik\Documents\TGL\chromedriver_win32\chromedriver'


# DEFINE RANDOM DELAY SELECTION (Follows Normal Distribution)
def selection_delay(min_delay=3, mean=5, sd=2, max_delay=7):
	sdelay = random.normalvariate(mean, sd)
	sdelay = max(min_delay, min(max_delay, sdelay))
##logging.info(f"Delaying for {sdelay:.2f} seconds")
	time.sleep(sdelay)

# DEFINE RANDOM DELAY AFTER EXTRACTION (Follows Normal Distribution)
def extraction_delay(min_delay=7, mean=10, sd=3, max_delay=15):
	edelay = random.normalvariate(mean, sd)
	edelay = max(min_delay, min(max_delay, edelay))
	#logging.info(f"Delaying for {(edelay/60):.2f} minutes")
	time.sleep(edelay)

# Scraping Step 1: LOGIN function
def login(driver, login_url, username, password, username_xpath, password_xpath, login_button_xpath, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        try:
            driver.get(login_url)
            driver.find_element(By.XPATH, username_xpath).send_keys(username)
            selection_delay()
            driver.find_element(By.XPATH, password_xpath).send_keys(password)
            selection_delay()
            driver.find_element(By.XPATH, login_button_xpath).click()
            selection_delay()

            if 'Single Destination' in driver.page_source:
                logging.info("LOGIN SUCCESSFUL, moving to Single destination page")
                return
            else:
                logging.error("LOGIN ERROR, trying again")
                attempts += 1
                selection_delay()

        except Exception as e:
            logging.error("An error occurred while trying to login to Express Rates, Retrying... %s", str(e))
            attempts += 1
            selection_delay()

    if attempts == max_attempts:
        logging.error("Maximum login attempts reached. Unable to login.")
        raise Exception("Login failed after max attempts")


# Scraping Step 2: Navigate to Single Destination search page
def navigate_and_create_POD_ID_DF(driver, single_destination_xpath, xpath_for_chzn_results):
	global pod_id_df

	try:
		driver.find_element(By.XPATH, single_destination_xpath).click()
		selection_delay()
		logging.info("Current URL: %s", driver.current_url)

		wait = WebDriverWait(driver, 10)
		container = wait.until(ec.presence_of_element_located((By.XPATH, xpath_for_chzn_results)))

		elements = container.find_elements(By.TAG_NAME, "li")

		ids = []
		codes = []
		for element in elements:
			id_attribute = element.get_attribute('id')
			text_content = element.get_attribute("textContent").strip()
			split_text = text_content.split(' - ')
			if len(split_text) > 0:
				code = split_text[0].strip()
				ids.append(id_attribute)
				codes.append(code)

		pod_id_df = pd.DataFrame({'PODCODE': codes, 'PODID': ids})
		# Assuming 'engine' is defined elsewhere in your code for SQL operations
		#pod_id_df.to_sql('pod_ids', engine, if_exists='replace', index=False)

		return pod_id_df

	except Exception as e:
		logging.error(f"Error navigating and creating POD_ID_DF")
		raise Exception("POD_ID_DF creation failed")


#Scraping Step 3.1: EXTRACT THE POD_ID from pod_id_df
def find_pod_id(pod_id_df, destination):
	if destination in pod_id_df['PODCODE'].values:
		# Get the corresponding 'PODID' value
		pod_id = pod_id_df.loc[pod_id_df['PODCODE'] == destination, 'PODID'].iloc[0]
		return pod_id
	else:
		logging.error("Destination code not found in the POD_ID")


def select_tariff(driver, tariff_xpath, tariff):
	wait = WebDriverWait(driver, 10)
	attempts = 3

	try:
		for i in range(attempts):
			try:
				tariff_dropdown = wait.until(ec.element_to_be_clickable((By.XPATH, tariff_xpath)))
				Select(tariff_dropdown).select_by_value(str(tariff))
				selection_delay()
				break  # If successful, break the loop
			except TimeoutException:
				if i < attempts - 1:  # If not the last attempt, log and continue
					logging.warning(f"Attempt {i + 1} failed while selecting tariff. Retrying...")
					logging.info("Current URL: %s", driver.current_url)
					if str(driver.current_url) == 'https://www.expressrates.in/login.php':
						logging.info("Express rates logged out the scraper, Retrying...")
						break
				else:  # If the last attempt, log and break
					logging.error(
						f"Failed to select tariff after {attempts} attempts. Moving forward with default tariff :General Cargo")
					logging.info("Current URL: %s", driver.current_url)
					raise Exception("Failed to select tariff after multiple attempts")

	except Exception as e:
		logging.error(f"Error selecting tariff %s", str(e))
		raise Exception("Failed to select tariff")


def select_origin(driver, origin_xpath, origin):
	wait = WebDriverWait(driver, 10)
	attempts = 3

	for i in range(attempts):
		try:
			origin_dropdown = wait.until(ec.element_to_be_clickable((By.XPATH, origin_xpath)))
			Select(origin_dropdown).select_by_value(origin)
			logging.info("Selected origin: %s", origin)
			selection_delay()
			break  # If successful, break the loop
		except TimeoutException:
			if i < attempts - 1:  # If not the last attempt, log and continue
				logging.warning(f"Attempt {i + 1} failed while selecting origin. Retrying...")
				logging.info("Current URL: %s", driver.current_url)
				if str(driver.current_url) == 'https://www.expressrates.in/login.php':
					logging.info("Express rates logged out the scraper, Retrying...")
					break
			else:  # If the last attempt, log and break
				logging.error(f"Failed to select origin after {attempts} attempts. Retrying...")
				logging.info("Current URL: %s", driver.current_url)
				raise Exception("Failed to select origin")


def select_destination(driver, destination, pod_id):
	wait = WebDriverWait(driver, 10)
	attempts = 3

	for i in range(attempts):
		try:
			destination_dropdown_span = wait.until(
				ec.element_to_be_clickable((By.CSS_SELECTOR, "#idDestination_chzn span")))
			destination_dropdown_span.click()

			destination_input = wait.until(
				ec.visibility_of_element_located((By.CSS_SELECTOR, "#idDestination_chzn input")))
			destination_input.send_keys(destination)

			pod_option = wait.until(ec.element_to_be_clickable((By.ID, pod_id)))
			pod_option.click()

			logging.info("Selected destination: %s", destination)
			selection_delay()
			break  # If successful, break the loop
		except TimeoutException:
			if i < attempts - 1:  # If not the last attempt, log and continue
				logging.warning(f"Attempt {i + 1} failed while selecting destination. Retrying...")
				logging.info("Current URL: %s", driver.current_url)
				if str(driver.current_url) == 'https://www.expressrates.in/login.php':
					logging.info("Express rates logged out the scraper, trying again")
					break
			else:  # If the last attempt, log and break
				logging.error(f"Failed to select destination after {attempts} attempts.")
				logging.info("Current URL: %s", driver.current_url)
				raise Exception("Failed to select destination")


def SEARCH_RATES(driver, search_button_xpath):
	wait = WebDriverWait(driver, 10)
	attempts = 3

	for i in range(attempts):
		try:
			search_button = wait.until(ec.element_to_be_clickable((By.XPATH, search_button_xpath)))
			search_button.click()
			logging.info("SEARCH INITIATED")
			break  # If successful, break the loop
		except TimeoutException:
			if i < attempts - 1:  # If not the last attempt, log and continue
				logging.warning(f"Attempt {i + 1} failed while initiating search. Retrying...")
			else:  # If the last attempt, log and break
				logging.error(f"Failed to initiate search after {attempts} attempts.")
				raise Exception("Failed to initiate search")


#DEFINE TIMESTAMP UTC+5:30
def Created_At():
	ist_timezone = pytz.timezone('Asia/Kolkata')
	current_time_ist = datetime.now(ist_timezone)
	return current_time_ist.strftime('%d-%m-%Y %H:%M:%S')

# Scraping step 3: Main loop to take inputs from input sheet and call all the scraping function and return a df of scraped data
def SEARCH_AND_EXTRACT_DATA(driver, input_search_params_df, pod_id_df, tariff_xpath, origin_xpath, search_button_xpath,
							tariff):
	dfs = []
	global processed_port_pairs_log_df

	for index, row in input_search_params_df.iterrows():
		origin = row.iloc[0]
		destination = row.iloc[1]

		try:
			pod_id = find_pod_id(pod_id_df, destination)  # STEP_1
			select_tariff(driver, tariff_xpath, tariff)  # STEP_2
			select_origin(driver, origin_xpath, origin)  # STEP_3
			select_destination(driver, destination, pod_id)  # STEP_4
			SEARCH_RATES(driver, search_button_xpath)  # STEP_5
		except Exception as e:
			logging.error(f"Error processing {origin} - {destination} pair")
			raise Exception(f"Failed to process {origin} - {destination} pair")

		time.sleep(30)  # Wait for the results to load and scrape the data
		html_content = driver.page_source
		try:
			df = pd.read_html(StringIO(html_content))[-1]
			df["From"] = origin
			df["To"] = destination
			df["Created_At"] = Created_At()
		except Exception as e:
			logging.error(f"No data received for {origin} - {destination} pair")

		try:
			# Check if the columns exist in the DataFrame
			if all(col in df.columns for col in ['Delete', 'Check All', 'AMS']):
				df = df.reset_index().drop(['Delete', 'Check All', 'AMS'], axis=1)
				new_row = pd.DataFrame({'Origin': [origin], 'Destination': [destination], 'Created_At': [Created_At()], 'Remarks': ['Data Received']})
				processed_port_pairs_log_df = pd.concat([processed_port_pairs_log_df, new_row], ignore_index=True)
				dfs.append(df)
			else:
				new_row = pd.DataFrame({'Origin': [origin], 'Destination': [destination], 'Created_At': [Created_At()], 'Remarks': ['No Data Received']})
				processed_port_pairs_log_df = pd.concat([processed_port_pairs_log_df, new_row], ignore_index=True)
				logging.error(f"No data received for {origin} - {destination} pair")
		except KeyError as ke:
			logging.error(f"No data received for {origin} - {destination} pair %s")

		driver.refresh()
		logging.info(f'Search completed for: {origin} - {destination}, Moving to next combo if present')
		extraction_delay()  # Wait before starting the next iteration

	#logging.info('That was the last combo.')
	if len(dfs) > 0:
		final_df = pd.concat(dfs, ignore_index=True)
		return final_df
	else:
		logging.error(f"No data received from the express rates")
		raise Exception("No data received from the express rates")


# Scraping Step 4: LOG OUT
def log_out(driver, logout_button_xpath):
	try:
		wait = WebDriverWait(driver, 10)
		logout_button = wait.until(ec.element_to_be_clickable((By.XPATH, logout_button_xpath)))
		logout_button.click()
		logging.info("Logged out successfully.")
		wait.until(ec.presence_of_element_located((By.TAG_NAME, "body")))
		return driver.current_url
	except TimeoutException:
		logging.error("TimeoutException: Unable to locate logout button within the specified time.")
		raise Exception("Failed to logout")
	except Exception as e:
		logging.error(f"Error during logout")
		raise Exception("Failed to logout")

# Define a function to log eliminated records
def log_eliminated_records(df, reason):
	# Append eliminated records to the DataFrame
	global eliminated_records
	eliminated_df = df.copy()
	eliminated_df['Elimination Reason'] = reason
	eliminated_records = pd.concat([eliminated_records, eliminated_df], ignore_index=True)

#Function to replace blank spaces and reaplce Nan with 0
def clean_dataframe(df):
	try:
		# Check if input is a DataFrame
		if not isinstance(df, pd.DataFrame):
			logging.error('Input is not a dataframe')
			raise ValueError("Input is not a pandas DataFrame")

		# Create a copy of the original DataFrame
		cleaned_df = df.copy()

		# Replace empty strings with NaN
		for col in cleaned_df.columns:
			cleaned_df[col] = cleaned_df[col].apply(lambda x: '' if isinstance(x, str) and x.strip() == '' else x)

		return cleaned_df

	except Exception as e:
		logging.error("An error occurred while cleaning the input data")
		return df  # Return original DataFrame if an error occurs


# Step funct 1: From, To, Airline Code, Service from scraped data to output_df
def populate_from_to_airline_service(output_df, df):
	output_df['From'] = df['From']
	output_df['To'] = df['To']
	output_df['Airline Code'] = df['Airline']
	output_df['Service'] = df['Service']
	output_df['Date'] = df['Date']
	output_df['Created_At'] = df['Created_At']
	return output_df


#Step 2: Replace old codes with new codes using Airline Port Mapping sheet and remove duplicates
def replace_oldcode_with_newcode(df, airline_port_mapping_df):
	try:
		code_mapping = dict(zip(airline_port_mapping_df['OldCode'], airline_port_mapping_df['NewCode']))
		df['To'] = df['To'].map(code_mapping).fillna(df['To'])
		df = df.drop_duplicates(subset=['From', 'To', 'Airline Code', 'Service'])
		return df
	except Exception as e:
		logging.error("An error occurred while replacing old POD codes with new POD codes, Please check the master file and make sure there are no mistakes")
		return df  # Return the original DataFrame in case of errors


# Airline Step 3: Validate airlines against Airlines_Cleanup sheet
def validate_and_cleanup_airlines(output_df, airlines_cleanup_df, log_eliminated_records):
	try:
		required_columns_output = ['Airline Code']
		required_columns_cleanup = ['Pattern_To_Match', 'Airline', 'Rename_To']

		# Validate column existence
		if not all(col in output_df.columns for col in required_columns_output):
			raise ValueError(f"Required column 'Airline Code' not found in output_df.")
		if not all(col in airlines_cleanup_df.columns for col in required_columns_cleanup):
			raise ValueError(
				f"Required columns 'Pattern_To_Match', 'Airline', and 'Rename_To' not found in airlines_cleanup_df.")

		for index, row in output_df.iterrows():
			airline_name = row['Airline Code'].lower()  # Convert to lower case
			trigger_values = set(airlines_cleanup_df['Pattern_To_Match'].str.lower())

			if any(trigger in airline_name for trigger in trigger_values):
				exact_match = airlines_cleanup_df[airlines_cleanup_df['Airline'].str.lower() == airline_name]
				if not exact_match.empty:
					new_airline_name = exact_match['Rename_To'].iloc[0]
					output_df.at[index, 'Airline Code'] = new_airline_name
				else:
					log_eliminated_records(output_df.loc[[index]],
										   reason='Record eliminated during the Airlines Cleanup Process')  # Remove other matching airlines not listed in the 'Airline' column of 'Airlines_Cleanup' sheet
					output_df.drop(index, inplace=True)
	except Exception as e:
		logging.error("An error occurred while validating Airline against Airlines Cleanup Sheet")

	return output_df


#Airline Step 4: Validate Airline names from AirlineMaster sheet : Rename names to strip trailing or leading words
def validate_airlines_against_airlinemaster(output_df, airlinemaster_df, log_eliminated_records):
	try:
		# Check if required columns exist in output_df and airlinemaster_df
		required_columns_output = ['Airline Code']
		required_columns_master = ['Name']
		if not all(col in output_df.columns for col in required_columns_output):
			raise ValueError("Required column 'Airline Code' not found in output_df.")
		if not all(col in airlinemaster_df.columns for col in required_columns_master):
			raise ValueError("Required column 'Name' not found in airlinemaster_df.")

		# Convert airline names to lowercase for comparison
		airline_names = airlinemaster_df['Name'].str.lower()
		output_df_airlines = output_df['Airline Code'].str.lower()

		# Create a dictionary to map lowercase airline names to their original case counterparts
		airline_name_mapping = dict(zip(airlinemaster_df['Name'].str.lower(), airlinemaster_df['Name']))

		# Iterate over the rows where the condition is False
		for index, row in output_df[~output_df_airlines.isin(airline_names)].iterrows():
			# Check if any word in the airline name from airlinemaster_df matches the airline code
			matching_airline_names = [name for name in airline_names if row['Airline Code'].lower().startswith(name.lower())]
			if matching_airline_names:
				# Replace the airline code in output_df with the matching airline name
				matched_name = matching_airline_names[0]
				original_name = airline_name_mapping[matched_name]  # Retrieve original case format
				output_df.at[index, 'Airline Code'] = original_name
			else:
				# Remove the record if no match is found
				log_eliminated_records(output_df.loc[[index]],
									   reason='Record eliminated during the validating Airline name against Airlinemaster sheet')
				output_df.drop(index, inplace=True)

	except Exception as e:
		logging.error("An error occurred during validating Airline against AirlineMaster sheet")

	return output_df


# Airline Step 5: Validate POD-Airline Mapping
def validate_pod_airline_mapping(output_df, pod_airline_mapping_df, log_eliminated_records):
	try:
		# Check if required columns exist in output_df
		required_columns = ['To', 'Airline Code']
		if not all(col in output_df.columns for col in required_columns):
			raise ValueError("Required columns 'To' and 'Airline Code' not found in output_df.")

		pod_airline_combinations = set(zip(output_df['To'].str.lower(), output_df['Airline Code'].str.lower()))

		# Check if required columns exist in pod_airline_mapping_df
		if not all(col in pod_airline_mapping_df.columns for col in ['Code', 'AirlineMapping']):
			raise ValueError("Required columns 'Code' and 'AirlineMapping' not found in pod_airline_mapping_df.")

		pod_mapped_combinations = set(
			zip(pod_airline_mapping_df['Code'].str.lower(), pod_airline_mapping_df['AirlineMapping'].str.lower()))
		pod_missing_combinations = pod_airline_combinations - pod_mapped_combinations

		# Check if output_df is empty after filtering
		if len(pod_missing_combinations) > 0:
			log_eliminated_records(output_df[output_df.apply(
				lambda x: (x['To'].lower(), x['Airline Code'].lower()) in pod_missing_combinations, axis=1)],
								   reason='Record eliminated during validation against POD-Airline Mapping')
			output_df = output_df[
				output_df.apply(lambda x: (x['To'].lower(), x['Airline Code'].lower()) not in pod_missing_combinations,
								axis=1)]
		else:
			logging.info("No missing combinations found in output_df")
		#print("No missing combinations found in output_df")

	except Exception as e:
		logging.error("An error occurred while validating Airline against POD mapping")
	# Handle the error gracefully, such as logging the error message or taking corrective action
	return output_df


# Airline Step 6: Validate POL-Airline Mapping
def validate_pol_airline_mapping(output_df, pol_airline_mapping_df, log_eliminated_records):
	try:

		# Check if required columns exist in output_df and pol_airline_mapping_df
		required_columns_output = ['From', 'Airline Code']
		required_columns_mapping = ['POLCode', 'AirlineCode']
		if not all(col in output_df.columns for col in required_columns_output):
			raise ValueError("Required columns 'From' and 'Airline Code' not found in output_df.")
		if not all(col in pol_airline_mapping_df.columns for col in required_columns_mapping):
			raise ValueError("Required columns 'POLCode' and 'AirlineCode' not found in pol_airline_mapping_df.")

		# Convert columns to lowercase for comparison
		pol_airline_combinations = set(zip(output_df['From'].str.lower(), output_df['Airline Code'].str.lower()))
		pol_mapped_combinations = set(
			zip(pol_airline_mapping_df['POLCode'].str.lower(), pol_airline_mapping_df['AirlineCode'].str.lower()))
		pol_missing_combinations = pol_airline_combinations - pol_mapped_combinations

		# Filter output_df based on missing combinations
		log_eliminated_records(output_df[output_df.apply(
			lambda x: (x['From'].lower(), x['Airline Code'].lower()) in pol_missing_combinations, axis=1)],
							   reason='Record eliminated during validation against POL-Airline Mapping')
		output_df = output_df[
			output_df.apply(lambda x: (x['From'].lower(), x['Airline Code'].lower()) not in pol_missing_combinations,
							axis=1)]

	except Exception as e:
		logging.error("An error occurred while validating Airline against POD mapping")
	# Handle the error gracefully, such as logging the error message or taking corrective action

	return output_df


#Step 7: Populating output_df from the source data and adding default markups to weight breaks and check for blank spaces
def apply_markups(output_df, df, min_markup_amount, normal_markup_amount, m_45_markup_amount, m_100_markup_amount,
				  m_250_markup_amount, m_300_markup_amount, m_500_markup_amount, m_1000_markup_amount,
				  m_2000_markup_amount):
	try:
		# logging.info(output_df['Min'])
		# logging.info(df['Min'])
		# logging.info(df.columns)
		output_df['Min'] = df['Min'] + min_markup_amount if 'Min' in df.columns and (df['Min'] != 0).any() else 0
		output_df['>=45'] = df['+45'] + m_45_markup_amount if '+45' in df.columns and (df['+45'] != 0).any() else 0
		output_df['>=100'] = df['100'] + m_100_markup_amount if '100' in df.columns and (df['100'] != 0).any() else 0
		output_df['Normal'] = df['Normal'] + normal_markup_amount if 'Normal' in df.columns and (
			df['Normal'] != 0).any() else 0
		output_df['>=250'] = df['+250'] + m_250_markup_amount if '+250' in df.columns and (df['+250'] != 0).any() else 0
		output_df['>=300'] = df['+300'] + m_300_markup_amount if '+300' in df.columns and (df['+300'] != 0).any() else 0
		output_df['>=500'] = df['+500'] + m_500_markup_amount if '+500' in df.columns and (df['+500'] != 0).any() else 0
		output_df['>=1000'] = df['+1000'] + m_1000_markup_amount if '+1000' in df.columns and (
			df['+1000'] != 0).any() else 0
		output_df['>=2000'] = df['+2000'] + m_2000_markup_amount if '+2000' in df.columns and (
			df['+2000'] != 0).any() else 0
	except Exception as e:
		print(e)
	return output_df


#Supporting functions for step 8 : Functions to split Rates using rate_separator
def get_rate(rate):
	rate_split = rate.split(rate_separator)
	if len(rate_split) > 1:
		return rate_split[0]
	else:
		return "error"

def get_basis(rate):
	rate_split = rate.split(rate_separator)
	if len(rate_split) > 1:
		return rate_split[1]
	else:
		return "error"

#Step 8: Populate Rates and Basis columns
def calculate_rates_basis(output_df, df, get_rate, get_basis, ams_rate, ams_basis):
	try:
		output_df['WSC Rate'] = df['WSC'].map(get_rate)
		output_df['FSC Rate'] = df['FSC'].map(get_rate)
		output_df['XRay Rate'] = df['Xray'].map(get_rate)
		output_df['MCC Rate'] = df['MCC'].map(get_rate)
		output_df['CTG Rate'] = df['CTG'].map(get_rate)
		output_df['WSC Basis'] = df['WSC'].map(get_basis)
		output_df['FSC Basis'] = df['FSC'].map(get_basis)
		output_df['XRay Basis'] = df['Xray'].map(get_basis)
		output_df['MCC Basis'] = df['MCC'].map(get_basis)
		output_df['CTG Basis'] = df['CTG'].map(get_basis)
		output_df['AMS Rate'] = ams_rate
		output_df['AMS Basis'] = ams_basis

	except Exception as e:
		print(e)

	return output_df

#Step 9: Calculate Total C and Total G columns
def calculate_total_c_g(output_df):
	rate_columns = ['FSC Rate', 'WSC Rate', 'XRay Rate', 'MCC Rate', 'CTG Rate']
	basis_columns = ['FSC Basis', 'WSC Basis', 'XRay Basis', 'MCC Basis', 'CTG Basis']

	# Convert rate columns to numeric
	try:
		output_df[rate_columns] = output_df[rate_columns].apply(pd.to_numeric, errors='coerce')

		# Calculate 'Total C' and 'Total G' based on basis columns
		output_df['Total C'] = output_df.apply(
			lambda row: sum(row[col1] for col1, col2 in zip(rate_columns, basis_columns) if row[col2] == 'C'), axis=1)
		output_df['Total G'] = output_df.apply(
			lambda row: sum(row[col1] for col1, col2 in zip(rate_columns, basis_columns) if row[col2] == 'G'), axis=1)
	except Exception as e:
		print(e)

	return output_df

def fillna(output_df):
	try:
		columns_to_exclude = ['From', 'To', 'Airline Code', 'Service']
		present_columns = [col for col in output_df.columns if col not in columns_to_exclude]
		output_df[present_columns] = output_df[present_columns].fillna(0)
	except KeyError:
		pass
	return output_df

def format_value(value):
    if pd.isna(value):  # Check for NaN values
        return np.nan
    elif value == 0.00:
        return 0
    elif isinstance(value, float):
        return "{:.2f}".format(value).rstrip('0').rstrip('.')
    else:
        return value

#Step 10: Remove duplicate airlines by checking lower rates in columns min, normal, ....2000
# Sort the DataFrame by 'From', 'To', 'Airline Code'
def remove_duplicate_airlines(output_df, log_eliminated_records):
	output_df.sort_values(by=['From', 'To', 'Airline Code'], inplace=True)
	output_df.reset_index(drop=True, inplace=True)  # Reset index

	prev_from, prev_to, prev_airline, prev_service = None, None, None, None  # Initialize variables to track the previous row's values

	# Initialize list to store indices
	same_service_indices = []

	# Iterate over the DataFrame
	for index, row in output_df.iterrows():
		# Check if the current row matches the previous one on the first 3 columns
		if (row['From'], row['To'], row['Airline Code']) == (prev_from, prev_to, prev_airline):
			# Check if the 'Service' column is the same
			if row['Service'] == prev_service:
				# Store the indices of the current and previous rows
				same_service_indices.append(index)
				same_service_indices.append(index - 1)

			# Update the previous row's values
		prev_from, prev_to, prev_airline, prev_service = row['From'], row['To'], row['Airline Code'], row['Service']

	# Compare the values in the 'Normal' column and other columns for duplicate rows with same service
	for i in range(0, len(same_service_indices), 2):
		current_index = same_service_indices[i]
		next_index = same_service_indices[i + 1]
		current_row = output_df.iloc[current_index]
		next_row = output_df.iloc[next_index]
		columns_to_compare = ['Normal', '>=45', '>=100', '>=250', '>=300', '>=500', '>=1000', '>=2000', 'Min']
		for column in columns_to_compare:
			if int(current_row[column]) < int(next_row[column]):
				# Drop the next row if current row has lower value in any other column
				log_eliminated_records(output_df.loc[[next_index]],
									   reason='Duplicate record with lower value in column {}'.format(column))
				output_df.drop(next_index, inplace=True)
				break
			elif int(current_row[column]) > int(next_row[column]):
				# Drop the current row if next row has lower value in any other column
				log_eliminated_records(output_df.loc[[current_index]],
									   reason='Duplicate record with lower value in column {}'.format(column))
				output_df.drop(current_index, inplace=True)
				break

	# Print the updated DataFrame
	return output_df

#Step 11: Apply dynamic markups from markup_exceptions sheet
def apply_dynamic_markups(markup_exceptions_df, output_df, min_markup_amount, normal_markup_amount, m_45_markup_amount,
						  m_100_markup_amount, m_250_markup_amount, m_300_markup_amount, m_500_markup_amount,
						  m_1000_markup_amount, m_2000_markup_amount):
	markup_exceptions_df['POL'] = markup_exceptions_df['POL'].str.strip()
	markup_exceptions_df['POD'] = markup_exceptions_df['POD'].str.strip()
	markup_exceptions_df['Airline'] = markup_exceptions_df['Airline'].str.strip()

	# Remove leading and trailing spaces from 'From', 'To', and 'Airline Code' columns in final_df
	output_df['From'] = output_df['From'].str.strip()
	output_df['To'] = output_df['To'].str.strip()
	output_df['Airline Code'] = output_df['Airline Code'].str.strip()

	# Create sets of combinations
	dynamic_markup_combinations = set(
		zip(markup_exceptions_df['POL'].str.lower(), markup_exceptions_df['POD'].str.lower(),
			markup_exceptions_df['Airline'].str.lower()))
	output_df_combinations = set(
		zip(output_df['From'].str.lower(), output_df['To'].str.lower(), output_df['Airline Code'].str.lower()))

	# Initialize an empty DataFrame to store log records
	log_markup_exception_records = pd.DataFrame(columns=output_df.columns.tolist() + ['Markup Exception Reason'])

	# Apply dynamic markups from Markup_exceptions sheet
	for combination in dynamic_markup_combinations:
		pol, pod, airline = combination
		columns = ['Min', 'Normal', '>=45', '>=100', '>=250', '>=300', '>=500', '>=1000', '>=2000']
		# Check if the combination exists in final_df_combinations
		output_df[columns] = output_df[columns].astype(float)
		exception_values = markup_exceptions_df[(markup_exceptions_df['POL'].str.lower() == pol) &
												(markup_exceptions_df['POD'].str.lower() == pod) &
												(markup_exceptions_df['Airline'].str.lower() == airline) &
												(markup_exceptions_df['is_active'] == 'Y')]
		exception_values = exception_values.drop('is_active', axis=1)

		markup_values = [min_markup_amount, normal_markup_amount, m_45_markup_amount, m_100_markup_amount,
						 m_250_markup_amount, m_300_markup_amount, m_500_markup_amount, m_1000_markup_amount,
						 m_2000_markup_amount]
		default_markup_amounts = pd.DataFrame(columns=columns)
		default_markup_amounts.loc[0] = markup_values
		if combination in output_df_combinations:
			try:
				#Update values in final_df based on some calculation for multiple columns
				updated_indices = ((output_df['From'].str.lower() == pol) &
								   (output_df['To'].str.lower() == pod) &
								   (output_df['Airline Code'].str.lower() == airline))

				non_zero_mask = (output_df.loc[updated_indices, columns] != 0)
				output_df.loc[updated_indices, columns] += np.where(non_zero_mask, exception_values[columns].values, 0)
				output_df.loc[updated_indices, columns] -= np.where(non_zero_mask,
																	default_markup_amounts[columns].values, 0)

				updated_records = output_df[updated_indices].copy()
				updated_records['Markup Exception Reason'] = 'Markup exception applied for combination {}'.format(
					combination)
				updated_records['Date'] = Created_At()
				log_markup_exception_records = pd.concat(
					[log_markup_exception_records.dropna(axis=1, how='all'), updated_records.dropna(axis=1, how='all')],
					ignore_index=True)

			except ValueError as ve:
				logging.error(f"An error occurred while applying dynamic markups for {combination}")

	return output_df, log_markup_exception_records

# ------------------- Driver Code ----------------------------------#
logging.info('------------- Starting execution -------------')

try:
	input_sheet_df = pd.read_csv(input_sheet_path)
except FileNotFoundError:
	logging.error("Error: Input sheet not found. Please check the path of the input sheet or ensure that it is not opened elsewhere.")
	# Exit or raise an exception to stop further execution
	sys.exit()

try:
	master_sheet_path = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'master_sheets_path', 'value'].values[0]
	master_sheet_df = pd.read_excel(master_sheet_path)
	input_search_params_df = pd.read_excel(master_sheet_path, sheet_name='Search_input_params', header=0)
	pod_list_df = pd.read_excel(master_sheet_path, sheet_name='POD_List', header=0)
	pod_airline_mapping_df = pd.read_excel(master_sheet_path, sheet_name='POD_Airline_Mapping', header=0)
	pol_airline_mapping_df = pd.read_excel(master_sheet_path, sheet_name='POL_Airline_Mapping', header=0)
	airlinemaster_df = pd.read_excel(master_sheet_path, sheet_name='AirlineMaster', header=0)
	airline_port_mapping_df = pd.read_excel(master_sheet_path, sheet_name='Airline_Port_Mapping', header=0)
	airlines_cleanup_df = pd.read_excel(master_sheet_path, sheet_name='Airlines_Cleanup', header=0)
	markup_exceptions_df = pd.read_excel(master_sheet_path, sheet_name='Markup_Exceptions', header=0)
	pol_list_df = pd.read_excel(master_sheet_path, sheet_name='POL_List', header=0)
	airline_exceptions_df = pd.read_excel(master_sheet_path, sheet_name='Airline_Exceptions', header=0)
except Exception as e:
	logging.error("Error while reading reference sheets form the Master_sheet, please recheck all the master sheets and try again")
	sys.exit()

#Read the input variables from the input sheet
try:
	pod_id_df = None
	username = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'express_rates_username', 'value'].values[0]
	password = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'express_rates_password', 'value'].values[0]
	login_url = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'express_rates_login_url', 'value'].values[0]
	tariff = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'tariff', 'value'].values[0]

	ams_rate = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'ams_rate', 'value'].values[0]
	ams_basis = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'ams_basis', 'value'].values[0]
	rate_separator = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'rate_separator', 'value'].values[0]
	final_airfreight_uploadsheet_landing_path = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'final_airfreight_uploadsheet_landing_path', 'value'].values[0]
	eliminated_records_file_path = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'eliminated_records_file_path', 'value'].values[0]
	log_markup_exception_records_path = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'applied_markup_exception_records_file_path', 'value'].values[0]
	raw_data_file_path = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'raw_data_file_path', 'value'].values[0]
	processed_port_pairs_log_file_path = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'processed_port_pairs_log_file_path', 'value'].values[0]
	change_history_file_path = input_sheet_df.loc[input_sheet_df['input_parameter'] == 'change_history_file_path', 'value'].values[0]

	min_markup_amount = float(input_sheet_df.loc[input_sheet_df['input_parameter'] == 'min_markup_amount', 'value'].values[0])
	normal_markup_amount = float(input_sheet_df.loc[input_sheet_df['input_parameter'] == 'normal_markup_amount', 'value'].values[0])
	m_45_markup_amount = float(input_sheet_df.loc[input_sheet_df['input_parameter'] == '45_markup_amount', 'value'].values[0])
	m_100_markup_amount = float(input_sheet_df.loc[input_sheet_df['input_parameter'] == '100_markup_amount', 'value'].values[0])
	m_250_markup_amount = float(input_sheet_df.loc[input_sheet_df['input_parameter'] == '250_markup_amount', 'value'].values[0])
	m_300_markup_amount = float(input_sheet_df.loc[input_sheet_df['input_parameter'] == '300_markup_amount', 'value'].values[0])
	m_500_markup_amount = float(input_sheet_df.loc[input_sheet_df['input_parameter'] == '500_markup_amount', 'value'].values[0])
	m_1000_markup_amount = float(input_sheet_df.loc[input_sheet_df['input_parameter'] == '1000_markup_amount', 'value'].values[0])
	m_2000_markup_amount = float(input_sheet_df.loc[input_sheet_df['input_parameter'] == '2000_markup_amount', 'value'].values[0])

except Exception as e:
	logging.error("Error while reading input parameters from the config input, please check the input config sheet and try again")
	sys.exit()

# DEFINE ALL THE NECESSARY XPATHS FOR SELECTING ELEMENTS ON THE SITE
username_xpath = "(//input[@name='username'])[2]"
password_xpath = "(//input[@name='password'])[2]"
login_button_xpath = "(//button[@class='btn btn-primary'][normalize-space()='Sign in'])[1]"
single_destination_xpath = "//a[@href='expressrates.php']"
tariff_xpath = "//select[@id='idTarrifType']"
origin_xpath = "//select[@id='idSourceCode']"
search_button_xpath = "//button[@id='btnSearchRates']"
logout_button_xpath = '//*[@id="navigation"]/div/div[2]/div/ul/li/a'
xpath_for_chzn_results = '//*[@id="idDestination_chzn"]/div/ul'

processed_port_pairs_log_df = pd.DataFrame()  #Initialize an empty df for storing all the processed port pairs
active_drivers = []


def setup_driver(chrome_driver_path):
    try:
        # service = webdriver.chrome.service.Service(ptions=chrome_driver_path)
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-dev-shm-usage')

        # driver = webdriver.Chrome(service=service, options=options)
        driver = webdriver.Chrome(options=options)
        return driver
        # return driver, service
    except Exception as e:
        logging.error(f"Error setting up driver")
        raise Exception("Driver setup failed")

def drop_extra_columns(final_df, new_order):
	# Get the columns that are not present in the new_order list
	extra_columns = [col for col in final_df.columns if col not in new_order]

	# Drop the extra columns from the final_df
	final_df = final_df.drop(columns=extra_columns)

	return final_df

new_order = ['From', 'To', 'Airline', 'Service', 'Min', 'Normal', '+45', '100', '+250', '+300', '+500', '+1000',
			 '+2000', 'FSC', 'WSC', 'Xray', 'MCC', 'CTG', 'Date', 'Created_At']

# --- Scraping code starts here ----
def process_batches(input_search_params_df,chrome_driver_path, batch_size=20, max_batch_attempts=3):
	"""Process input_search_params_df in batches of batch_size."""
	dfs = []  # List to store DataFrames for each batch
	combinations_processed = 0  # Counter for combinations processed today
	driver = setup_driver(chrome_driver_path)
	for i in range(0, len(input_search_params_df), batch_size):
		attempts = 0
		while attempts < max_batch_attempts:
			try:
				batch_df = input_search_params_df.iloc[i:i + batch_size]

				# Scraping Step 1: Login to the Express Rates page
				login(driver, login_url, username, password, username_xpath, password_xpath, login_button_xpath, max_attempts=3)

				# Scraping Step 2: Navigate to Single Destination page
				pod_id_df = navigate_and_create_POD_ID_DF(driver, single_destination_xpath, xpath_for_chzn_results)

				# Scraping Step 3: Perform the search using POD-POL port pairs and extract the data
				df = SEARCH_AND_EXTRACT_DATA(driver, batch_df, pod_id_df, tariff_xpath, origin_xpath, search_button_xpath, tariff)

				# Scraping step 4: Log out once data is extracted
				log_out(driver, logout_button_xpath)

				# Append the df for the current batch to the list
				dfs.append(df)

				# Increment the counter for combinations processed
				combinations_processed += len(batch_df)
				logging.info(f"Successfully processed batch no.  {i // batch_size + 1} on attempt {attempts}")

				break  # Exit the while loop if the batch processing is successful

			except Exception as e:
				attempts += 1
				logging.error(f"Error processing batch {i//batch_size + 1} on attempt {attempts}")
				if attempts == max_batch_attempts:
					logging.error(f"Max attempts reached for batch {i//batch_size + 1}. Skipping this batch and trying the next batch after 10 mins")
					time.sleep(200)
					break

		# Check if the daily limit of 200 combinations is reached
		if combinations_processed >= 201:
			logging.info("Daily limit of combinations reached. Pausing until next working day around 10 am.")
			final_df = pd.concat(dfs, ignore_index=True)
			# logging.info(final_df.columns)
			final_df = drop_extra_columns(final_df, new_order)
			final_df.to_sql('temp_cache', engine, if_exists='append', index=False)
			logging.info('Batch saved in the table temp_cache in incremental mode')
			dfs=[]
			# Calculate the time to pause until next working day around 10 am
			# Calculate the next day at 10:00 AM
			# next_day = datetime.now() + timedelta(days=1)
			# next_day = next_day.replace(hour=10, minute=0, second=0, microsecond=0)
			# # Calculate the time to sleep
			# sleep_time = (next_day - datetime.now()).total_seconds()

			# #Calculate the next day at the same time
			next_time = datetime.now() + timedelta(minutes=1)
			# Sleep until the specified time
			sleep_time = (next_time - datetime.now()).total_seconds()

			time.sleep(sleep_time)
			# Reset the combinations_processed counter
			combinations_processed = 0

	try:
		final_df = pd.concat(dfs, ignore_index=True)
		final_df = drop_extra_columns(final_df, new_order)
		final_df.to_sql('temp_cache', engine, if_exists='append', index=False)
		logging.info('Batch saved in the table temp_cache in the final mode')
		driver.close()
		driver.quit()
	except Exception as e:
		logging.error('No data received')
		sys.exit('No data received from the source, terminating the program.')

logging.info('Fetching old temp data')
df = fetch_raw_data('temp_cache')

if df is None or df.empty:
	#Call the function to process batches of 20 combinations
	logging.info('Old temp data not found, starting the scraping process')
	process_batches(input_search_params_df,chrome_driver_path)
else:
	logging.info("Old temp data found!!")
	logging.info("Processing the old data from temp_cache table")
	pass

# --- Scraping code ends here ----
df = fetch_raw_data('temp_cache')
#Reorder columns in the extracted data and save it as csv
new_order = ['From', 'To', 'Airline', 'Service', 'Min', 'Normal', '+45', '100', '+250', '+300', '+500', '+1000',
			 '+2000', 'FSC', 'WSC', 'Xray', 'MCC', 'CTG', 'Date', 'Created_At']
df = df.reindex(columns=new_order)  #Reorder columns in the extracted data
df = clean_dataframe(df)
df['Date'] = pd.to_datetime(df['Date'], format='%d %b %y', errors='coerce')
df['Date'] = df['Date'].dt.date

exceptions = set(airline_exceptions_df['Airline_To_Remove'])
df = df[~df['Airline'].isin(exceptions)]

#Create emtpy dataframe to store transformed data
#Define the column list
columns = ['From', 'To', 'Airline Code', 'Service', 'Min', 'Normal', '>=45', '>=100', '>=250', '>=300', '>=500',
		   '>=1000', '>=2000',
		   'FSC Rate', 'FSC Basis', 'WSC Rate', 'WSC Basis', 'XRay Rate', 'XRay Basis', 'MCC Rate', 'MCC Basis',
		   'CTG Rate', 'CTG Basis',
		   'AMS Rate', 'AMS Basis', 'Total C', 'Total G', 'Date', 'Created_At']

# Define the data types for each column
dtypes = {'From': str, 'To': str, 'Airline Code': str, 'Service': str,
		  'Min': float, 'Normal': float, '>=45': float, '>=100': float, '>=250': float, '>=300': float, '>=500': float,
		  '>=1000': float, '>=2000': float,
		  'FSC Rate': float, 'FSC Basis': str, 'WSC Rate': float, 'WSC Basis': str, 'XRay Rate': float,
		  'XRay Basis': str, 'MCC Rate': float, 'MCC Basis': str,
		  'CTG Rate': float, 'CTG Basis': str, 'AMS Rate': float, 'AMS Basis': str, 'Total C': float, 'Total G': float,
		  'Date': object, 'Created_At': object}

output_df = pd.DataFrame(columns=columns)  # Create an Empty DataFrame to load data from source dataframe
output_df = output_df.astype(dtypes)  # Assign the data types

eliminated_records = pd.DataFrame()  #Create an empty df to store eliminated records

# ----------- Start the data transformation from here step by step -----------------------------#
logging.info('Starting the data transformations on the extracted data')
#Step 1: Populate first 4 columns from scraped data into output_df
output_df = populate_from_to_airline_service(output_df, df)

#Step 2 (Airline): Replace old codes with new codes using Airline Port Mapping sheet and remove duplicates
output_df = replace_oldcode_with_newcode(output_df, airline_port_mapping_df)

#Step 3 (Airline): Validate airlines against Airlines_Cleanup sheet
output_df = validate_and_cleanup_airlines(output_df, airlines_cleanup_df, log_eliminated_records)

#Step 4 (Airline): Validate Airline names from AirlineMaster sheet
output_df = validate_airlines_against_airlinemaster(output_df, airlinemaster_df, log_eliminated_records)

#Step 5 (Airline): Validate POD-Airline Mapping
#output_df = validate_pod_airline_mapping(output_df, pod_airline_mapping_df, log_eliminated_records)

# Step 6: Validate POL-Airline Mapping
output_df = validate_pol_airline_mapping(output_df, pol_airline_mapping_df, log_eliminated_records)


#Step 7: Apply default markups to the extracted rates
output_df = apply_markups(output_df, df, min_markup_amount, normal_markup_amount, m_45_markup_amount,
						  m_100_markup_amount, m_250_markup_amount, m_300_markup_amount, m_500_markup_amount,
						  m_1000_markup_amount, m_2000_markup_amount)

#Step 8: Populate Rates and Basis columns
output_df = calculate_rates_basis(output_df, df, get_rate, get_basis, ams_rate, ams_basis)

# Step 9: #Calculate Total C and Total G columns
output_df = calculate_total_c_g(output_df)

# Step 10: #Replace null values by 0
output_df = fillna(output_df)

#Step 11: Custom function to format the values 0.00 -> 0
output_df = output_df.map(format_value)

#Step 12: #Remove duplicate airlines by checking lower rates in columns min, normal, ....2000
output_df = remove_duplicate_airlines(output_df, log_eliminated_records)

# Step 12: Apply dynamic markups
output_df, log_markup_exception_records = apply_dynamic_markups(markup_exceptions_df, output_df, min_markup_amount,
																normal_markup_amount, m_45_markup_amount,
																m_100_markup_amount, m_250_markup_amount,
																m_300_markup_amount, m_500_markup_amount,
																m_1000_markup_amount, m_2000_markup_amount)

output_df = output_df.drop_duplicates(subset=['From', 'To', 'Service', 'Airline Code'])

logging.info('Transforming the data is completed, saving the files')

# raw_df = df.copy()
df.fillna(value=0, inplace=True)
df = df.map(format_value)
df['unique_id'] = df[['From', 'To', 'Airline', 'Service', 'Min', 'Normal', '+45', '100', '+250', '+300', '+500', '+1000',
			 '+2000','Date']].apply(lambda x: '_'.join(x.astype(str)), axis=1)
df.columns = map(str.lower, df.columns)
#Check change history logs and update in the table
try:
	historical_df = fetch_raw_data('raw_data')
	historical_df.fillna(value=0, inplace=True)
	historical_df = historical_df.map(format_value)

	df_temp= df.copy()
	if historical_df is not None:
		change_df = compare_data(df_temp, historical_df)
		# Save change history to SQL database
		#change_df.drop('index', axis=1, inplace=True)
		change_df.to_sql('change_history', engine, if_exists='append', index=False)
		length = change_df.shape[0]
		logging.info("Change history log table updated with %s records", length)
	else:
		logging.warning("Historical data is empty.")
except PermissionError as pe:
	logging.error("CSV couldn't be updated because it's not accessible")
except Exception as e:
	logging.error("An error occurred while checking change log history: %s", str(e))
	logging.info('No change history data logged')

try:
	# if os.path.exists(raw_data_file_path):
	# 	# Append data to existing CSV file
	# 	df.to_csv(raw_data_file_path, mode='a', header=False, index=False)
	# else:
	# 	# Write DataFrame to new CSV file
	# 	df.to_csv(raw_data_file_path, index=False)
	for index, row in df.iterrows():
		try:
			row.to_frame().T.to_sql('raw_data', engine, if_exists='append', index=False)
		except IntegrityError as e:
			# Handle the IntegrityError (duplicate entry) here
			pass
		except Exception as e:
			logging.error("Error while storing the raw data into table: %s", str(e))
			pass
	logging.info('Raw Data table updated')
except Exception as e:
	logging.error('Error while storing the raw data into table : %s', str(e))

#Rename the final csv and save it as csv and upload to db
today_date = datetime.now().strftime("%d_%m_%Y_%H_%M")
filename = f'Airfreight_Uploadsheet_{today_date}.csv'
final_filename = final_airfreight_uploadsheet_landing_path + filename
output_df = output_df.map(format_value)
output_df_csv = output_df.drop(['Date', 'Created_At'], axis=1)
output_df_csv.to_csv(final_filename, index=False)
logging.info('Final Airfreight Uploadsheet stored at: %s', final_filename)
output_df['unique_id'] = output_df[['From', 'To', 'Airline Code', 'Service', 'Min', 'Normal', '>=45','>=100','>=250','>=300','>=500','>=1000','>=2000','Date']].apply(lambda x: '_'.join(x.astype(str)), axis=1)

output_df.columns = map(str.lower, output_df.columns)
output_df.columns = output_df.columns.str.replace(' ', '_')

try:
	for index, row in output_df.iterrows():
		try:
			row.to_frame().T.to_sql('airfreight_rates', engine, if_exists='append', index=False)
		except IntegrityError as e:
			# Handle the IntegrityError (duplicate entry) here
			pass
		except Exception as e:
			logging.error(f"Error while storing the Airfreight Rates data into table: %s", str(e))
			pass
	logging.info('Airfreight Rates table updated')
except Exception as e:
	logging.error('Error while storing the Airfreight Rates into table')
except Exception as e:
	logging.error("Error while storing the Airfreight Rates into table")
	#
	# output_df.to_sql('airfreight_rates', engine, if_exists='append', index=False)
	# logging.info('Airfreight Rates table updated')

log_markup_exception_records.columns = map(str.lower, log_markup_exception_records.columns)

#Save the applied markup exceptions log file and upload to db
# if os.path.exists(log_markup_exception_records_path):
#     # Append data to existing CSV file
#     log_markup_exception_records.to_csv(log_markup_exception_records_path,mode='a',header=False, index =False)
# else:
#     # Write DataFrame to new CSV file
#     log_markup_exception_records.to_csv(log_markup_exception_records_path, index =False)

log_markup_exception_records.columns = log_markup_exception_records.columns.str.replace(' ', '_')

#logging.info('Applied Markup Exceptions log file stored at: %s', log_markup_exception_records_path)
try:
	log_markup_exception_records.to_sql('applied_markup_exceptions', engine, if_exists='append', index=False)
	logging.info('Applied markup exceptions table updated')
except Exception as e:
	logging.error("Error while storing the applied markup exception records data into table")

#Save the eliminated records log file and upload to db
eliminated_records = eliminated_records.drop(
	columns=['Min', 'Normal', '>=45', '>=100', '>=250', '>=300', '>=500', '>=1000', '>=2000', 'FSC Rate', 'FSC Basis',
			 'WSC Rate', 'WSC Basis', 'XRay Rate', 'XRay Basis', 'MCC Rate', 'MCC Basis',
			 'CTG Rate', 'CTG Basis', 'AMS Rate', 'AMS Basis', 'Total C', 'Total G'], axis=1)

eliminated_records.columns = map(str.lower, eliminated_records.columns)

# if os.path.exists(eliminated_records_file_path):
#     # Append data to existing CSV file
#     eliminated_records.to_csv(eliminated_records_file_path,mode='a',header=False, index =False)
# else:
#     # Write DataFrame to new CSV file
#     eliminated_records.to_csv(eliminated_records_file_path, index =False)

eliminated_records.columns = eliminated_records.columns.str.replace(' ', '_')
try:
	eliminated_records.to_sql('eliminated_records', engine, if_exists='append', index=False)
	logging.info('Eliminated records table updated')
except Exception as e:
	logging.error("Error while storing the Eliminated records data into table")

processed_port_pairs_log_df.columns = map(str.lower, processed_port_pairs_log_df.columns)

#Save the processed port pairs file and upload to db
# if os.path.exists(processed_port_pairs_log_file_path):
#     # Append data to existing CSV file
#     processed_port_pairs_log_df.to_csv(processed_port_pairs_log_file_path,mode='a',header=False, index =False)
# else:
#     # Write DataFrame to new CSV file
#     processed_port_pairs_log_df.to_csv(processed_port_pairs_log_file_path, index =False)

#logging.info('Processed port pairs log files stored at: %s', processed_port_pairs_log_file_path)
try:
	processed_port_pairs_log_df.to_sql('processed_port_pairs', engine, if_exists='append', index=False)
	logging.info('Processed port pairs table updated')
except Exception as e:
	logging.error("Error while storing the processed port pairs data into table", e)

#Truncate temp table
truncate_table('temp_cache')

update_log_table(engine, log_file_path)
logging.info('Log table has been updated')
logging.info("--------- Execution Finished ---------")