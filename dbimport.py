import csv
import mysql.connector
from mysql.connector import Error

def load_flight_data(csv_file):
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host='localhost',
            database='flights_db',
            user='root',
            password='p@sskey123'
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # Open CSV file and load data
            with open(csv_file, 'r') as file:
                csv_data = csv.reader(file)
                next(csv_data)  # Skip header row
                
                for row in csv_data:
                    # Convert empty strings to None for decimal fields
                    processed_row = []
                    for i, value in enumerate(row):
                        if i >= 6 and value == '':  # Columns 6+ are numeric fields
                            processed_row.append(None)
                        else:
                            processed_row.append(value)
                    
                    # Prepare SQL query
                    query = """INSERT INTO flights (
                        year, month, carrier, carrier_name, airport, airport_name,
                        arr_flights, arr_del15, carrier_ct, weather_ct, nas_ct,
                        security_ct, late_aircraft_ct, arr_cancelled, arr_diverted,
                        arr_delay, carrier_delay, weather_delay, nas_delay,
                        security_delay, late_aircraft_delay
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                    
                    # Execute query
                    cursor.execute(query, processed_row)
            
            # Commit changes
            connection.commit()
            print("Data loaded successfully")
            
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Call the function with your CSV file path
load_flight_data('C:/Users/swaga/Downloads/Airline_Delay_Cause.csv/Airline_Delay_Cause.csv')
