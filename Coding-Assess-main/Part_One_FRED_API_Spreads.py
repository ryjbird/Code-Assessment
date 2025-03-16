import requests as req
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# Map treasury series IDs to their respective maturities in years
def map_treasury_maturities( series_id ):
  string_array = series_id.split( 'DGS' )
  if( 'MO' in string_array[ 1 ]  ):
    string_array = string_array[ 1 ].split( 'MO' )
    return float( string_array[ 0 ] ) / 12
  else:
    return float( string_array[ 1 ] )

# Add interpolated yield curves
def interpolate_yield_curve( maturities, tsyDf, Tsy_Mats_Map ):
  
  # Create a dictionary to store the new curves
  new_yield_curves = dict( zip( list( map( lambda x: 'Interp {}y'.format( x ), maturities ) ), [ [] for i in range( len( maturities ) ) ] ) )  
  new_yield_curves[ 'date' ] = []

  # For each date, store the interpolated yield
  for row in tsyDf.iterrows(): 
    interp_yields = np.interp( maturities, list( map( lambda x: Tsy_Mats_Map[ x ], row[ 1 ].index ) ), list( map( lambda x: float( x ), row[ 1 ].values ) ) ) 
    for i in range( len( maturities ) ):
      new_yield_curves[ 'Interp {}y'.format( maturities[ i ] ) ].append( interp_yields[ i ] )
    new_yield_curves[ 'date' ].append( row[ 0 ] )

  # Create a Tds from the dictionary of curves
  interpolated_yield_curves = pd.DataFrame( new_yield_curves)
  interpolated_yield_curves.set_index( 'date', inplace = True )
  
  updated_tsyDf = tsyDf.join( interpolated_yield_curves )
  return updated_tsyDf

# Fetch Treasury yield data from FRED API for a given series ID and date range
def fetch_fred_yield(series_id, start_date="2023-01-01", end_date="2023-12-31", api_key="your_api_key"):
    # Define the API URL and parameters here
    endpoint = 'https://api.stlouisfed.org/fred/series/observations?series_id={}&observation_start={}&observation_end={}&api_key={}&file_type=json'.format( series_id, start_date, end_date, api_key )
    response = req.get( endpoint )
        
    # Make an API request and handle errors
    if response.status_code != 200:
      raise Exception( 'API Failed with {} error'.format( response.status_code ) )

    # Convert response JSON into a DataFrame, setting 'date' as index
    # Convert the 'value' column to numeric and rename it based on series_id
    data = response.json()
    data = list( map( lambda x: { 'date' : x[ 'date'], series_id : x[ 'value' ] }, data[ 'observations'] ) )
    df = pd.DataFrame( data )
    
    #Filter problematic data and clean data tyeps
    df = df[ df[ series_id ] != '.' ] 
    df[series_id] = df.apply( lambda x: float( x[ series_id ] ), axis = 1 )
    df.set_index( 'date', inplace = True )

    print( 'Fetched {}'.format( series_id ) )
    return df[[series_id]]  

# Main function to fetch yields, interpolate, and calculate spreads
def main():
    
  # List of Treasury yield series IDs on FRED
  tenor_series_ids = [
      "DGS1MO", "DGS3MO", "DGS6MO", "DGS1",  # Short-term yields
      "DGS2", "DGS3", "DGS5",               # Medium-term yields
      "DGS7", "DGS10", "DGS20", "DGS30"     # Long-term yields
  ] 
  # Map each tenor series ID to its respective maturity in years
  tenor_series_mats = dict( zip( tenor_series_ids, list( map( lambda x: map_treasury_maturities( x ), tenor_series_ids ) ) ) )

  # Initialize API key from 
  load_dotenv()
  Fred_API_Key = os.environ.get( 'FRED_API_KEY' )
  
  # Fetch data for each tenor, store in a dictionary of DataFrames
  yields = { x : fetch_fred_yield( x, api_key = Fred_API_Key ) for x in tenor_series_ids }

  # Combine all DataFrames into a single DataFrame, joining on the date index
  yields_df_arr = list( yields.values() )
  yields_df = yields_df_arr[ 0 ].join( yields_df_arr[ 1: ] )

  # Read Bonds Excel
  bonds_df = pd.read_excel( 'C:/Dev/Code-Assessment/Coding-Assess-main/data/Part 1. bonds_yields.xlsx' )
  
  # Interpolate the required missing yield curves
  walsToInterp = list( map( lambda x: float( x ), bonds_df[ 'WAL (years)' ].unique().tolist() ) )
  interped_tsy_yields_df = interpolate_yield_curve( walsToInterp, yields_df, tenor_series_mats )
  
  # Use the interpolated yield curve to calculate the spread for each bond over time
  bond_dict = dict()
  sector_dict = dict()
  for bond in bonds_df.iterrows():
    bond_ID = bond[ 1 ][ 'Bond ID' ]
    bond_yield = bond[ 1 ][ 'Yield (%)' ]
    bond_sector = bond[ 1 ][ 'Sector' ]
    bond_WAL = bond[ 1 ][ 'WAL (years)' ]
    spread = bond_yield - interped_tsy_yields_df[ 'Interp {}y'.format( bond[ 1 ][ 'WAL (years)' ] ) ]

    bond_dict[ bond_ID ]= spread
    
    #Create spread curve by sector
    if bond_sector not in sector_dict:
      sector_dict[ bond_sector ] = [ ( float( bond_WAL ), float( spread.values[ -1 ] ) ) ]
    else:
      sector_dict[ bond_sector ].append( ( float( bond_WAL ), float( spread.values[ -1 ] ) ) )
  
    
    # Map Bond to sector
    bond_to_sector = {bond[1]['Bond ID']: bond[1]['Sector'] for bond in bonds_df.iterrows()}

    # Define a color map for sectors
    sector_colors = {
        sector: color for sector, color in zip(sector_dict.keys(), plt.cm.tab20.colors)
    }

  spread_df = pd.DataFrame(bond_dict)  
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
  
  for bond_id in spread_df.columns:
      ax1.plot(spread_df.index, spread_df[bond_id], label=bond_id, color=sector_colors [ bond_to_sector[ bond_id ] ] )

  # Add labels and legend
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Spread')
  ax1.set_title('Bond Spreads Over Time')
  ax1.legend()
  ax1.set_xticks(ticks=spread_df.index[::50])
  
  # Generate spread curves for each sector
  for sector, data in sector_dict.items():
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    ax2.scatter(x, y, label=sector)
    
    coefficients = np.polyfit(x, y, 2)
    polynomial = np.poly1d(coefficients)
    
    # Generate x values for the fitted curve
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = polynomial(x_fit)
    
    # Plot the fitted curve
    ax2.plot(x_fit, y_fit, label=f'{sector} fit')

  # Add labels and legend
  ax2.set_xlabel('WAL')
  ax2.set_ylabel('Spread')
  ax2.set_title('Sector Spread Curves as of EOY 2023')
  ax2.legend()

  # Show the plot
  plt.tight_layout()
  plt.show()  
  return

if __name__ == "__main__":
    main()