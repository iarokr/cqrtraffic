# import dependencies
import pandas as pd
import time
import requests
import datetime
import warnings
import pathlib
import os

from .constant import DEF_COL_NAMES_FINTRAFFIC, URL_FINTRAFFIC


# Download the .csv report of `tms_id` station for the day `day_of_year`
def read_report(
    tms_id: int,
    year: int,
    day: int,
    direction: int,
    hour_from: int = 6,
    hour_to: int = 20,
    delete_if_faulty: bool = True,
    save_name: str = None,
) -> pd.DataFrame:
    """
    Download the raw data from Fintraffic for the `tms_id` station for the `day` of the `year`. \
    By default the data is cleaned - faulty observations are deleted. \
    Also, the type of vehicle is determined based on the classification from Fintraffic.

    | It is possible to save a `.gzip` file, specifying a `save_name`. 
    
    Parameters
    ----------
    tms_id : int
        The identity number of a traffic measurement station (TMS). \
        Meta-data about TMS is available `here <https://www.digitraffic.fi/en/road-traffic/#current-data-from-tms-stations>`_.
    year : int
        The year data was collected in the 4-digit format. Data is available starting from 1995 only. 
    day : int
        The day of the year the data was collected. \
        The day is provide as an integer in range(1, 366), with 1 - January 1st, \
        365 (366 in leap year) - December 31st. \
        To caclulate the day of the year from a date you can use `cqrtraffic.utils.funcs.date_to_day()` function.
    direction : int
        Flow direction of interest. Marked as 1 or 2. Check the TMS meta-data to get the necessary direction. 
    hour_from : int, optional
        Hour from which the data is loaded, by default 6
    hour_to : int, optional
        Hour until which the data is loaded (this our is not inclued), by default 20
    delete_if_faulty : bool, optional
        If `True`, observations with the `1` value for the column `if_faulty` will be deleted, by default True
    save_name : str, optional
        If specified, saves loaded data in the .gzip format, by default None

    Returns
    -------
    pd.DataFrame
        A `pd.DataFrame` with the raw data.

    See Also
    --------
    :func:`~cqrtraffic.utils.funcs.date_to_day`
        A function to convert a date into day of the year. 
    """  # noqa E501
    # Initiate timer
    start_time = time.perf_counter()

    # Initiation message
    print(f"Trying to load data for the day {day} of year {year} from the server...")

    # Check assert errors
    # fmt: off
    assert 0 <= hour_from < 24, "Error: the hour_from is incorrect, it should be 0 <= hour_from < 24"  # noqa
    assert 0 < hour_to <= 24, "Error: the hour_to is incorrect, it should be 0 < hour_to <= 24"  # noqa
    assert hour_from < hour_to, "Error: the hour_to should be less than hour_to"  # noqa 
    assert 1 <= day <= 366, "Error: the day is incorrect, it should be 1 < day < 366" # noqa 
    assert 1995 <= year, "Error: the data is available starting from year 1995 only " # noqa
    assert direction == 1 or direction == 2, "Error: direction must be either 1 or 2, check TMS station description" # noqa
    # fmt: on

    # Create column names for the pd.DataFrame
    column_names = DEF_COL_NAMES_FINTRAFFIC

    # Initiate an empty the pd.DataFrame
    df = pd.DataFrame()

    # Assign URL path for data loading
    url = URL_FINTRAFFIC

    # Create the actual url
    url = (
        url.replace("TMS", str(tms_id))
        .replace("YY", str(year)[2:4])
        .replace("DD", str(day))
    )

    # Try to download the file
    if requests.get(url).status_code != 404:

        # Download the file from the server
        df = pd.read_csv(url, delimiter=";", names=column_names)

        # Assign dates
        df["date"] = datetime.date(year, 1, 1) + datetime.timedelta(day - 1)

        # Calculate the number of different vehicles
        df["cars"] = df["vehicle"].apply(lambda x: 1 if x == 1 else 0)
        df["buses"] = df["vehicle"].apply(lambda x: 1 if x == 3 else 0)
        df["trucks"] = df["vehicle"].apply(
            lambda x: 1 if x == 2 or x == 4 or x == 5 or x == 6 or x == 7 else 0
        )

        # Delete faulty data point
        if delete_if_faulty is True:
            df = df[df.faulty != 1]

        # Select data only from the specified timeframe
        df = df[df.total_time >= hour_from * 60 * 60 * 100]
        df = df[df.total_time <= hour_to * 60 * 60 * 100]

        # Filter the correct direction
        df = df[df.direction == direction]

        # Stop timer
        end_time = time.perf_counter()
        print(
            f"Download successful - file for the sensor {tms_id} for the day {day} in year {year} was loaded in {end_time-start_time:0.4f} seconds"  # noqa E501
        )

        # Save to .gzip if necessary
        if save_name is not None:
            assert save_name.lower().endswith(
                ".gzip"
            ), "Error: Filename is wrong, please, provide it in *.gzip format"
            df.to_parquet(save_name, engine="pyarrow", compression="gzip")
            print(f"Data is successfully saved to {save_name}")
    else:
        # Stop timer
        end_time = time.perf_counter()
        print(f"Time spent: {end_time-start_time:0.4f} seconds")
        message = (
            "Warning: The data for the TMS "
            + str(tms_id)
            + " for the day "
            + str(day)
            + " of year "
            + str(year)
            + " does not exist. "
            + "Try to select another day. "
            + "An empty pd.DataFrame will be returned."
        )
        warnings.warn(message=message)
        if save_name is not None:
            warnings.warn(
                "Warning: It is impossible to save a file, as data was not loaded. Check warning above."  # noqa E501
            )
    return df


# Donwload .csv reports of `tms_id` station for the `days_list` from Fintraffic
def read_several_reports(
    tms_id: int,
    year_day_list: list,
    direction: int,
    hour_from: int = 6,
    hour_to: int = 20,
    delete_if_faulty: bool = True,
    save_name: str = None,
) -> pd.DataFrame:
    """
    Download the raw data from Fintraffic for the `tms_id` station for the `days_list` of the `year`. \
    Data for each day is loaded separately using :func:`~cqrttraffic.utils.load.read_report` and then appended together.
    By default the data is cleaned - faulty observations are deleted. \
    Also, the type of vehicle is determined based on the classification from Fintraffic.

    | It is possible to save a `.gzip` file, specifying a `save_name`. 

    Parameters
    ----------
    tms_id : int
        The identity number of a traffic measurement station (TMS). \
        Meta-data about TMS is available `here <https://www.digitraffic.fi/en/road-traffic/#current-data-from-tms-stations>`_.
    year_day_list : list
        A list of days in the format `[[year1, day1], [year2, day2],...]`
    direction : int
        low direction of interest. Marked as 1 or 2. Check the TMS meta-data to get the necessary direction. 
    hour_from : int, optional
        Hour from which the data is loaded, by default 6
    hour_to : int, optional
        Hour until which the data is loaded (this our is not inclued), by default 20
    delete_if_faulty : bool, optional
        If `True`, observations with the `1` value for the column `if_faulty` will be deleted, by default True
    save_name : str, optional
        If specified, saves loaded data in the .gzip format, by default None

    Returns
    -------
    pd.DataFrame
        A `pd.DataFrame` with loaded data

    See Also
    --------
    :func:`~cqrttraffic.utils.load.read_report`
    """  # noqa E501
    # Initiate timer
    start_time = time.perf_counter()

    # Initiate counter
    counter = 0

    # Initiate an empty the pd.DataFrame
    df = pd.DataFrame()

    # Check assert errors
    # fmt: off
    assert 0 <= hour_from < 24, "Error: the hour_from is incorrect, it should be 0 <= hour_from < 24"  # noqa
    assert 0 < hour_to <= 24, "Error: the hour_to is incorrect, it should be 0 < hour_to <= 24"  # noqa
    assert hour_from < hour_to, "Error: the hour_to should be less than hour_to"  # noqa 
    assert direction == 1 or direction == 2, "Error: direction must be either 1 or 2, check TMS station description" # noqa
    # fmt: on

    # Interate through each day
    for count, value in enumerate(year_day_list):
        # Initiate loaded pd.DataFrame
        read_df = pd.DataFrame()

        if df.empty:
            read_df = read_report(
                tms_id,
                value[0],
                value[1],
                direction,
                hour_from,
                hour_to,
                delete_if_faulty,
                save_name=None,
            )
            if read_df.empty is False:
                df = read_df
                counter += 1
        else:
            read_df = read_report(
                tms_id,
                value[0],
                value[1],
                direction,
                hour_from,
                hour_to,
                delete_if_faulty,
                save_name=None,
            )
            if read_df.empty is False:
                df = pd.concat((df, read_df), ignore_index=True)
                counter += 1
    # Check that some data was loaded
    assert df.empty is False, "Error: Data was not loaded, check the input given."

    # Stop timer
    end_time = time.perf_counter()

    # Confirm the result
    print(
        f"Loading sucessful: {counter} out of {len(year_day_list)} files loaded in {end_time-start_time:0.4f} seconds"  # noqa E521
    )

    # Save to .gzip if necessary
    if save_name is not None:
        assert save_name.lower().endswith(
            ".gzip"
        ), "Error: Filename is wrong, please, provide it in *.gzip format"
        df.to_parquet(save_name, engine="pyarrow", compression="gzip")
        print(f"Data is successfully saved to {save_name}")

    return df


def read_gzip_ft(filepath: str) -> pd.DataFrame:
    """
    Reads a .gzip file, containing raw data from Fintraffic, which was previously loaded and saved using the functionality of this package.

    Parameters
    ----------
    filepath : str
        A string, which contains a path to the .gzip file.

    Returns
    -------
    pd.DataFrame
        A `pd.DataFrame` with loaded data

    Raises
    ------
    Exception
        Incorrect path provided or the file is empty.
    """  # noqa E501
    # Initialize timer
    start_time = time.perf_counter()

    # Initiation message
    print(f"Trying to load data locally from {filepath} ...")

    # Initialize pd.DataFrame
    df = pd.DataFrame()

    # Check assert errors
    # fmt: off
    assert filepath.lower().endswith('.gzip'), "Error: Wrong file format or a file is not found. Please, provide a valid .gzip file." # noqa E521
    # fmt: on

    # Make a unified location identifyer
    filepath = pathlib.Path(filepath)

    # Load a file locally
    if (os.path.exists(filepath) is True) and (os.path.getsize(filepath) != 0):
        df = pd.read_parquet(filepath)
    else:
        raise Exception(
            "File is empty or it does not exist at the given filepath. Please, check the file or the filepath."  # noqa E521
        )

    # Stop the timer
    end_time = time.perf_counter()

    # Confirm the result
    print(
        f"Loading completed: the file was loaded in {end_time-start_time:0.4f} seconds"  # noqa E521
    )

    return df
