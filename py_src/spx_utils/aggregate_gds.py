__all__ = ["AggregateGDS"]

# Standard library imports

# Related third party imports
from astropy.table import Table
import numpy as np

# Local application/library specific imports


class AggregateGDS:
    def __init__(self, data_table: Table, numerical_keywords: list, flag_keywords: list):
        """Initialize the AggregateGDS class with a data table,
           numerical and flag keywords.

        Parameters
        ----------
        data_table : `astropy.table.Table`
            Table containing GDS data.
        numerical_keywords : `list`
            List of keywords to process.
        flag_keywords : `list`
            List of flag keywords to process.
        """
        self.data_table = data_table
        self.numerical_keywords = numerical_keywords
        self.flag_keywords = flag_keywords

    def compute_first(self, values: np.ndarray) -> float:
        """Get the first value from the sband data.

        Parameters
        ----------
        values : `np.ndarray`
            Array of values.

        Returns
        -------
        `float`
            The first value.
        """
        return values[0]

    def compute_last(self, values: np.ndarray) -> float:
        """Get the last value from the sband data.

        Parameters
        ----------
        values : `np.ndarray`
            Array of values.

        Returns
        -------
        `float`
            The last value.
        """
        return values[-1]

    def compute_min(self, values: np.ndarray) -> float:
        """Get the minimum value from the sband data.

        Parameters
        ----------
        values : `np.ndarray`
            Array of values.

        Returns
        -------
        `float`
            The minimum value.
        """
        return values.min()

    def compute_max(self, values: np.ndarray) -> float:
        """Get the maximum value from the sband data.

        Parameters
        ----------
        values : `np.ndarray`
            Array of values.

        Returns
        -------
        `float`
            The maximum value.
        """
        return values.max()

    def compute_mean(self, values: np.ndarray) -> float:
        """Get the mean value from the sband data.

        Parameters
        ----------
        values : `np.ndarray`
            Array of values.

        Returns
        -------
        `float`
            The mean value.
        """
        return values.mean()

    def compute_median(self, values: np.ndarray) -> float:
        """Get the median value from the sband data.

        Parameters
        ----------
        values : `np.ndarray`
            Array of values.

        Returns
        -------
        `float`
            The median value.
        """
        return np.median(values)

    def compute_aggregates(self, sband_mnemonics) -> Table:
        """ Compute aggregates (first, last, min, max, mean, median)
            for numerical keywords.

        Returns
        -------
        `astropy.table.Table`
            Table containing aggregated values for each keyword.
        """
        # Convert all Mnemonic values to lowercase
        sband_mnemonics['Mnemonic'] = [mnemonic.lower() for mnemonic in sband_mnemonics['Mnemonic']]

        gds_nums = Table(
            names=['Keyword', 'Description', 'First', 'Last', 'Min', 'Max', 'Mean', 'Median', 'Type'],
            dtype=['str', 'str', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'str']
        )

        for keyword in self.numerical_keywords:
            # Extract `Type`` options from `sband_mnemonics`
            state_row = sband_mnemonics[sband_mnemonics['Mnemonic'] == keyword]
            if len(state_row) > 0:
                value = state_row['Type'][0]
                description = state_row['Description'][0]
            else:
                value = 'N/A'
                description = 'N/A'

            if keyword not in self.data_table['Mnemonic']:
                # Keyword not found, add placeholder values
                gds_nums.add_row([keyword, description, -1, -1, -1, -1, -1, -1, value])
            else:
                # Filter rows for the current keyword
                rows_for_keyword = self.data_table[self.data_table['Mnemonic'] == keyword]
                values = rows_for_keyword['Value2'].astype(float)

                # Compute statistics
                first_value = self.compute_first(values)
                last_value = self.compute_last(values)
                min_value = self.compute_min(values)
                max_value = self.compute_max(values)
                mean_value = self.compute_mean(values)
                median_value = self.compute_median(values)

                # Add the row to the aggregated table
                gds_nums.add_row([keyword, description, first_value, last_value,
                                  min_value, max_value, mean_value, median_value, value])

        return gds_nums

    def compute_flag_aggregates(self, sband_mnemonics: Table) -> Table:
        """Compute aggregates for flag keywords.

        Parameters
        ----------
        sband_mnemonics : `astropy.table.Table`
            Table containing `Value/State` column with flag options.

        Returns
        -------
        `astropy.table.Table`
            Table containing aggregated flag data for each keyword.
        """
        # Convert all Mnemonic values to lowercase
        sband_mnemonics['Mnemonic'] = [mnemonic.lower() for mnemonic in sband_mnemonics['Mnemonic']]

        gds_flags = Table(
            names=['Keyword', 'Description', 'First', 'Last', 'Mid', 'FlagOptions', 'Type', 'Bitmap'],
            dtype=['str', 'str', 'str', 'str', 'str', 'object', 'str', 'int']
        )

        for keyword in self.flag_keywords:
            # Extract `Value/State` options from `sband_mnemonics`
            state_row = sband_mnemonics[sband_mnemonics['Mnemonic'] == keyword]
            if len(state_row) > 0:
                flag_options = [option.lower() for option in state_row['Value/State'][0].split()]
                value = state_row['Type'][0]
                description = state_row['Description'][0]
            else:
                flag_options = []
                value = 'N/A'
                description = 'N/A'

            if keyword not in self.data_table['Mnemonic']:
                gds_flags.add_row([keyword, description, 'N/A', 'N/A', 'N/A', flag_options, value, 0])
            else:
                rows_for_keyword = self.data_table[self.data_table['Mnemonic'] == keyword]
                values = rows_for_keyword['Value2']  # Flags are stored as strings

                first_value = self.compute_first(values)
                last_value = self.compute_last(values)
                mid_value = values[len(values) // 2] if len(values) > 0 else 'N/A'

                # Compute the bitmap for encountered flags
                bitmap = compute_flag_bitmap(flag_options, list(set(values)))

                gds_flags.add_row([keyword, description, first_value, last_value,
                                   mid_value, flag_options, value, bitmap])

        return gds_flags


def compute_flag_bitmap(flag_options, encountered_flags) -> int:
    """Compute a bitmap for the encountered flags based on the list of flag options.

    Parameters
    ----------
    flag_options : `list[str]`
        List of all possible flags in the format '<bit>/<name>', e.g., '0/Wait'.
    encountered_flags : list of str
        List of flags that have been encountered, e.g., ['Wait', 'Point'].

    Returns
    -------
    `int`
        An integer representing the bitmap of encountered flags.
    """
    # Create a mapping from flag names to their bit positions
    flag_map = {option.split('/')[1]: int(option.split('/')[0]) for option in flag_options}

    # Initialize the bitmap to zero
    bitmap = 0

    # Iterate through encountered flags and set the corresponding bit
    for flag in encountered_flags:
        if flag in flag_map:
            bit_position = flag_map[flag]
            bitmap |= (1 << bit_position)  # Set the bit at the bit_position

    return bitmap
