__all__ = ["AggregateGDS"]

# Standard library imports

# Related third party imports
from astropy.table import Table
import numpy as np

# Local application/library specific imports

def safe_round(v, ndigits=8):
    if isinstance(v, (float, np.floating)):
        return round(v, ndigits)
    return v


def is_missing_value(v) -> bool:
    """Return True for placeholder/missing values."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip() in {"", "--", "-"}:
        return True
    try:
        return np.isnan(v)
    except Exception:
        return False


def first_real_value(values):
    """Return the first non-placeholder value, preserving strings if needed."""
    for v in values:
        if not is_missing_value(v):
            return v
    return "--"


class AggregateGDS:
    """Aggregate GDS data by computing statistics over numerical and flag keywords."""

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

    def compute_variability(self, values: np.ndarray, tol: float = 1e-12) -> str:
        """Determine if values are constant or changing."""
        if len(values) == 0:
            return "UNKNOWN"
        if np.all(np.abs(values - values[0]) < tol):
            return "CONSTANT"
        return "TIME-VARYING"

    def compute_flag_variability(self, values) -> str:
        """Determine if flag values are constant or varying."""
        if len(values) == 0:
            return "UNKNOWN"

        unique_vals = {str(v) for v in values if not is_missing_value(v)}

        if len(unique_vals) <= 1:
            return "CONSTANT"
        return "TIME-VARYING"

    def compute_aggregates(self, sband_mnemonics) -> Table:
        """Compute aggregates (first, last, min, max, mean, median, std)
        for numerical keywords.

        Returns
        -------
        `astropy.table.Table`
            Table containing aggregated values for each keyword.
        """
        # Convert all Mnemonic values to lowercase
        sband_mnemonics['Mnemonic'] = [mnemonic.lower() for mnemonic in sband_mnemonics['Mnemonic']]

        gds_nums = Table(
            names=[
                'Keyword', 'Description', 'First', 'Last',
                'Min', 'Max', 'Mean', 'Median', 'Std',
                'Variability', 'Type'
            ],
            dtype=[
                'str', 'str', 'str', 'str',
                'float64', 'float64', 'float64', 'float64', 'float64',
                'str', 'str'
            ]
        )

        for keyword in self.numerical_keywords:
            state_row = sband_mnemonics[sband_mnemonics['Mnemonic'] == keyword]
            if len(state_row) > 0:
                value_type = state_row['Type'][0]
                description = state_row['Description'][0]
            else:
                value_type = 'N/A'
                description = 'N/A'

            if keyword not in self.data_table['Mnemonic']:
                gds_nums.add_row([
                    keyword, description,
                    '--', '--',
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    'UNKNOWN',
                    value_type
                ])
                continue

            rows_for_keyword = self.data_table[self.data_table['Mnemonic'] == keyword]
            raw_values = rows_for_keyword['Value2']

            numeric_values = []
            for v in raw_values:
                try:
                    if not is_missing_value(v):
                        numeric_values.append(float(v))
                except Exception:
                    pass

            first_raw = first_real_value(raw_values)
            last_raw = first_real_value(raw_values[::-1])

            first_value = "--" if is_missing_value(first_raw) else str(first_raw)
            last_value = "--" if is_missing_value(last_raw) else str(last_raw)

            if len(numeric_values) > 0:
                numeric_values = np.asarray(numeric_values, dtype=float)
                min_value = self.compute_min(numeric_values)
                max_value = self.compute_max(numeric_values)
                mean_value = self.compute_mean(numeric_values)
                median_value = self.compute_median(numeric_values)
                std_value = np.std(numeric_values)
                variability = self.compute_variability(numeric_values)
            else:
                min_value = np.nan
                max_value = np.nan
                mean_value = np.nan
                median_value = np.nan
                std_value = np.nan
                variability = "NON_NUMERIC"

            gds_nums.add_row([
                keyword, description,
                first_value, last_value,
                safe_round(min_value),
                safe_round(max_value),
                safe_round(mean_value),
                safe_round(median_value),
                safe_round(std_value),
                variability,
                value_type
            ])

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
            names=['Keyword', 'Description', 'First', 'Last', 'Mid', 'FlagOptions',
                   'Type', 'Bitmap', 'Variability'],
            dtype=['str', 'str', 'str', 'str', 'str', 'object', 'str', 'int', 'str']
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
                gds_flags.add_row([keyword, description, 'N/A', 'N/A', 'N/A',
                                   flag_options, value, 0, 'UNKNOWN'])
            else:
                rows_for_keyword = self.data_table[self.data_table['Mnemonic'] == keyword]
                values = rows_for_keyword['Value2']  # Flags are stored as strings
                # compute variability
                variability = self.compute_flag_variability(values)
                # compute aggregate values
                first_value = self.compute_first(values)
                last_value = self.compute_last(values)
                mid_value = values[len(values) // 2] if len(values) > 0 else 'N/A'

                # Compute the bitmap for encountered flags
                bitmap = compute_flag_bitmap(flag_options, list(set(values)))

                # add flags to row
                gds_flags.add_row([
                    keyword, description, first_value, last_value,
                    mid_value, flag_options, value, bitmap, variability
                ])

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
