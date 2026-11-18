import subprocess


class EmptyLVFQueryResultError(Exception):
    """Exception raised when no targets are found in any LVF image.

    Parameters
    ----------
    n_src : `int`
        The number of targets searched.
    instrument : `str`
        The instrument name.
    dataset_type : `str`
        The dataset type name.

    """

    def __init__(self, n_src: int, instrument: str, dataset_type: str) -> None:
        self.n_src = n_src
        self.instrument = instrument
        self.dataset_type = dataset_type

    def __reduce__(self):
        # This method tells pickle how to serialize/deserialize the exception
        return (
            self.__class__,
            (self.n_src, self.instrument, self.dataset_type)
        )

    def __str__(self) -> str:
        return (f"No LVF images contain the required {self.n_src} input sources.\n"
                f"Instrument: {self.instrument}, Dataset Type: {self.dataset_type}.")


class PipetaskError(Exception):
    """Exception raised on pipetask run failure.

    Parameters
    ----------
    short_descr : `str`
        Short description of the failure.
    err : `subprocess.CalledProcessError`

    """

    def __init__(self, short_desc: str, err: subprocess.CalledProcessError) -> None:
        self.short_desc = short_desc
        self.err = err

    def __str__(self) -> str:
        return f"{self.short_desc}. " + str(self.err)


class VOTableSearchError(Exception):
    """Exception raised when unable to find opening or closing `<VOTAB>`
    element in XML content.

    Parameters
    ----------
    substring : `str`
        The search substring.
    start_idx : `int`
        The index of the character where the search began.

    """

    def __init__(self, substring: str, start_idx: int) -> None:
        self.substring = substring
        self.start_idx = start_idx

    def __str__(self) -> str:
        return (f"Substring '{self.substring}' not found after index {self.start_idx} "
                f"in VOTable file content.")
