def get_version():
    # Import things in this function to keep the package namespace clean
    from pathlib import Path

    package_directory = Path(__file__).parent
    with open(package_directory / "version.txt") as f:
        return f.read().strip()


__version__ = get_version()
del get_version
