def extract_zip_file(zip_file_path, extract_path):
    """
    Extract a ZIP file to a specified directory.

    Parameters:
    - zip_file_path (str): Path to the ZIP file.
    - extract_path (str): Directory where the ZIP file will be extracted.

    Returns:
    None
    """
    zip_ref = zipfile.ZipFile(zip_file_path, 'r')
    zip_ref.extractall(extract_path)
    zip_ref.close()