from pathlib import Path
from openhexa.sdk import current_run, pipeline, workspace

@pipeline("snt_setup", name="0. SNT Setup", timeout=600)
def snt_folders_setup() -> None:
    """
    A simple and robust pipeline to initialize the core folder 
    structure in an OpenHEXA workspace.
    """
    current_run.log_info("Starting workspace folder setup...")
    
    # workspace.files_path points to the root of the workspace filesystem
    root_path = Path(workspace.files_path)
    
    # The strictly requested folders
    folders_to_create = [
        "configuration",
        "code",
        "data",
        "pipelines",
        "results",
    ]
    
    for folder in folders_to_create:
        folder_path = root_path / folder
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            current_run.log_info(f"Successfully created or verified existence of directory: '{folder_path}'.")
        except Exception as e:
            current_run.log_error(f"Failed to create directory {folder}: {e}")
            raise Exception(f"Pipeline failed while creating {folder}") from e

    current_run.log_info("Workspace folder setup completed successfully!")

if __name__ == "__main__":
    snt_folders_setup()