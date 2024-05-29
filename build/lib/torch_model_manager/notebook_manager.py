import os
import subprocess
    
def clone_repo(project, user, branch = None, token=None, move_to_root=True, delete_original = True, change_dir=None):
    """
    Clone a repository from a given URL to a given directory.
    """
    assert (move_to_root and not change_dir) or (not move_to_root and change_dir), "Only one of move_to_root and change_dir can be True"

    # Clone the repository
    if token is not None:
        github_url = f"https://{token}@github.com/{user}/{project}.git"
    else:
        github_url = f"https://github.com/{user}/{project}.git"
    
    # Clone the repository using variables
    if branch is not None:
        os.system(f"git clone -b {branch} {github_url}")
    else:
        os.system(f"git clone {github_url}")
        
    
    if move_to_root:
        os.system(f"mv {project}/* .")
        if delete_original:
            os.system(f"rm -rf {project}")
    

    
    if change_dir:
        os.chdir(project)

def install_dependencies(requirements_file="requirements.txt", script_file="requirements.sh", compiler="bash"):
    """
    Install the requirements from the requirements.txt file.
    """
    
    assert requirements_file is not None or script_file is not None, "Both requirements.txt and requirements.sh files are not found"
    
    if requirements_file is not None:
        os.system(f"pip install -r {requirements_file}")
    
    if script_file is not None:
        os.system(f"{compiler} {script_file}")
        
def execute_command(*args):
    """
    Execute a command with given executable and arguments.
    """
    # Construct the command
    command = " ".join(list(args))
    # Execute the command using subprocess
    os.system(command)