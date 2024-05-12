import os
import subprocess
    
def clone_repo(project, branch, user, token=None, move_to_root=True, change_dir=None):
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
    os.system(f"git clone -b {branch} {github_url}")
    
    if move_to_root:
        os.system(f"mv {project}/* .")
    
    if change_dir:
        os.chdir(project)

def install_dependencies(requirements_file="requirements.txt", script_file="requirements.sh"):
    """
    Install the requirements from the requirements.txt file.
    """
    assert os.path.exists(requirements_file) or os.path.exists(script_file), "Both requirements.txt and requirements.sh files are not found"
    assert requirements_file is not None or script_file is not None, "Both requirements.txt and requirements.sh files are not found"
    
    if requirements_file is not None:
        os.system(f"pip install -r {requirements_file}")
    
    if script_file is not None:
        os.chmod(script_file, 0o777)
        os.system(f"./{script_file}")
        
def execute_command(executable, *args):
    """
    Execute a command with given executable and arguments.
    """
    # Construct the command
    command = [executable] + list(args)

    # Execute the command using subprocess
    subprocess.run(command)