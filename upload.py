import fileinput
import sys
import subprocess
# Ask the user for the new version
new_version = input("Enter the new version: ")

# Open the file in read+write mode
with fileinput.FileInput("setup.py", inplace=True) as file:
    for line in file:
        if line.strip().startswith('version='):
            sys.stdout.write(line.replace(line, f"    version='{new_version}',\n"))
        else:
            sys.stdout.write(line)

# Execute the command to build the package
subprocess.run(['chmod', '777', './build.sh'])
subprocess.run(["./build.sh"], shell=True)

# Uninstall the old version and install the new one
subprocess.run(['pip', 'uninstall', '-y', 'torch-project-manager'])
subprocess.run(['pip', 'install', '.'])