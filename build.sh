rm -r dist
rm -r torch_model_manager.egg-info
python3 -m build
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*