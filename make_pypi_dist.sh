#!/bin/bash
# Package files for PyPI (pip installation)
# Delete contents of dist folder
if [ -d dist ] ; then
    echo "Removing the contents of dist/ dir."
    rm -r dist
fi

# Ensure setuptools, wheel, and twine are the latest versions
pip install --upgrade setuptools wheel
# Build the distro based on info in setup.py and MANIFEST.in
python -m build

echo "Barring errors above, package build is complete."
echo "Upload the .tar.gz file in dist/ as a new release on GitHub and run the following"
echo "command to update the package in PyPI (authorized users only, use __token__):"
echo "python -m twine upload dist/* --verbose"
