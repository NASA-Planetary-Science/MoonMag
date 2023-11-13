from setuptools import setup

setup(
    name='MoonMag',
    version='1.6.0',
    author='Marshall J. Styczinski',
    author_email='marshall.styczinski@bmsis.org',
    description='Magnetic induction calculations for sounding of icy moons',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/itsmoosh/MoonMag',
    project_urls={
        'Bug tracker': 'https://github.com/itsmoosh/MoonMag/issues',
        'Original publication': 'https://doi.org/10.1016/j.icarus.2021.114840'
    },
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent'
    ],
    packages=['MoonMag'],
    package_dir={'MoonMag': 'MoonMag'},
    install_requires=[
        'numpy >= 1.24.2',
        'scipy >= 1.10.1',
        'mpmath >= 1.2.1',
        'matplotlib >= 3.7.1',
        'spiceypy >= 5.1.2'
    ],
    include_package_data=True  # Files to include are listed in MANIFEST.in
)
