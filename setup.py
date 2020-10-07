from setuptools import setup

setup(
    name='DimLiftBackend',
    version='1.0',
    long_description=__doc__,
    packages=['resources'],
    include_package_data=True,
    zip_safe=False,
    install_requires=['Flask', 'jsonpickle', 'numpy', 'pandas', 'scipy', 'sklearn']
)