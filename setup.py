# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mustard-gui',
    packages=find_packages(),
    version='1.0.4',
    license='gpl',
    description='MUlti STream Agnostic Representation Dataplayer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Event-driven Perception for Robotics group at Istituto Italiano di Tecnologia: Massimiliano Iacono',
    author_email='massimiliano.iacono@iit.it',
    url='https://github.com/event-driven-robotics/mustard',
    keywords=['event', 'event camera', 'event-based', 'event-driven', 'spike', 'dvs', 'dynamic vision sensor',
              'neuromorphic', 'aer', 'address-event representation' 'spiking neural network', 'davis', 'atis', 'celex'],
    install_requires=['kivy>=2.0.0',
                      'matplotlib',
                      'numpy',
                      'tqdm',
                      'bimvee>=1.0.15'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    include_package_data=True,
    scripts=['mustard/mustard']
)
