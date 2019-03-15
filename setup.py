from distutils.core import setup

setup(name='HTMRL',
      version='0.1',
      description='Python3 HTM',
      author='Jakob Struye',
      author_email='jakob.struye@uantwerpen.be',
      packages=['HTMRL'],
      install_requires=[
         'numpy==1.16.2',
         'matplotlib==3.0.3',
         'pyyaml==3.13',
         'psutil==5.5.0'
      ]
     )
