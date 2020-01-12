from setuptools import setup
import io


setup(name='promid',
      description='PromID: A deep learning-based tool to identify promoters',
      long_description=io.open('README.md', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      version='v1.01',
      author='Ramzan Umarov',
      author_email='umarov256@gmail.com',
      license='MIT',
      url='https://github.com/PromStudy/PromID',
      download_url='https://github.com/PromStudy/PromID/archive/v1.01.tar.gz', 
      packages=['promid'],
      install_requires=['numpy>=1.14.0'],
      extras_require={'cpu': ['tensorflow>=1.7.0'],
                      'gpu': ['tensorflow-gpu>=1.7.0']},
      package_data={'promid': [  'models/model_pos/saved_model.pbtxt',
                                 'models/model_pos/variables/variables.data-00000-of-00001',
                                 'models/model_pos/variables/variables.index',
                                 'models/model_scan/saved_model.pbtxt',
                                 'models/model_scan/variables/variables.data-00000-of-00001',
                                 'models/model_scan/variables/variables.index',]},
      entry_points={'console_scripts': ['promid=promid.__main__:main']})
