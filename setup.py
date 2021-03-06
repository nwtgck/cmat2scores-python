
# (from: https://github.com/masaponto/Python-MLP/blob/master/setup.py)
# (from: https://qiita.com/masashi127/items/5bfcba5cad8e82958844)
# (from: https://qiita.com/hotoku/items/4789533f5e497f3dc6e0)

from setuptools import setup, find_packages
import sys
import toml

def name_ver_join(name, ver):
    if ver == "*":
        return name
    else:
        return name + ver

with open("Pipfile") as f:
  pipfile_dict = toml.load(f)

install_requires = [name_ver_join(name, ver) for name, ver in pipfile_dict['packages'].items()]

sys.path.append('./tests')

setup(
    name='cmat2scores',
    version='1.0.0',
    description='Calculate accuracy, precision, recall f-measure from confusion matrix',
    author='Ryo Ota',
    author_email='nwtgck@gmail.com',
    install_requires=install_requires,
    py_modules=["cmat2scores"],
    packages=find_packages(),
    test_suite='tests'
)