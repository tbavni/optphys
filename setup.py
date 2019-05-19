from setuptools import setup, find_packages

def get_github_url(package_name: str, user_name: str):
    """URL for one of my GitHub repos for install_requires.

    See PEP 508.
    """
    # Will keep ssh version for reference.
    # '%s @ git+ssh://git@github.com/draustin/%s.git'%(name, name)
    return '%s @ git+https://github.com/%s/%s.git'%(package_name, user_name, package_name)

# I attempted to separate test and install dependencies but couldn't figure it out (in 10 minutes). Keeping them
# toegether for now - pytest is lightweight.
setup(name="optphys", version=0.1, description="Optical physics toolbox", author='Dane Austin',
      author_email='dane_austin@fastmail.com.au', url='https://github.com/draustin/optphys', license='BSD',
      packages=find_packages(),
      install_requires=['numpy', 'pytest', 'scipy', 'pyyaml', 'periodictable', 'pytest-qt', get_github_url('mathx', 'draustin'),
                        get_github_url('pyqtgraph_extensions', 'draustin')], python_requires='>=3.4')
