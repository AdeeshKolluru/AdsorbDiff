from setuptools import find_packages, setup

setup(
    name="adsorbdiff",
    version="0.0.1",
    description="AdsorbDiff: Adsorbate Placement via Conditional Denoising Diffusion",
    license="MIT",
    author="Adeesh Kolluru",
    author_email = "kolluru.adeesh@gmail.com",
    url="https://github.com/AdeeshKolluru/AdsorbDiff",
    packages=find_packages(),
    include_package_data=True,
)
