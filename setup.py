from setuptools import setup, find_packages

setup(
    name="geo_customer_segmentation",
    version="1.0.0",
    description="Location-Based Customer Segmentation for Targeted Marketing",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy", "pandas", "geopandas", "h3",
        "scikit-learn", "mlflow", "fastapi", "pydantic",
    ],
)
