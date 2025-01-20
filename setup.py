from setuptools import setup, find_packages

setup(
    name="pystallment",  # Name deines Pakets
    version="0.1.0",  # Versionsnummer
    author="CayOest",
    description="Python library for installment options",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CayOest/pystallment",  # URL zu deinem Repository
    packages=find_packages(),  # Sucht automatisch alle Pakete im Verzeichnis
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Mindest-Python-Version
    install_requires=[
        # Hier kommen die Abhängigkeiten deines Projekts rein
        "numpy",
        "pytest",
        "scipy",
        "matplotlib"
    ],
    extras_require={
        "dev": ["pytest", "flake8"],  # Zusätzliche Abhängigkeiten für Entwicklung
    },
    entry_points={
        "console_scripts": [
            # Hier kannst du CLI-Befehle definieren
            # "command-name = modulname:funktion"
        ],
    },
)
