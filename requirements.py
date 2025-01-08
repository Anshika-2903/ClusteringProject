# requirements.py
def install_requirements():
    import subprocess
    requirements = [
        "numpy", 
        "pandas", 
        "scikit-learn", 
        "matplotlib", 
        "seaborn"
    ]
    for package in requirements:
        subprocess.check_call(["pip", "install", package])
