import sys
import subprocess

def install():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "kedro-datasets[pandas]"])

if __name__ == '__main__':
    install()
