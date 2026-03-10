from setuptools import setup, find_packages

setup(
    name="embodied-fly",
    version="0.1.0",
    description="Driving a virtual fruit fly with a spiking neural CPG wired from Drosophila motor circuit connectivity",
    author="Embodied Fly Contributors",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "flygym>=0.2.0",
        "mujoco>=3.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "mediapy>=1.1.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0"],
    },
)
