from setuptools import setup, find_packages

setup(
    name="chef-intelligence",
    version="1.0.0",
    author="Saurabh Kadam",
    description="Chef Intelligence - Method 1: Direct RAG with Keyword Search",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "openai==0.28.0",
        "pytest>=7.4.3",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
