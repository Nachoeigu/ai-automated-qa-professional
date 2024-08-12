from setuptools import setup, find_packages

setup(
    name='ai-automated-qa-professional',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
      'langchain_openai',
      'python-dotenv',
      'langchain-community',
      'langsmith'
    ],
    author='Nachoeigu',
    author_email='ignacio.eiguren@gmail.com',
    description='This software enables you to respond to job-related questions using your personal and professional information. It can be seamlessly integrated with external applications, such as bots, to automate responses in job application forms.',
    url='https://github.com/Nachoeigu/ai-automated-qa-professional',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
