#!/usr/bin/env python
# coding: utf-8

# Run subprocesses to prepare website to 
# git push
import subprocess

def main():
    # Clean latex of .ipynb files in ./base
    subprocess.run(["python", "./base/clean_latex.py"])
    
    # Override the files in ./base to ./content
    subprocess.run(["python", "./copy_ipynb_from_base_to_content.py"])
    
    # Convert all the relevant files in ./content to markdown in ./_build
    subprocess.run(["jupyter-book", "build", "--toc", "./_data/toc.yml", "--overwrite", "--config", "_config.yml", "./"])
    
if __name__ == '__main__':
    main()