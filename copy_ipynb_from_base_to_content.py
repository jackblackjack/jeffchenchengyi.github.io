#!/usr/bin/env python
# coding: utf-8

# Copy all `*.ipynb` files from base to content 
import fnmatch
from os.path import isdir, join
from shutil import copytree, rmtree
import os

def include_patterns(*patterns):
    """ Function that can be used as shutil.copytree() ignore parameter that
    determines which files *not* to ignore, the inverse of "normal" usage.

    This is a factory function that creates a function which can be used as a
    callable for copytree()'s ignore argument, *not* ignoring files that match
    any of the glob-style patterns provided.

    ‛patterns’ are a sequence of pattern strings used to identify the files to
    include when copying the directory tree.

    Example usage:

        copytree(src_directory, dst_directory,
                 ignore=include_patterns('*.sldasm', '*.sldprt'))
    """
    def _ignore_patterns(path, all_names):
        # Determine names which match one or more patterns (that shouldn't be
        # ignored).
        keep = (name for pattern in patterns
                        for name in fnmatch.filter(all_names, pattern))
        # Ignore file names which *didn't* match any of the patterns given that
        # aren't directory names and those directories that contain 'data'
        dir_names = (name for name in all_names if isdir(join(path, name)) and path.split('/')[-1] != 'data')
        return set(all_names) - set(keep) - set(dir_names)

    return _ignore_patterns

src = './base/'
dst = './content/'

# Make sure the destination folder does not exist.
if os.path.exists(dst) and os.path.isdir(dst):
    print('Removing existing directory "{}"'.format(dst))
    rmtree(dst, ignore_errors=False)

print('Copying files from {} to {} ...'.format(src, dst))
copytree(src, dst, ignore=include_patterns('*.ipynb', '*.md', '*.html', '*.png', '*jpg'))

print('Completed!')