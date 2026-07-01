#! /usr/bin/env python

import re
import pandas as pd
from pathlib import Path

here = Path(__file__).parent


def replace(content):
    content = content.replace('ssdc_pipe_level = 0', f'ssdc_pipe_level = {level:d}')
    content = content.replace('Lx', f'L{level:d}')
    content = content.replace('SSDC-MS-xxx', ssdc_tag)
    content = content.replace('ModuleName', ClassName)
    content = content.replace('Full Module-Name', full_name)
    content = content.replace('module-name', DefaultName)
    return content


# > read table of modules
df = pd.read_csv(here / 'table_of_modules.csv')

# If no short name, set it to full name
df['short_name'].fillna(df['full_name'], inplace=True)

# collet a list of relative imports to be written in an __init__.py
imports = []

for i in df.index:
    # > get data
    level = df.at[i, 'level']
    ssdc_id = df.at[i, 'ssdc_id']
    full_name = df.at[i, 'full_name']
    short_name = df.at[i, 'short_name']
    full_comp = re.split(' |-', full_name)
    short_comp = re.split(' |-', short_name)

    # > construct names
    ClassName = ''.join(short_comp)
    DefaultName = '-'.join(short_comp).lower()
    fileName = '_'.join(short_comp).lower()
    ssdc_tag = f'SSDC-MS-{ssdc_id:03d}'

    # > read template module
    with open(here / 'module_name.py', mode='r') as f:
        content = f.read()

    # > read template module test
    with open(here / 'test_module_name.py', mode='r') as f:
        test_content = f.read()

    # > populate placeholders
    content = replace(content)
    test_content = replace(test_content)

    # > Append an import statement
    imports.append(f'from .{fileName} import *')

    # > write under products > modules
    module_dir_path = here / 'products/modules/'
    test_dir_path = here / 'products/tests/'

    module_dir_path.mkdir(exist_ok=True, parents=True)
    test_dir_path.mkdir(exist_ok=True, parents=True)

    file_path = module_dir_path / (fileName + '.py')
    test_file_path = test_dir_path / ('test_' + fileName + '.py')

    with file_path.open('w') as f:
        f.write(content)

    with test_file_path.open('w') as f:
        f.write(test_content)

# > Write the __init__.py
imports.sort()
init_path = module_dir_path / '__init__.py'
init_content = '\n'.join(imports)

with init_path.open('w') as f:
    f.write(init_content)
