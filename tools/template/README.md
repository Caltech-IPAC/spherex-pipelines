# Pipeline Module Template

## Instructions

Execute the `populate.py` script. It will generate pre-populated pipeline module files, their corresponding tests and a sorted `__init__.py` file under `./products/modules/` with all the relative imports that.

## Mechanism

This script will read the the content the following files:

1. `table_of_modules.csv`: The table of pipeline modules with the following columns:
    1. ssdc_id
    2. ticket_id
    3. level_id
    4. full_name
    5. short_name (if empty, full_name will be used)
2. `module_name.py`: The pipeline module template with placeholders
3. `test_module_name.py`: The pipeline module test template with placeholders

The `populate.py` script will replace the following placeholders with an appropriate strings and writes the file with the correctly auto-generated name:

1. `'Lx.xx.'` → `level_tag`)
2. `'SSDC-MS-xxx'` → `ssdc_tag`)
3. `'ModuleName'` → `ClassName`)
4. `'Full Module-Name'` → `full_name`)
5. `'module-name'` → `DefaultName`)

## Style Example

The following example highlights the occurrence of the placeholders in the pipeline module template.
It also shows the chosen style for each placeholder.

Module: L2.1. Correct Non-Linearity (SSDC-MS-010)

1. file_name: module_name.py → correct_non_linearity.py
2. class_name: ModuleName → CorrectNonLinearity
    1. ModuleNameConfig
    2. ModuleNameTask
    3. `__all__`
3. _DefaultName: module-name → correct-non-linearity
    1. Task
4. level_tag: Lx.xx. → L2.01.
    1. Top comment
    2. Task
5. full_name: Full Module-Name → Correct Non-Linearity
    1. Top comment
    2. Task
6. ssdc_tag: SSDC-MS-xxx → SSDC-MS-010
    1. Top comment
    2. Task
