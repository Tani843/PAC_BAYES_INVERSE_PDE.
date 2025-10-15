# fix_all_unicode.py
import os
import glob

def fix_unicode_in_file(filepath):
    """Fix Unicode issues in Python source files"""
    try:
        # Try to read with different encodings
        content = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"❌ Could not read: {filepath}")
            return False
        
        # Replace problematic characters
        replacements = {
            '\u00a9': '(c)',     # Copyright symbol (byte 0xa9)
            '\u00ba': 'o',       # Masculine ordinal indicator (byte 0xba)
            '\u2013': '-',       # En dash  
            '\u2014': '-',       # Em dash
            '\u2018': "'",       # Left single quote
            '\u2019': "'",       # Right single quote
            '\u201c': '"',       # Left double quote
            '\u201d': '"',       # Right double quote
            '\u2212': '-',       # Unicode minus sign
            '\u00ae': '(R)',     # Registered trademark
            '\u2122': '(TM)',    # Trademark
        }
        
        changes_made = 0
        for unicode_char, replacement in replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, replacement)
                changes_made += 1
        
        # Also remove any other non-ASCII characters by replacing with ASCII equivalent
        content_bytes = content.encode('ascii', errors='replace')
        content = content_bytes.decode('ascii')
        
        # Write back as UTF-8
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if changes_made > 0:
            print(f"✅ Fixed {changes_made} issues in: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

# Find all Python files in src/
python_files = glob.glob('src/**/*.py', recursive=True)

print(f"Found {len(python_files)} Python files to check")
print("Fixing Unicode issues...")

fixed_count = 0
for filepath in python_files:
    if fix_unicode_in_file(filepath):
        fixed_count += 1

print(f"\n✅ Processed {fixed_count}/{len(python_files)} files successfully")

# Test critical imports
print("\nTesting critical imports...")
test_imports = [
    ('src.forward_model.heat_equation', 'HeatEquationSolver'),
    ('src.data.data_generator', 'DataGenerator'),
    ('src.inference.classical_posterior', 'ClassicalPosterior'),
]

for module, class_name in test_imports:
    try:
        exec(f"from {module} import {class_name}")
        print(f"✅ {module}.{class_name}")
    except Exception as e:
        print(f"❌ {module}.{class_name}: {e}")