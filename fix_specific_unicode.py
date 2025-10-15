# fix_specific_unicode.py
import os

def fix_unicode_in_file(filepath):
    """Fix specific Unicode issue in Python source files"""
    print(f"Fixing Unicode issues in: {filepath}")
    
    try:
        # Try to read with different encodings
        content = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"  Successfully read with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"  ERROR: Could not read file with any encoding")
            return False
        
        # Replace problematic characters
        replacements = {
            '\u00a9': '(c)',  # Copyright symbol
            '\u2013': '-',    # En dash  
            '\u2014': '-',    # Em dash
            '\u2018': "'",    # Left single quote
            '\u2019': "'",    # Right single quote
            '\u201c': '"',    # Left double quote
            '\u201d': '"',    # Right double quote
            '\u2212': '-',    # Unicode minus sign
        }
        
        original_length = len(content)
        changes_made = 0
        
        for unicode_char, replacement in replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, replacement)
                changes_made += 1
                print(f"  Replaced {unicode_char} with {replacement}")
        
        # Write back as UTF-8
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Fixed {changes_made} Unicode issues")
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

# Fix the problematic file
files_to_fix = [
    'src/forward_model/heat_equation.py',
    'src/inference/classical_posterior.py',
]

for filepath in files_to_fix:
    if os.path.exists(filepath):
        fix_unicode_in_file(filepath)
    else:
        print(f"File not found: {filepath}")

print("\nTesting import after fix...")
try:
    from src.forward_model.heat_equation import HeatEquationSolver
    print("✅ HeatEquationSolver import successful!")
except Exception as e:
    print(f"❌ Import still failing: {e}")