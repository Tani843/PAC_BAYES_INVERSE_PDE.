# fix_unicode_issue.py
import codecs

# Find the problematic file
file_path = "src/forward_model/heat_equation.py"

# Read with explicit encoding
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Check around line 97
print("Lines 95-100:")
for i in range(94, min(100, len(lines))):
    line = lines[i]
    # Check for non-ASCII characters
    if any(ord(c) > 127 for c in line):
        print(f"Line {i+1}: Contains non-ASCII: {repr(line)}")
    else:
        print(f"Line {i+1}: OK")

# Fix by replacing non-ASCII characters
fixed_lines = []
for line in lines:
    # Replace common Unicode issues
    fixed_line = line.replace('−', '-')  # Unicode minus
    fixed_line = fixed_line.replace('–', '-')  # En dash
    fixed_line = fixed_line.replace('—', '-')  # Em dash
    fixed_line = fixed_line.replace(''', "'")  # Smart quotes
    fixed_line = fixed_line.replace(''', "'")
    fixed_line = fixed_line.replace('"', '"')
    fixed_line = fixed_line.replace('"', '"')
    fixed_lines.append(fixed_line)

# Write back with ASCII encoding
with open(file_path, 'w', encoding='ascii', errors='replace') as f:
    f.writelines(fixed_lines)

print("Fixed Unicode issues in heat_equation.py")