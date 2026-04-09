from pathlib import Path

# Create all necessary __init__.py files
init_files = [
    'src/__init__.py',
    'src/data/__init__.py',
    'src/models/__init__.py',
    'src/evaluation/__init__.py',
    'src/analytics/__init__.py',
    'src/utils/__init__.py',
]

print("Creating __init__.py files...")
for init_file in init_files:
    file_path = Path(init_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()
    print(f"✓ Created: {init_file}")

print("\n✓ Setup complete!")