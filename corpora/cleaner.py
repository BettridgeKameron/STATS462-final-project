import sys
import string

def clean_file(filename):
    try:
        name, ext = filename.rsplit('.', 1)
        outname = f"{name}_cleaned.{ext}"
        with open(filename, 'r', encoding='utf-8', errors='ignore') as infile:
            with open(outname, 'w', encoding='ascii') as outfile:
                for line in infile:
                    outfile.write(''.join(char for char in line if char in string.printable))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cleaner.py <filename>")
        sys.exit(1)
    
    clean_file(sys.argv[1])
