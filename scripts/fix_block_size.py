import sys
import re
from pathlib import Path

def patch_block_size():
    # Fix for: https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes/issues/28
    
    # If a command line argument is provided, use that as the file path, 
    # otherwise default to the relative path in the source tree.
    target = sys.argv[1] if len(sys.argv) > 1 else 'vllm/v1/attention/backend.py'
    p = Path(target)
    
    # Some installations might have it in site-packages, search for it if absolute path isn't given
    if not p.exists():
        # Let's see if we can find it in site-packages if we just run it inside a container
        try:
            import vllm
            p = Path(vllm.__path__[0]) / 'v1' / 'attention' / 'backend.py'
        except ImportError:
            pass
            
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    txt, count = re.subn(
        r'(\s*valid_sizes\s*=\s*get_args\(BlockSize\)\s*\n\s*if\s*block_size\s*not\s*in\s*valid_sizes:\s*\n\s*return\s*False\s*\n*)',
        '\n',
        p.read_text(),
        flags=re.MULTILINE
    )
    
    if count > 0:
        p.write_text(txt)
        print(f" -> Patched {p} (removed {count} block size checks)")
    else:
        print(f" -> Warning: Block size check not found in {p}. Regex might be outdated or already patched.")
        # Try a simpler replace just in case get_args is imported differently
        txt, count2 = re.subn(
            r'(\s*if\s*block_size\s*not\s*in\s*valid_sizes:\s*\n\s*return\s*False\s*\n*)',
            '\n',
            txt,
            flags=re.MULTILINE
        )
        if count2 > 0:
             p.write_text(txt)
             print(f" -> Patched {p} via fallback regex")

if __name__ == "__main__":
    patch_block_size()
