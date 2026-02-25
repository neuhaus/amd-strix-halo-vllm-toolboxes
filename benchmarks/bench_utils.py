import subprocess
import tempfile

def run_dialog(args):
    """Runs dialog and returns stderr (selection line). Returns None if user cancelled."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            # We don't trap stdout since dialog renders to TTY and writes choice to stderr
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None # User cancelled/pressed ESC
