import os
from .orchestrator.core import run_full_process, UPLOADS_DIR, OUTPUTS_DIR  # ← relative import

def setup_directories():
    """Create necessary directories for the project."""
    for directory in [UPLOADS_DIR, OUTPUTS_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    print(f"\n📁 Ready! Place your case files in: {UPLOADS_DIR}")
    print(f"📄 Analysis results will be saved in: {OUTPUTS_DIR}")

if __name__ == "__main__":
    setup_directories()
    if os.path.exists(UPLOADS_DIR) and any(f.endswith(('.pdf', '.txt')) for f in os.listdir(UPLOADS_DIR)):
        print("\n🚀 Starting case analysis...")
        run_full_process()
    else:
        print("\n📋 No case files found. Please add PDF or TXT files and run again.")
        print("\nTo add files:")
        print(f'   copy "C:\\path\\to\\your\\file.pdf" "{UPLOADS_DIR}\\"')
        print('   python -m orchestratorc.main')

