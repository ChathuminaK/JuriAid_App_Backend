import os
from orchestrator.core import run_full_process

def setup_directories():
    """Create necessary directories for the project."""
    directories = ["uploads", "outputs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print(f"\n📁 Ready! Place your case files in: uploads/")
    print(f"📄 Analysis results will be saved in: outputs/")

if __name__ == "__main__":
    setup_directories()
    
    # Check if we have any files to process
    if os.path.exists("uploads") and any(f.endswith(('.pdf', '.txt')) for f in os.listdir("uploads")):
        print("\n🚀 Starting case analysis...")
        run_full_process()
    else:
        print("\n📋 No case files found. Please add PDF or TXT files to the uploads/ folder and run again.")
        print("\nTo add files:")
        print('   copy "path\\to\\your\\file.pdf" "uploads\\"')
        print('   python main.py')

