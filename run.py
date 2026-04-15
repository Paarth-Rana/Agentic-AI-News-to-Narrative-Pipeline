import sys
from pipeline import run_pipeline

def main():
    query = " ".join(sys.argv[1:]).strip()
    state = run_pipeline(query)

    print("Chosen:", state.get("source_title"))
    print("Wiki:", state.get("source_url"))
    print("Output folder:", state.get("out_dir"))
    print("Manifest:", state.get("manifest_path"))

if __name__ == "__main__":
    main()
