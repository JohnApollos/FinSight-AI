import os
import sys

def main():
    print("==================================================")
    # Ensure current directory is in Python path so imports work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(current_dir)
    sys.path.insert(0, app_dir)
    sys.path.insert(0, os.path.dirname(app_dir))

    print("Step 1: Downloading NYU Damodaran Sector Ratios...")
    try:
        import damodaran_downloader
        damodaran_downloader.run_pipeline()
    except Exception as e:
        print(f"Error running Damodaran downloader: {e}")

    print("\nStep 2: Processing World Bank and IFC Industry Benchmarks...")
    try:
        import process_benchmarks
        process_benchmarks.run_aggregator()
    except Exception as e:
        print(f"Error running benchmarks aggregator: {e}")

    print("\nStep 3: Fetching SEC EDGAR Reference Filings (Outlier Baseline)...")
    try:
        import sec_downloader
        # We run a small fetch to prevent long development wait times.
        # User can adjust target count per SIC in the sec_downloader script.
        sec_downloader.download_and_parse_sec_data(target_count_per_sic=5)
    except Exception as e:
        print(f"Error running SEC downloader (likely rate limits/offline): {e}")
        print("Will fallback to synthetic reference data generation in Step 4.")

    print("\nStep 4: Training Anomaly Detection Engine (Isolation Forest)...")
    try:
        import train_model
        train_model.train_isolation_forest()
    except Exception as e:
        print(f"Critical error during Isolation Forest training: {e}")
        sys.exit(1)

    print("==================================================")
    print("BOOTSTRAP COMPLETE: Reference data prepared and Isolation Forest trained!")
    print("==================================================")

if __name__ == "__main__":
    main()
