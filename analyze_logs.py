import os
import glob
from tbparse import SummaryReader

def extract_final_values(log_dir):
    """
    Extract final values from TensorBoard logs using tbparse.
    """
    # Find all log directories
    log_dirs = glob.glob(os.path.join(log_dir, '*'))

    results = {}

    for log_subdir in log_dirs:
        if os.path.isdir(log_subdir):
            run_name = os.path.basename(log_subdir)

            try:
                reader = SummaryReader(log_subdir, pivot=True)
                df = reader.scalars

                if not df.empty:
                    results[run_name] = {}
                    for tag in df.columns:
                        if tag != 'step':
                            final_value = df[tag].iloc[-1]
                            results[run_name][tag] = final_value
            except Exception as e:
                print(f"Error reading {log_subdir}: {e}")

    return results

if __name__ == "__main__":
    log_dir = "logs"
    results = extract_final_values(log_dir)

    # Print results
    for run, tags in results.items():
        print(f"Run: {run}")
        for tag, value in tags.items():
            print(f"  {tag}: {value:.2f}")
        print()