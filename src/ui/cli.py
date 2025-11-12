import argparse, subprocess, sys
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ADA — Book Mentor CLI")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    cmd = [sys.executable, "-m", "src.infer.answer", "--q", args.q, "--k", str(args.k)]
    subprocess.run(cmd, check=True)
