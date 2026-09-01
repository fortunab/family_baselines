"""
Automated Synapse PolypGen Downloader (syn26376615).
Downloads the multi-center dataset from Synapse.org using synapseclient:
https://www.synapse.org/Synapse:syn26376615/wiki/613312
"""

import os
import sys
import argparse
from pathlib import Path


def download_polypgen_from_synapse(dest_dir: Path, auth_token: str = None):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("      SYNAPSE POLYPGEN DATASET DOWNLOADER (syn26376615)")
    print("="*80)
    print(f" Target Directory : {dest_dir.resolve()}")
    print(" Synapse Entity   : syn26376615")
    print("="*80)

    try:
        import synapseclient
        import synapseutils

        syn = synapseclient.Synapse()

        # Check for token in args, environment, or config
        token = auth_token or os.environ.get("SYNAPSE_AUTH_TOKEN")
        if token:
            print("[Synapse] Logging in with Personal Access Token (PAT)...")
            syn.login(authToken=token)
        else:
            print("[Synapse] Attempting login with cached credentials (~/.synapseConfig)...")
            try:
                syn.login()
            except Exception:
                print("\n[Notice] No Synapse credentials found.")
                print("To get a free Synapse Personal Access Token (PAT):")
                print("  1. Sign in or create a free account at https://www.synapse.org")
                print("  2. Go to: Account Settings -> Personal Access Tokens -> Create New Token")
                print("  3. Run: python download_synapse_polypgen.py --auth-token YOUR_SYNAPSE_TOKEN")
                print("     or set environment variable: export SYNAPSE_AUTH_TOKEN=YOUR_TOKEN\n")
                return False

        print("[Synapse] Downloading all multi-center files from syn26376615 (this may take a while)...")
        synapseutils.syncFromSynapse(syn, 'syn26376615', path=str(dest_dir))
        print(f"\n[Synapse] Successfully downloaded and synced PolypGen into: {dest_dir.resolve()}")
        return True

    except ImportError:
        print("[Notice] 'synapseclient' is not installed.")
        print("To install: pip install synapseclient")
        print("Then run: python download_synapse_polypgen.py --auth-token YOUR_TOKEN")
        return False
    except Exception as e:
        print(f"[Error] Synapse download returned: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download PolypGen from Synapse (syn26376615)")
    parser.add_argument("--auth-token", type=str, default=None, help="Synapse Personal Access Token (PAT)")
    parser.add_argument("--dest-dir", type=str, default="./data", help="Destination folder for data")
    args = parser.parse_args()

    dest = Path(__file__).parent / args.dest_dir
    download_polypgen_from_synapse(dest, auth_token=args.auth_token)


if __name__ == "__main__":
    main()
