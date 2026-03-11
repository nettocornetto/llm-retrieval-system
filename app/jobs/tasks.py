from subprocess import run


def run_ingest(dataset_name: str) -> None:
    run(["python", "scripts/ingest.py", "--dataset", dataset_name], check=True)
