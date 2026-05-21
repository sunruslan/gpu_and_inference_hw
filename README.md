# Homework

## Contents

- `hw1/`: implement and benchmark operations with different arithmetic intensity.
- `hw2/`: profile and optimize an autoregressive generation loop.

## Setup

```bash
sudo apt-get install -y python3-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

Use a fresh virtualenv for this repo. Reusing an older environment with extra
packages can create version conflicts with the pinned dependencies.

See the `README.md` inside each subfolder for task details, requirements, and
expected outputs.
