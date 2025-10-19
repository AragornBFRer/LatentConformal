Default configs live in `experiments/configs/gmm_em.yaml`.

## Running the full sweep (100 seeds)

```bash
# Linux / macOS
chmod +x run.sh
./run.sh

# Windows PowerShell
python main.py --config experiments/configs/gmm_em.yaml
```

## Running a single trial (edit the config to limit seeds)

Create a lightweight override, for example `experiments/configs/single.yaml`:

```yaml
global:
	seeds:
		values: [1]
```

Then execute either helper:

```bash
# Linux / macOS
./run.sh experiments/configs/single.yaml

# Windows PowerShell
python main.py --config experiments/configs/single.yaml
```
