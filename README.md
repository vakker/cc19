# Setup
Install the required Python packages by running:

``` sh
conda env create -f environment.yml
conda activate cc19
```

Or build the Docker image by:

``` sh
docker build -t cc19:latest .
```

# Run tune
To do a hyperparam tuning, copy the example config files from `configs` to your
`<log-dir>`, modify them accordingly and execute:

``` sh
python tune.py --logdir <log-dir> --num-samples <search-samples> --max-epochs <max-epochs>
```

You can use `-s` to run a smoke test (only 2 samples with 2 epochs) or pass in
`--subset <fraction>` to run the experiments with only `fraction*len(dataset)`
data points, i.e. 0.5 is half of the original ds.
