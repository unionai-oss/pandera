# Airspeed Velocity 

`pandera`'s performance benchmarks over time can be [viewed on this airspeed-velocity dashboard](https://pandera-dev.github.io/pandera-asv-logs/).

The [config](https://github.com/pandera-dev/pandera-asv-logs/tree/master/asv_bench/asv.conf.json) and [results files](https://github.com/pandera-dev/pandera-asv-logs/tree/master/results) files are tracked in the [pandera-asv-logs](https://github.com/pandera-dev/pandera-asv-logs) repo to avoid build files in the main repo.

The [benchmarks](https://github.com/pandera-dev/pandera/tree/master/benchmarks/) are tracked in the main [pandera repo](https://github.com/pandera-dev/pandera).

## Running `asv`

Ensure both the `pandera` and `pandera-asv-logs` repos are checked out to the same parent directory.

From the `pandera-asv-logs` repo, run:
```
asv run ALL --config asv_bench/asv.conf.json
```

## Publishing results:

To build the html and preview the results:
```
asv publish --config asv_bench/asv.conf.json
asv preview --config asv_bench/asv.conf.json
```

The `.json` results files are committed or PR'd into the master branch of `pandera-asv-logs`.

The published html is pushed directly to the gh-pages branch of `pandera-asv-logs` by running:

```
asv gh-pages --rewrite --config asv_bench/asv.conf.json
```

The `--rewrite` flag overwrites the existing `gh-pages`, avoiding duplication of data.

The `asv` docs are [here](https://asv.readthedocs.io/en/stable/index.html).
