# FLUX Klein INT8 evidence archive

This bundle preserves the existing local preparation and smoke records for the
distilled Klein 4B reference. It contains no model weights or tokenizer cache.
The original files remain available in the ignored local `models/` and `outputs/`
directories. Source and quantized weights must be prepared separately using the
pinned configuration and scripts described in [the reference report](../../flux-klein-int8.md).

- `source-manifest.json` records the pinned upstream files, hashes and reuse.
- `protected-quantization-manifest.json`, export markers and validation record
  describe the selected protected INT8 export.
- `rejected-quantization-manifest.json` and `rejected-validation.json` preserve
  the rejected all-linear INT8 export's identity and failure status.
- `runs/` retains the original JSON records and PNG outputs for BF16, failed
  all-linear INT8 and successful protected INT8 smokes. Incomplete mixed-precision
  attempts retain their `attempt.json` without an invented output.
- `SHA256.json` lists the copied evidence files' hashes. Records and outputs were
  copied byte-for-byte; this README documents packaging rather than a new run.
- `LICENSE.md` retains the upstream license associated with the pinned source.

These exposed mug examples are compatibility/diagnostic evidence. They do not
establish semantic acceptance, general precision parity or a performance benchmark.
The failed texture outputs are retained intentionally.
