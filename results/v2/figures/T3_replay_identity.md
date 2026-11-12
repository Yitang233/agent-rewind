# T3 replay identity (text-identical decoded calls)

| source | arm | calls compared | identical | mismatches |
|---|---|---|---|---|
| E5v2 full cells, 7B held-out (runs/v2/cells.jsonl) | prefix | 81 | 81 | 0 |
| E5v2 full cells, 7B held-out (runs/v2/cells.jsonl) | resync | 81 | 81 | 0 |
| D3 held-out call-level, 7b (runs/v2/D3/heldout.json) | K=64 margin=0.0 tie=local | 21 | 18 | 3 |
| D3 held-out call-level, 7b (runs/v2/D3/heldout.json) | K=64 margin=0.1 tie=local | 21 | 21 | 0 |
| D3 held-out call-level, 7b (runs/v2/D3/heldout.json) | K=32 margin=0.3 tie=full | 21 | 21 | 0 |
| D3 held-out call-level, 7b (runs/v2/D3/heldout.json) | K=32 margin=0.3 tie=local | 21 | 21 | 0 |
| D3 held-out call-level, 1.5b (runs/v2/D3/heldout.json) | K=64 margin=0.0 tie=local | 18 | 17 | 1 |
| D3 held-out call-level, 1.5b (runs/v2/D3/heldout.json) | K=64 margin=0.1 tie=local | 18 | 18 | 0 |
| D3 held-out call-level, 1.5b (runs/v2/D3/heldout.json) | K=32 margin=0.3 tie=full | 18 | 18 | 0 |
| D3 held-out call-level, 1.5b (runs/v2/D3/heldout.json) | K=32 margin=0.3 tie=local | 18 | 18 | 0 |
