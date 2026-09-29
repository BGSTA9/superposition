# superposition

A research-grade, independent replication of
[*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html)
(Elhage et al., 2022), with an interactive lab and mechanistic-interpretability tooling.

The repository does four things (see [SPEC.md](SPEC.md)):

1. **Replicates** the paper's experiments reproducibly, and reports every claim honestly as
   PASS, PARTIAL or FAIL ([docs/REPRODUCTION.md](docs/REPRODUCTION.md)).
2. **Provides an interactive lab**: a Streamlit app to train toy models live, probe them, and
   browse precomputed results.
3. **Provides mechanistic-interpretability tooling**: reverse-engineering trained networks,
   ablations, and sparse autoencoders scored against the known ground truth.
4. **Serves as a learning instrument**: notebooks and a study guide that follow the paper.

> **Status:** Phase 1 of 6 complete (engine, E00, E01, notebooks 00–02). Build progress is
> tracked in [PLAN.md](PLAN.md) and [docs/progress/](docs/progress/).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Everything runs on a laptop CPU; `--device auto`
uses CUDA or Apple MPS when available.

```bash
uv sync                                       # create the environment from uv.lock
uv run superposition list                     # list experiments
uv run superposition run intro_pentagon       # reproduce the paper's first figure (p. 2), ~1 min
uv run superposition figures intro_pentagon   # redraw its figures from the saved run
uv run superposition report                   # refresh the tables in docs/REPRODUCTION.md
uv run superposition app                      # launch the lab (Phase 2)
```

Developer commands: `make lint`, `make test` (fast tier), `make test-slow` (replication and
notebook tests). See [CLAUDE.md](CLAUDE.md) for the full list and conventions.

## Results gallery

**E00 · five features in two dimensions (p. 2).** As sparsity rises, the ReLU output model goes
from two dedicated dimensions, to two antipodal pairs, to a pentagon; the linear model keeps the
top two features throughout.

![E00: feature directions of the best restart at S = 0, 0.8, 0.9](docs/figures/intro_pentagon/pentagon_arrows.png)

**E01 · twenty features in five dimensions (p. 11).** WᵀW with the bias, and ‖Wᵢ‖ colored by
superposition, from dense (left) to sparse (right).

![E01: n = 20, m = 5 across sparsity](docs/figures/basic_results/weights_n20_m5.png)

## Reproduction status

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md); the table there is generated from the latest
runs by `superposition report`.

## Learning path

Start with [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md) and the notebooks in `notebooks/`, one per
paper section. [docs/PAPER_MAP.md](docs/PAPER_MAP.md) maps every paper section to code,
experiments, notebooks and app pages; [docs/GLOSSARY.md](docs/GLOSSARY.md) defines the terms.

## How to cite

Please cite the paper:

```bibtex
@article{elhage2022superposition,
  title={Toy Models of Superposition},
  author={Elhage, Nelson and Hume, Tristan and Olsson, Catherine and Schiefer, Nicholas and Henighan, Tom and Kravec, Shauna and Hatfield-Dodds, Zac and Lasenby, Robert and Drain, Dawn and Chen, Carol and Grosse, Roger and McCandlish, Sam and Kaplan, Jared and Amodei, Dario and Wattenberg, Martin and Olah, Christopher},
  year={2022},
  journal={Transformer Circuits Thread},
  note={https://transformer-circuits.pub/2022/toy_model/index.html}
}
```

To cite this repository, use [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this
repository" button for it).

## Related implementations

This is an independent implementation written from the paper. Two reference implementations
exist; they were not used as sources (see SPEC §2 and [docs/DECISIONS.md](docs/DECISIONS.md)
for any diagnostic consultation):

- Anthropic's archived notebook: https://github.com/anthropics/toy-models-of-superposition
- ARENA §1.5.4: https://learn.arena.education/chapter1_transformer_interp/34_superposition/

## License

MIT; see [LICENSE](LICENSE). The paper itself is not redistributed here.
