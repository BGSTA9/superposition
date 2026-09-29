# References

## The paper

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec,
Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish,
Jared Kaplan, Dario Amodei, Martin Wattenberg and Christopher Olah.
**Toy Models of Superposition.** *Transformer Circuits Thread*, 2022.

- Web article (canonical, recommended): https://transformer-circuits.pub/2022/toy_model/index.html
- arXiv: https://arxiv.org/abs/2209.10652

### BibTeX (from the paper, p. 55)

```bibtex
@article{elhage2022superposition,
  title={Toy Models of Superposition},
  author={Elhage, Nelson and Hume, Tristan and Olsson, Catherine and Schiefer, Nicholas and Henighan, Tom and Kravec, Shauna and Hatfield-Dodds, Zac and Lasenby, Robert and Drain, Dawn and Chen, Carol and Grosse, Roger and McCandlish, Sam and Kaplan, Jared and Amodei, Dario and Wattenberg, Martin and Olah, Christopher},
  year={2022},
  journal={Transformer Circuits Thread},
  note={https://transformer-circuits.pub/2022/toy_model/index.html}
}
```

## The local PDF

Page numbers throughout this repository (docstrings, docs, notebooks) refer to a 62-page print
of the web article, saved as `references/toy_models_of_superposition.pdf`. That file is
**git-ignored and must never be committed** (licensing). To get the same page numbering, print
the web article to PDF from a desktop browser (US Letter, default margins).

## Related implementations (not used as sources)

This repository is an independent implementation from the paper (SPEC §2, "Independence").
The following exist and may be consulted only to diagnose a failed replication after an honest
independent attempt; any consultation is recorded in `docs/DECISIONS.md`.

- Anthropic's archived notebook: https://github.com/anthropics/toy-models-of-superposition
- ARENA §1.5.4: https://learn.arena.education/chapter1_transformer_interp/34_superposition/
